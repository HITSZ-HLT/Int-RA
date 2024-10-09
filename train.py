import os
import time
import json
import logging
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.distributed.nn
import torch.distributed as dist
import torch.nn.functional as F
import sys,os
from params import is_DDIM, is_att, is_matrix
from cn_clip.training.params import parse_args
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
args = parse_args()
from cn_clip.clip.model import convert_state_dict
from data import get_all_features
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

def is_master(args):
    return args.rank == 0

intent2id = {}
id2intent = {}

def compute_and_plot_pr_curve(ground_truth, scores):
    precision, recall, _ = precision_recall_curve(ground_truth, scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('1.jpg')
    exit()

def get_loss(Flag, model,classifier_model, images, texts, sent_texts, keywords, speaker,  loss_img, loss_txt, number,
             memory_dict, speaker_list,comet,intent_ids,
             k_gesture,k_posture,k_facial,k_verbal,
             imgid2image,imgid2kges,imgid2kpos,imgid2kface,imgid2kver,intent2token,id2intent,imgid2intent,
             all_attn_features, all_intent_features,
             args, accum_image_features=None, accum_text_features=None,
             accum_idx=-1, teacher_model=None, teacher_accum_image_features=None):
    if args.accum_freq == 1:
        if Flag:
            image_features, text_features, logit_scale, style_loss, \
            all_attn_features, all_intent_features = model(Flag, images,  text=texts, sent_text=None,
                                                           keywords=keywords,
                                                           speaker=speaker, number=number, mask_ratio=args.mask_ratio,
                                                           comet=comet,
                                                           k_gesture=k_gesture, k_posture=k_posture, k_facial=k_facial,
                                                           k_verbal=k_verbal, imgid2image=imgid2image,
                                                           imgid2kges=imgid2kges, imgid2kpos=imgid2kpos,
                                                           imgid2kface=imgid2kface, imgid2kver=imgid2kver,
                                                           intent2token=intent2token, id2intent=id2intent)
        else:
            image_features, text_features, logit_scale, style_loss = model(Flag, images,  text=texts, sent_text=sent_texts,
                                                           keywords=keywords,
                                                           speaker=speaker, number=number, mask_ratio=args.mask_ratio,
                                                           comet=comet,
                                                           k_gesture=k_gesture, k_posture=k_posture, k_facial=k_facial,
                                                           k_verbal=k_verbal, imgid2image=imgid2image,
                                                           imgid2kges=imgid2kges, imgid2kpos=imgid2kpos,
                                                           imgid2kface=imgid2kface, imgid2kver=imgid2kver,
                                                           intent2token=intent2token, id2intent=id2intent)
        intent_prob, predict_label = classifier_model(text_features)
        predict_label = predict_label.tolist()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        intent_label =[int(label) for label in intent_ids]

        intent_ids = torch.tensor([int(label) for label in intent_ids])
        loss = criterion(intent_prob.cpu(), intent_ids.cpu())

        # intent_tok
        attn_list, img_list = [], []
        intent_list = []
        predict_list = []
        for k,v in all_attn_features.items():
            attn_list.append(v[0])
            img_list.append(k)

        bz_loss = 0
        real_intent_list = []
        for each_label in intent_label:
            for j in img_list:
                # print('j:',j, each_label)
                if int(imgid2intent[j][0]) == each_label:
                    real_intent_list.append(1)
                else:
                    real_intent_list.append(0)

        top_1_acc,top_3_acc,top_5_acc = 0,0,0
        # all_intent_list = []
        all_predict_list = []
        bz = 0
        print('text_features:',text_features.size())

        for each in text_features:

            index_list = get_images_similarity(each, attn_list, top_K=1)
            all_predict_list+=index_list
            index_top_1_list = get_top_N_images(each, attn_list, top_K=1)
            index_top_3_list = get_top_N_images(each, attn_list, top_K=3)
            index_top_5_list = get_top_N_images(each, attn_list, top_K=5)
            # print(intent_label[bz])
            sticker_intentid = []
            for j in index_top_5_list:
                sticker_intentid.append(int(imgid2intent[img_list[j]][0]))

            if intent_label[bz] in sticker_intentid[:1]:
                top_1_acc+=1
            if intent_label[bz] in sticker_intentid[:3]:
                top_3_acc += 1
            if intent_label[bz] in sticker_intentid[:5]:
                top_5_acc += 1
            # print(index_list)

            predict_list.append(torch.tensor(index_list))
            temp = []
            for j in img_list:
                print(j,int(imgid2intent[j][0]),predict_label[bz])
                # exit()
                if int(imgid2intent[j][0]) == predict_label[bz]:
                    temp.append(1)
                else:
                    temp.append(0)
            intent_list.append(torch.tensor(temp))
            bz +=1

        print('intent_list:',intent_list)
        print('all_predict_list:',all_predict_list)

        with open('intent_true.txt', 'a') as file:
            for item in real_intent_list:
                file.write(f"{item}\n")
        with open('intent_predict.txt', 'a') as file:
            for item in all_predict_list:
                file.write(f"{item}\n")

        ground_truth = np.array(real_intent_list)
        # 模型返回的分数，表示每个图像的相关性得分
        scores = np.array(all_predict_list)

        target = torch.stack(intent_list,dim=0)
        output = torch.stack(predict_list,dim=0)
        criterion = nn.MultiLabelMarginLoss()
        print(target.size(), output.size())
        hinge_loss = criterion(output, target)
        print(hinge_loss)

        if args.distllation:
            with torch.no_grad():
                # different teacher model has different output
                output = teacher_model.module.get_feature(images)
                if (isinstance(output, tuple)):
                    teacher_image_features = output[0]
                else:
                    teacher_image_features = output
    else:
        assert accum_image_features and accum_text_features and accum_idx != -1
        print('here')
        exit()
        chunk_image_features, chunk_text_features, \
        logit_scale = model(images, texts, real_image, args.mask_ratio)

        if args.distllation:
            with torch.no_grad():
                # different teacher model has different output
                output = teacher_model.module.get_feature(images)
                if (isinstance(output, tuple)):
                    teacher_chunk_image_features = output[0]
                else:
                    teacher_chunk_image_features = output
            teacher_image_features = torch.cat(
                teacher_accum_image_features[:accum_idx] + [
                    teacher_chunk_image_features] + teacher_accum_image_features[accum_idx + 1:])

        image_features = torch.cat(
            accum_image_features[:accum_idx] + [chunk_image_features] + accum_image_features[accum_idx + 1:])
        text_features = torch.cat(
            accum_text_features[:accum_idx] + [chunk_text_features] + accum_text_features[accum_idx + 1:])
    logit_scale = logit_scale.mean()
    if args.aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        if args.gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)

            if args.distllation:
                all_teacher_image_features = torch.cat(torch.distributed.nn.all_gather(teacher_image_features), dim=0)
        else:
            gathered_image_features = [
                torch.zeros_like(image_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]

            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)

            all_image_features = torch.cat(
                [image_features]
                + gathered_image_features[:rank]
                + gathered_image_features[rank + 1:]
            )
            all_text_features = torch.cat(
                [text_features]
                + gathered_text_features[:rank]
                + gathered_text_features[rank + 1:]
            )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()

        if args.distllation:
            gathered_teacher_image_features = [
                torch.zeros_like(teacher_image_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_teacher_image_features, teacher_image_features)
            all_teacher_image_features = torch.cat(
                [teacher_image_features]
                + gathered_teacher_image_features[:rank]
                + gathered_teacher_image_features[rank + 1:]
            )
            kd_loss = cosineSimilarityLoss(all_teacher_image_features, all_image_features)

    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        if args.distllation:
            kd_loss = cosineSimilarityLoss(teacher_image_features, image_features)

    ground_truth = torch.arange(len(logits_per_image)).long()
    ground_truth = ground_truth.cuda(args.local_device_rank, non_blocking=True)


    if is_DDIM:
        total_loss = (style_loss +
                      loss_img(logits_per_image, ground_truth)
                      + loss_txt(logits_per_text, ground_truth)
                      ) / 3
    else:
        total_loss = (
                      loss_img(logits_per_image, ground_truth)
                      + loss_txt(logits_per_text, ground_truth)
                      + loss + hinge_loss
                      ) / 4

    acc = None
    intent_acc = 0
    for j in range(len(predict_label)):
        if predict_label[j]==intent_label[j]:
            intent_acc +=1

    if args.report_training_batch_acc:
        i2t_acc = (logits_per_image.argmax(-1) == ground_truth).sum() / len(logits_per_image)
        t2i_acc = (logits_per_text.argmax(-1) == ground_truth).sum() / len(logits_per_text)
        intent_acc = intent_acc /  len(predict_label)
        top_1_acc = top_1_acc / len(predict_label)
        top_3_acc = top_3_acc / len(predict_label)
        top_5_acc = top_5_acc / len(predict_label)
        acc = {"i2t": i2t_acc, "t2i": t2i_acc,'intent_acc': intent_acc,
               'top1':top_1_acc,'top3':top_3_acc, 'top5':top_5_acc}

    if args.distllation:
        total_loss += kd_loss * args.kd_loss_weight

    return total_loss, acc,all_attn_features, all_intent_features


def freeze_vision_bn(args, model):
    # freeze bn running mean and variance
    if 'RN' in args.vision_model:
        RN_visual_modules = model.module.visual.modules() if isinstance(model,
                                                                        nn.parallel.DistributedDataParallel) else model.visual.modules()
        for m in RN_visual_modules:
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

def train(model, classifier_model, attn_model, data,memory_dict, speaker_list, epoch, optimizer, scaler, scheduler, args, global_trained_steps, teacher_model=None):
    id2intent, intent2id, intent2token, \
    imgid2image, imgid2kges, imgid2kpos, imgid2kface, imgid2kver,imgid2intent,txt2intent=get_all_features()

    model.train()
    classifier_model.train()
    if args.freeze_vision:
        freeze_vision_bn(args, model)

    dataloader, sampler = data['train'].dataloader, data['train'].sampler

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    loss_img = loss_img.cuda(args.local_device_rank)
    loss_txt = loss_txt.cuda(args.local_device_rank)

    if sampler is not None:
        sampler.set_epoch(epoch)

    num_steps_per_epoch = dataloader.num_batches // args.accum_freq
    data_iter = iter(dataloader)
    all_data_iter = iter(dataloader)

    if args.accum_freq > 1:
        accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
        if args.distllation:
            teacher_accum_image_features = []

    end = time.time()
    epoch_trained_steps = 0

    Flag = 1
    all_attn_features, all_intent_features = None, None
    for i in range(0, dataloader.num_batches):
        batch = next(data_iter)

        i_accum = i // args.accum_freq
        step = num_steps_per_epoch * epoch + i_accum
        # reach the args.max_steps, exit training:
        if step >= args.max_steps:
            logging.info("Stopping training due to step {} has reached max_steps {}".format(step,
                                                                                            args.max_steps // args.accum_freq))
            return epoch_trained_steps
        scheduler(step)
        optimizer.zero_grad()
        image_ids, images, text_id, raw_text, texts, sent_texts, intent_zh, keywords, speaker, number, \
        comet, intent_ids, k_gesture, k_posture, k_facial, k_verbal = batch
        images = images.cuda(args.local_device_rank, non_blocking=True)
        data_time = time.time() - end
        m = model.module

        if args.accum_freq == 1:
            # with automatic mixed precision.
            if args.precision == "amp":
                with autocast():
                    if args.distllation:
                        total_loss, acc = get_loss(model, images, texts,sent_texts, loss_img, loss_txt, args,
                                                   teacher_model=teacher_model)
                    else:
                        # print('1')
                        # exit()
                        total_loss, acc,all_attn_features, all_intent_features = get_loss(Flag, model,classifier_model, images, texts, sent_texts, keywords, speaker, loss_img, loss_txt, number,
                                                   # real_image, cartoon_image, people_image, animal_image,
                                                   memory_dict, speaker_list,comet,intent_ids,k_gesture,k_posture,k_facial,k_verbal,imgid2image,
                                                    imgid2kges,imgid2kpos,imgid2kface,imgid2kver,intent2token,id2intent,imgid2intent,
                                                   all_attn_features, all_intent_features,args)
                        Flag = 0

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                scaler.update()

            else:
                if args.distllation:
                    total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args,
                                               teacher_model=teacher_model)
                else:
                    total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args)
                total_loss.backward()
                optimizer.step()
        else:
            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                with autocast(enabled=(args.precision == "amp")):
                    chunk_image_features, chunk_text_features, _ = model(images, texts, keywords)
                if args.distllation:
                    output = teacher_model.module.get_feature(images)
                    if (len(output) == 2):
                        teacher_chunk_image_features = output[0]
                    else:
                        teacher_chunk_image_features = output
                accum_image_features.append(chunk_image_features)
                accum_text_features.append(chunk_text_features)
                if args.distllation:
                    teacher_accum_image_features.append(teacher_chunk_image_features)

                accum_images.append(images)
                accum_texts.append(texts)

            if ((i + 1) % args.accum_freq) > 0:
                # FIXME this makes data time logging unreliable when accumulating
                continue

            optimizer.zero_grad()
            for j in range(args.accum_freq):
                images = accum_images[j]
                texts = accum_texts[j]
                with autocast(enabled=(args.precision == "amp")):
                    if args.distllation:
                        total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args, accum_image_features,
                                                   accum_text_features, j, teacher_model, teacher_accum_image_features)
                    else:
                        total_loss, acc = get_loss(model, images, texts, sent_texts, loss_img, loss_txt, args, accum_image_features,
                                                   accum_text_features, j)
                if args.precision == "amp":
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

            if args.precision == "amp":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        # reset gradient accum, if enabled
        if args.accum_freq > 1:
            accum_images, accum_texts, accum_image_features, accum_text_features = [], [], [], []
            if args.distllation:
                teacher_accum_image_features = []

        m.logit_scale.data = torch.clamp(m.logit_scale.data, 0, 4.6052)

        batch_time = time.time() - end
        end = time.time()

        epoch_trained_steps += 1

        if is_master(args) and ((step + 1) % args.log_interval) == 0:
            batch_size = len(images) * args.accum_freq
            num_samples = (i_accum + 1) * batch_size * args.world_size
            samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * (i_accum + 1) / num_steps_per_epoch

            logging.info(
                f"Global Steps: {step + 1}/{args.max_steps} | " +
                f"Train Epoch: {epoch + 1} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | " +
                f"Loss: {total_loss.item():.6f} | " +
                (f"Image2Text Acc: {acc['i2t'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Text2Image Acc: {acc['t2i'].item() * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"Intent Acc: {acc['intent_acc'] * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"top1 Acc: {acc['top1'] * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"top3 Acc: {acc['top3'] * 100:.2f} | " if args.report_training_batch_acc else "") +
                (f"top5 Acc: {acc['top5'] * 100:.2f} | " if args.report_training_batch_acc else "") +
                f"Data Time: {data_time:.3f}s | " +
                f"Batch Time: {batch_time:.3f}s | " +
                f"LR: {optimizer.param_groups[0]['lr']:5f} | " +
                f"logit_scale: {m.logit_scale.data:.3f} | " +
                f"Global Batch Size: {batch_size * args.world_size}"
            )

        if True:
            assert "val" in data, "Error: Valid dataset has not been built."
            if not args.use_flash_attention:
                evaluate(model, data, epoch, args, step + 1,imgid2image, imgid2kges, imgid2kpos, imgid2kface, imgid2kver,imgid2intent,txt2intent,intent2id,id2intent,all_intent_features,classifier_model,all_attn_features,intent2token)
            else:
                # fp16 is needed in flash attention
                with autocast():
                    evaluate(model, data, epoch, args, step + 1,imgid2image, imgid2kges, imgid2kpos, imgid2kface, imgid2kver,imgid2intent,txt2intent,intent2id,id2intent,all_intent_features,classifier_model,all_attn_features,intent2token)
            # set model back to train mode
            model.train()
            if args.freeze_vision:
                freeze_vision_bn(args, model)

        if args.should_save and args.save_step_frequency > 0 and ((step + 1) % args.save_step_frequency) == 0:
            save_path = os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}_{step + 1}.pt")
            t1 = time.time()
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

            t1 = time.time()
            save_path = os.path.join(args.checkpoint_path, f"epoch_latest.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "name": args.name,
                    "state_dict": model.state_dict() if not args.use_flash_attention else convert_state_dict(
                        model.state_dict()),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )
            logging.info(
                "Saved checkpoint {} (epoch {} @ {} steps) (writing took {} seconds)".format(save_path, epoch + 1,
                                                                                             step + 1,
                                                                                             time.time() - t1))

    return epoch_trained_steps


def get_top_N_images(query, data_list, top_K=4):
    sim_list = []
    for img in data_list:
        similarity = cosine_similarity(query.cpu().detach().numpy(), img.cpu().detach().numpy())
        sim_list.append(similarity)
    sorted_indices = sorted(enumerate(sim_list), key=lambda x: x[1], reverse=True)

    top_five_indices = [index for index, value in sorted_indices[:top_K]]

    # print(top_five_indices)
    return top_five_indices


def get_images_similarity(query, data_list, top_K=4):
    sim_list = []
    for img in data_list:
        similarity = cosine_similarity(query.cpu().detach().numpy(), img.cpu().detach().numpy())
        sim_list.append(similarity)

    return sim_list

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)

    return similarity


def metric_intent_label(json_file):
    with open('Dataset/' +
              args.data_mode + '/' + args.data_mode + '_response.json', 'r',
              encoding='utf-8') as f:
        datas = json.load(f)
    img_list = []
    img_dict = {}
    label_list = []
    for data in datas:
        #  获取到所有的intent
        if 'split' not in data and data['msgtype'] == 'sticker':
            label = data['fined_intent'].lower()
            if label not in label_list:
                label_list.append(label)
            img_name = data["type_info"]
            if '\\' in img_name:
                img_name = img_name.split('\\')[-1].split('.')[0].split('_')[-1]
            elif '/' in img_name:
                img_name = img_name.split('/')[-1].split('.')[0].split('_')[-1]
            else:
                print(img_name)
                exit()
            if img_name not in img_list:
                img_list.append(img_name)
                img_dict[img_name] = []

            img_dict[img_name].append(label)

    new_img = {}
    for item, value in img_dict.items():
        new_value = []
        for v in value:
            new_value.append(label_list.index(v))
        new_img[item] = new_value

    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    pred_list = []
    real_list = []
    for data in datas:
        image_id = data['imageid']
        text_id = data['textid']
        raw_text = data['raw_text']
        target_image = data['target_image']

        target_list = img_dict[target_image.split('_')[1]]
        print('target_list:',target_list)
        # 只要有一个匹配即可
        # flag = 1
        print('image_id:',image_id)
        # exit()
        flag=1
        for i in range(len(image_id)):
            if img_dict[image_id[i].split('_')[-1]] in target_list: #这应该是个list
                pred_list.append(target_list[0])
                real_list.append(target_list[0])
                flag=0
                break
        if flag:
            pred_list.append(img_dict[image_id[0].split('_')[-1]][0])
            real_list.append(target_list[0])

    print('real list:',real_list)
    print('pred list:',pred_list)
        # 计算准确率
    accuracy = accuracy_score(real_list, pred_list)
    # 计算精确度（precision）：它衡量的是预测为正例的样本中有多少是真正的正例
    precision = precision_score(real_list, pred_list, average='weighted')
    # 计算 F1 分数：综合考虑了精确度和召回率
    f1 = f1_score(real_list, pred_list, average='weighted')
    print(accuracy, precision, f1)
    return accuracy, precision, f1


def metric_one_by_one(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    acc = 0
    pred_list = []
    real_list = []
    for data in datas:
        image_id = data['imageid']
        text_id = data['textid']
        raw_text = data['raw_text']
        target_image = data['target_image']

        flag = 1
        for i in range(len(image_id)):
            if image_id[i] in target_image[i]:
                pred_list.append(str(id.split('_')[1]))
                flag = 0
                break

        if flag:
            pred_list.append(str(image_id[0].split('_')[1]))

        real_list.append(str(target_image.split('_')[1]))
    print('pred_list:', len(pred_list))
    print('real_list:', len(real_list))

    # 计算准确率
    accuracy = accuracy_score(real_list, pred_list)
    precision = precision_score(real_list, pred_list, average='weighted')
    f1 = f1_score(real_list, pred_list, average='weighted')
    print(accuracy, precision, f1)
    return accuracy, precision, f1


def evaluate(model, data, epoch, args, steps,imgid2image, imgid2kges, imgid2kpos, imgid2kface, imgid2kver,imgid2intent,txt2intent,intent2id,id2intent,all_intent_features,classifier_model,all_attn_features,intent2token):
    logging.info("Begin to eval on validation set (epoch {} @ {} steps)...".format(epoch + 1, steps))
    model.eval()
    dataloader = data['val'].dataloader
    data_iter = iter(dataloader)

    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    all_image_ids, all_text_ids, all_raw_texts = [], [], []
    Flag = 1
    top_1_acc, top_3_acc, top_5_acc = 0, 0, 0
    with torch.no_grad():
        for i in range(dataloader.num_batches):
            batch = next(data_iter)
            image_ids, images, text_ids, raw_text, texts, sent_texts, intent_zh, keywords, speaker, number, \
            comet, intent_ids, k_gesture, k_posture, k_facial, k_verbal = batch
            image_ids = [item.tolist() if isinstance(item, torch.Tensor) else item for item in image_ids]
            text_ids = [item.tolist() if isinstance(item, torch.Tensor) else item for item in text_ids]
            raw_text = [item.tolist() if isinstance(item, torch.Tensor) else item for item in raw_text]

            all_image_ids = all_image_ids + image_ids
            all_text_ids = all_text_ids + text_ids

            all_raw_texts = all_raw_texts + raw_text  # .append(raw_text)
            images = images.cuda(args.local_device_rank, non_blocking=True)
            texts = texts.cuda(args.local_device_rank, non_blocking=True)

            if Flag:
                image_features, text_features, logit_scale, style_loss, \
                all_attn_features, all_intent_features = model(Flag, images, text=texts, sent_text=sent_texts,
                                                               keywords=keywords,
                                                               speaker=speaker, number=number,
                                                               mask_ratio=args.mask_ratio,
                                                               comet=comet,
                                                               k_gesture=k_gesture, k_posture=k_posture,
                                                               k_facial=k_facial,
                                                               k_verbal=k_verbal, imgid2image=imgid2image,
                                                               imgid2kges=imgid2kges, imgid2kpos=imgid2kpos,
                                                               imgid2kface=imgid2kface, imgid2kver=imgid2kver,
                                                               intent2token=intent2token, id2intent=id2intent)
            else:
                image_features, text_features, logit_scale, style_loss = model(Flag, images, text=texts,
                                                                               sent_text=sent_texts,
                                                                               keywords=keywords,
                                                                               speaker=speaker, number=number,
                                                                               mask_ratio=args.mask_ratio,
                                                                               comet=comet,
                                                                               k_gesture=k_gesture, k_posture=k_posture,
                                                                               k_facial=k_facial,
                                                                               k_verbal=k_verbal,
                                                                               imgid2image=imgid2image,
                                                                               imgid2kges=imgid2kges,
                                                                               imgid2kpos=imgid2kpos,
                                                                               imgid2kface=imgid2kface,
                                                                               imgid2kver=imgid2kver,
                                                                               intent2token=intent2token,
                                                                               id2intent=id2intent)
            Flag = 0
            print('img_id:',image_ids)
            intent_prob, predict_label = classifier_model(text_features)
            predict_label = predict_label.tolist()

            intent_label = [int(label) for label in intent_ids]
            intent_ids = torch.tensor([int(label) for label in intent_ids])
            attn_list, img_list,all_image_features = [], [],[]

            for k, v in all_attn_features.items():
                attn_list.append(v[0])
                img_list.append(k)
            bz_loss = 0
            real_intent_list = []
            for each_label in intent_label:
                for j in img_list:
                    if int(imgid2intent[j][0]) == each_label:
                        real_intent_list.append(1)
                    else:
                        real_intent_list.append(0)
            all_predict_list = []
            bz = 0
            print(' predict_label:', predict_label)

            for each in text_features:
                index_list = get_images_similarity(each, attn_list, top_K=1)
                all_predict_list += index_list
                index_top_5_list = get_top_N_images(each, attn_list, top_K=5)
                sticker_intentid = []
                for j in index_top_5_list:
                    sticker_intentid.append(int(imgid2intent[img_list[j]][0]))
                print('sticker_intentid:',sticker_intentid)
                print('intent_label[bz]:',intent_label[bz])

                if intent_label[bz] in sticker_intentid[:1]:
                    top_1_acc += 1
                if intent_label[bz] in sticker_intentid[:3]:
                    top_3_acc += 1
                if intent_label[bz] in sticker_intentid[:5]:
                    top_5_acc += 1

        print('Test:',top_1_acc,top_3_acc,top_5_acc)

def cosineSimilarityLoss(feature1, feature2):
    scale_factor_h = feature1.shape[0] / feature2.size(0)
    scale_factor_w = feature1.shape[1] / feature2.size(1)

    feature2_interpolated = F.interpolate(feature2.unsqueeze(0).unsqueeze(0),
                                          size=(feature1.shape[0], feature1.shape[1]),
                                          mode='bilinear',
                                          align_corners=False)
    feature2_interpolated = feature2_interpolated.squeeze(0).squeeze(0)

    cosine_sim = F.cosine_similarity(feature1, feature2_interpolated, dim=1)
    similarity_loss = 1 - cosine_sim.mean()
    return similarity_loss
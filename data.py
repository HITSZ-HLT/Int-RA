from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass
from params import data_mode
import lmdb
import pickle
from nltk.corpus import wordnet
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from cn_clip.clip import _tokenizer
from cn_clip.clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text

def get_all_features():
    # print(lmdb_path)

    lmdb_path = '/data/wangbb/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/train'
    if lmdb_path.split('/')[-1] == 'train':
        lmdb_path_1 = lmdb_path.replace('train', 'test')
        lmdb_path_2 = lmdb_path.replace('train', 'valid')
    elif lmdb_path.split('/')[-1] == 'valid':
        lmdb_path_1 = lmdb_path.replace('valid', 'test')
        lmdb_path_2 = lmdb_path.replace('valid', 'train')
    elif lmdb_path.split('/')[-1] == 'test':
        lmdb_path_1 = lmdb_path.replace('test', 'train')
        lmdb_path_2 = lmdb_path.replace('test', 'valid')
    # assert LMDB directories exist
    assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
    lmdb_pairs = os.path.join(lmdb_path, "pairs")
    lmdb_pairs_1 = os.path.join(lmdb_path_1, "pairs")
    lmdb_pairs_2 = os.path.join(lmdb_path_2, "pairs")
    assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
        lmdb_pairs, split)
    print('lmdb:',lmdb_path,lmdb_path_1,lmdb_path_2)
    # lmdb_img_path = lmdb_path#.replace('train', '').replace('test', '').replace('valid', '')
    lmdb_imgs_1 = os.path.join('/data/wangbb/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
    # lmdb_imgs_2 = os.path.join('/data10T/wangbingbing/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
    # lmdb_imgs_3 = os.path.join('/data10T/wangbingbing/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
    # lmdb_imgs_4 = os.path.join(lmdb_img_path, "imgs4")


    print('lmdb_pairs:', lmdb_imgs_1)
    # exit()
    # open LMDB files
    env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
    txn_pairs = env_pairs.begin(buffers=True)
    env_pairs_1 = lmdb.open(lmdb_pairs_1, readonly=True, create=False, lock=False, readahead=False,
                                 meminit=False)
    txn_pairs_1 = env_pairs_1.begin(buffers=True)
    env_pairs_2 = lmdb.open(lmdb_pairs_2, readonly=True, create=False, lock=False, readahead=False,
                                 meminit=False)
    txn_pairs_2 = env_pairs_2.begin(buffers=True)

    env_imgs_1 = lmdb.open(lmdb_imgs_1, readonly=True, create=False, lock=False, readahead=False,
                                meminit=False)
    txn_imgs_1 = env_imgs_1.begin(buffers=True)
    # env_imgs_2 = lmdb.open(lmdb_imgs_2, readonly=True, create=False, lock=False, readahead=False,
    #                             meminit=False)
    # txn_imgs_2 = env_imgs_2.begin(buffers=True)
    # env_imgs_3 = lmdb.open(lmdb_imgs_3, readonly=True, create=False, lock=False, readahead=False,
    #                             meminit=False)
    # txn_imgs_3 = env_imgs_3.begin(buffers=True)
    # env_imgs_4 = lmdb.open(lmdb_imgs_4, readonly=True, create=False, lock=False, readahead=False,
    #                             meminit=False)
    # txn_imgs_4 = env_imgs_4.begin(buffers=True)
    # print('2')
    # number_images_1 = int(txn_imgs_1.get(key=b'num_images').tobytes().decode('utf-8'))
    # number_images_2 = int(txn_imgs_2.get(key=b'num_images').tobytes().decode('utf-8'))
    # number_images_3 = int(txn_imgs_3.get(key=b'num_images').tobytes().decode('utf-8'))
    # number_images_4 = int(txn_imgs_4.get(key=b'num_images').tobytes().decode('utf-8'))

    img_id_list_1, img_id_list_2, img_id_list_3, img_id_list_4 = [], [], [], []
    cursor = env_imgs_1.begin().cursor()
    for key, value in cursor:
        if key.decode('utf-8') != 'num_images':
            img_id_list_1.append(key.decode('utf-8'))

    # cursor = env_imgs_2.begin().cursor()
    # for key, value in cursor:
    #     if key.decode('utf-8') != 'num_images':
    #         img_id_list_2.append(key.decode('utf-8'))
    # cursor = env_imgs_3.begin().cursor()
    # for key, value in cursor:
    #     if key.decode('utf-8') != 'num_images':
    #         img_id_list_3.append(key.decode('utf-8'))
    # cursor = env_imgs_4.begin().cursor()
    # for key, value in cursor:
    #     if key.decode('utf-8') != 'num_images':
    #         img_id_list_4.append(key.decode('utf-8'))

    id2intent_txt = '/data/wangbb/Chinese-CLIP/Dataset/total/intent_label.txt'
    id2intent = {}
    intent2id = {}
    intent2token = {}
    with open(id2intent_txt, 'r', encoding='utf-8') as f:
        datas = f.readlines()
    # print(datas)
    for data in datas:
        data = data.strip('\n').split('\t')
        # print(data)
        # exit()
        id2intent[data[0]] = data[1]
        intent2id[data[1]] = data[0]

        intent_token = tokenize([_preprocess_text(data[2])], None, context_length=10)
        intent2token[data[1]] = intent_token

    # =================================================================
    imgid2image = {}
    imgid2kges, imgid2kpos, imgid2kface, imgid2kver = {}, {}, {}, {}
    imgid2intent = {}
    number_samples = int(txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
    Flag = 0
    txt2intent = {}
    # print()
    with open('/data/wangbb/Chinese-CLIP/Dataset/total/test_contexts_total_key.jsonl','r',encoding='utf-8') as f:
        datas = f.readlines()
    for data in datas:
        data = eval(data)
        txt2intent[data['text']]=data['fined_intent']

        if len(data['keyword'].split(','))!=4:
            print('test:',data)
            Flag = 1

    with open('/data/wangbb/Chinese-CLIP/Dataset/total/train_contexts_total_key.jsonl','r',encoding='utf-8') as f:
        datas = f.readlines()
    for data in datas:
        data = eval(data)
        txt2intent[data['text']]=data['fined_intent']
        if len(data['keyword'].split(','))!=4:
            print('train:',data)
            Flag=1

    with open('/data/wangbb/Chinese-CLIP/Dataset/total/valid_contexts_total_key.jsonl','r',encoding='utf-8') as f:
        datas = f.readlines()
    for data in datas:
        data = eval(data)
        txt2intent[data['text']]=data['fined_intent']
        if len(data['keyword'].split(','))!=4:
            print('valid:',data)
            Flag = 1
    # for training data
    # exit()
    # print('txt2intent:',txt2intent)
    if Flag:
        print(Flag)
        exit()
    for i in range(number_samples):
        pair = pickle.loads(txn_pairs.get("{}".format(i).encode('utf-8')).tobytes())
        image_id, text_id, raw_text, style, form, \
        coarse_intent, fined_intent, intent_zh, keyword, speaker, comet = pair

        # print(keyword.split(','))
        k_gesture, k_posture, k_facial, k_verbal = keyword.split(',')
        # print(k_gesture,k_posture,k_facial,k_verbal)
        # exit()
        k_gesture = tokenize([_preprocess_text(k_gesture)], None, context_length=15)
        k_posture = tokenize([_preprocess_text(k_posture)], None, context_length=15)
        k_facial = tokenize([_preprocess_text(k_facial)], None, context_length=15)
        k_verbal = tokenize([_preprocess_text(k_verbal)], None, context_length=15)

        # print('0',image_id)
        # exit()
        imgid2kges[image_id] = k_gesture
        imgid2kpos[image_id] = k_posture
        imgid2kface[image_id] = k_facial
        imgid2kver[image_id] = k_verbal

        # if coarse_intent.lower() == 'entertain':
        #     intent_id = intent2id['entertain']
        # else:
        # fined_intent = txt2intent[raw_text].lower()
        intent_id = intent2id[fined_intent.lower()]

        imgid2intent[image_id] = intent_id
    number_samples_1 = int(txn_pairs_1.get(key=b'num_samples').tobytes().decode('utf-8'))
    number_samples_2 = int(txn_pairs_2.get(key=b'num_samples').tobytes().decode('utf-8'))

    for i in range(number_samples_1):
        pair = pickle.loads(txn_pairs_1.get("{}".format(i).encode('utf-8')).tobytes())
        image_id, text_id, raw_text, style, form, \
        coarse_intent, fined_intent, intent_zh, keyword, speaker, comet = pair

        k_gesture, k_posture, k_facial, k_verbal = keyword.split(',')
        k_gesture = tokenize([_preprocess_text(k_gesture)], None, context_length=15)
        k_posture = tokenize([_preprocess_text(k_posture)], None, context_length=15)
        k_facial = tokenize([_preprocess_text(k_facial)], None, context_length=15)
        k_verbal = tokenize([_preprocess_text(k_verbal)], None, context_length=15)

        # self.imgid2image[image_id] = image
        # print('1:',image_id)
        # exit()
        imgid2kges[image_id] = k_gesture
        imgid2kpos[image_id] = k_posture
        imgid2kface[image_id] = k_facial
        imgid2kver[image_id] = k_verbal

        # if coarse_intent.lower() == 'entertain':
        #     intent_id = intent2id['entertain']
        # else:
        # fined_intent = txt2intent[raw_text].lower()
        intent_id = intent2id[fined_intent.lower()]

        imgid2intent[image_id] = intent_id

    for i in range(number_samples_2):
        pair = pickle.loads(txn_pairs_2.get("{}".format(i).encode('utf-8')).tobytes())
        image_id, text_id, raw_text, style, form, \
        coarse_intent, fined_intent, intent_zh, keyword, speaker, comet = pair

        k_gesture, k_posture, k_facial, k_verbal = keyword.split(',')
        k_gesture = tokenize([_preprocess_text(k_gesture)], None, context_length=15)
        k_posture = tokenize([_preprocess_text(k_posture)], None, context_length=15)
        k_facial = tokenize([_preprocess_text(k_facial)], None, context_length=15)
        k_verbal = tokenize([_preprocess_text(k_verbal)], None, context_length=15)

        # self.imgid2image[image_id] = image
        # print('3:',image_id)
        # exit()
        imgid2kges[image_id] = k_gesture
        imgid2kpos[image_id] = k_posture
        imgid2kface[image_id] = k_facial
        imgid2kver[image_id] = k_verbal

        # if coarse_intent.lower() == 'entertain':
        #     intent_id = intent2id['entertain']
        # else:
        # fined_intent = txt2intent[raw_text].lower()
        intent_id = intent2id[fined_intent.lower()]

        imgid2intent[image_id] = intent_id

    # print(len(imgid2intent), len(imgid2image), len(imgid2kver))
    # exit()
    resolution=224
    e_transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            # print('transform')
    e_transform = Compose(e_transform.transforms[:-3] + [_convert_to_rgb] + e_transform.transforms[-3:])
    for i in range(len(img_id_list_1)):
        # print('img_id:', img_id_list_1[i])
        image_b64 = txn_imgs_1.get("{}".format(img_id_list_1[i]).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
        image =   e_transform(image)
        imgid2image[img_id_list_1[i]] = image

    # for i in range(len(img_id_list_2)):
    #     print('img_id:', img_id_list_2[i])
    #     image_b64 = txn_imgs_2.get("{}".format(img_id_list_2[i]).encode('utf-8')).tobytes()
    #     image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
    #     image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
    #     image =  e_transform(image)
    #     imgid2image[img_id_list_2[i]] = image
    # for i in range(len(img_id_list_3)):
    #     print('img_id:', img_id_list_3[i])
    #     image_b64 = txn_imgs_3.get("{}".format(img_id_list_3[i]).encode('utf-8')).tobytes()
    #     image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
    #     image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
    #     image =   e_transform(image)
    #     imgid2image[img_id_list_3[i]] = image
    # for i in range(len(img_id_list_4)):
    #     print('img_id:', img_id_list_4[i])
    #     image_b64 = txn_imgs_4.get("{}".format(img_id_list_4[i]).encode('utf-8')).tobytes()
    #     image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
    #     image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64)))  # already resized
    #     image = e_transform(image)
    #     imgid2image[img_id_list_4[i]] = image



    return id2intent, intent2id, intent2token, \
           imgid2image, imgid2kges, imgid2kpos, imgid2kface, imgid2kver,imgid2intent, txt2intent

class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="valid", max_txt_length=64, use_augment=False, resolution=224):
        lmdb_path = lmdb_path
        print(lmdb_path)
        if lmdb_path.split('/')[-1] == 'train':
            lmdb_path_1 = lmdb_path.replace('train', 'test')
            lmdb_path_2 = lmdb_path.replace('train', 'valid')
        elif lmdb_path.split('/')[-1] == 'valid':
            lmdb_path_1 = lmdb_path.replace('val', 'test')
            lmdb_path_2 = lmdb_path.replace('valid', 'train')
        elif lmdb_path.split('/')[-1] == 'test':
            lmdb_path_1 = lmdb_path.replace('test', 'train')
            lmdb_path_2 = lmdb_path.replace('test', 'valid')
        else:
            print(lmdb_path)
            exit()
        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        # lmdb_pairs_1 = os.path.join(lmdb_path_1, "pairs")
        # lmdb_pairs_2 = os.path.join(lmdb_path_2, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(
            lmdb_pairs, split)
        # lmdb_img_path = lmdb_path.replace('train', '').replace('test', '').replace('valid', '')
        lmdb_imgs_1 = os.path.join('/data/wangbb/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
        # lmdb_imgs_2 = os.path.join('/data10T/wangbingbing/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
        # lmdb_imgs_3 = os.path.join('/data10T/wangbingbing/Chinese-CLIP/Dataset/total/lmdb_'+data_mode+'/', "imgs")
        # print(lmdb_path,lmdb_path_1,lmdb_path_2)
        # exit()
        # lmdb_imgs_4 = os.path.join(lmdb_img_path, "imgs4")
        # assert os.path.isdir(lmdb_imgs_1), "The LMDB directory {} of {} image base64 strings does not exist!".format(
        #     lmdb_imgs, split)

        # print('lmdb_pairs:',lmdb_pairs,lmdb_imgs)
        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)

        # self.env_pairs_1 = lmdb.open(lmdb_pairs_1, readonly=True, create=False, lock=False, readahead=False,
        #                              meminit=False)
        # self.txn_pairs_1 = self.env_pairs_1.begin(buffers=True)
        # self.env_pairs_2 = lmdb.open(lmdb_pairs_2, readonly=True, create=False, lock=False, readahead=False,
        #                              meminit=False)
        # self.txn_pairs_2 = self.env_pairs_2.begin(buffers=True)

        self.env_imgs_1 = lmdb.open(lmdb_imgs_1, readonly=True, create=False, lock=False, readahead=False,
                                    meminit=False)
        self.txn_imgs_1 = self.env_imgs_1.begin(buffers=True)
        # self.env_imgs_2 = lmdb.open(lmdb_imgs_2, readonly=True, create=False, lock=False, readahead=False,
        #                             meminit=False)
        # self.txn_imgs_2 = self.env_imgs_2.begin(buffers=True)
        # self.env_imgs_3 = lmdb.open(lmdb_imgs_3, readonly=True, create=False, lock=False, readahead=False,
        #                             meminit=False)
        # self.txn_imgs_3 = self.env_imgs_3.begin(buffers=True)
        # self.env_imgs_4 = lmdb.open(lmdb_imgs_4, readonly=True, create=False, lock=False, readahead=False,
        #                             meminit=False)
        # self.txn_imgs_4 = self.env_imgs_4.begin(buffers=True)
        # print('2')
        # self.number_images_1 = int(self.txn_imgs_1.get(key=b'num_images').tobytes().decode('utf-8'))
        # self.number_images_2 = int(self.txn_imgs_2.get(key=b'num_images').tobytes().decode('utf-8'))
        # self.number_images_3 = int(self.txn_imgs_3.get(key=b'num_images').tobytes().decode('utf-8'))
        # self.number_images_4 = int(self.txn_imgs_4.get(key=b'num_images').tobytes().decode('utf-8'))

        self.img_id_list_1, self.img_id_list_2, self.img_id_list_3, self.img_id_list_4 = [], [], [], []
        cursor = self.env_imgs_1.begin().cursor()
        for key, value in cursor:
            if key.decode('utf-8') != 'num_images':
                self.img_id_list_1.append(key.decode('utf-8'))
        # cursor = self.env_imgs_2.begin().cursor()
        # for key, value in cursor:
        #     if key.decode('utf-8') != 'num_images':
        #         self.img_id_list_2.append(key.decode('utf-8'))
        # cursor = self.env_imgs_3.begin().cursor()
        # for key, value in cursor:
        #     if key.decode('utf-8') != 'num_images':
        #         self.img_id_list_3.append(key.decode('utf-8'))
        # cursor = self.env_imgs_4.begin().cursor()
        # for key, value in cursor:
        #     if key.decode('utf-8') != 'num_images':
        #         self.img_id_list_4.append(key.decode('utf-8'))
        # exit()
        # print('1:', self.img_id_list_1)
        # print('2:', self.img_id_list_2)
        # print('3:', self.img_id_list_3)
        # print('4:', self.img_id_list_4)

        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        # print('self:', self.number_samples)
        # logging.info(
        #     "{} LMDB file contains {} images and {} pairs.".format(split, self.number_images_1, self.number_samples))
        # print('3')
        # self.number_samples_1 = int(self.txn_pairs_1.get(key=b'num_samples').tobytes().decode('utf-8'))
        # logging.info(
        #     "{} LMDB file contains {} images and {} pairs.".format(split, self.number_images_1, self.number_samples))

        # self.number_samples_2 = int(self.txn_pairs_2.get(key=b'num_samples').tobytes().decode('utf-8'))

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1  # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split

        self.max_txt_length = max_txt_length

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)
        # # 给每个speaker设置一个style存储器
        # self.memory_dict = {}
        # self.speaker_list = []

        self.id2intent, self.intent2id, self.intent2token, \
        self.imgid2image, self.imgid2kges, self.imgid2kpos, self.imgid2kface, self.imgid2kver, self.imgid2intent, self.txt2intent = get_all_features()


    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            # print('here')
            transform = create_transform(
                input_size=resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            # print('transform')
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text, style, form, \
        coarse_intent, fined_intent, intent_zh, keyword, speaker, comet = pair
        # print('speaker 1:',speaker)
        # print('image_id:',image_id)
        # 尝试获取image_id对应的值
        # print('pair:',raw_text, coarse_intent, fined_intent)
        # if coarse_intent.lower() == 'entertain':
        #     intent_id = self.intent2id['entertain']
        # else:
        # fined_intent = self.txt2intent[raw_text].lower()

        intent_id = self.intent2id[fined_intent.lower()]
        # exit()
        # print(intent_id)
        image = self.imgid2image[image_id]
        '''
        image_b64 = self.txn_imgs.get(image_id.encode('utf-8'))
        print('intent_id:',intent_id)
        # 如果image_b64为None，即image_id不存在
        if image_b64 is None:
            # print('self.split:', self.split)
            raise ValueError(f"Image with id {image_id} not found in the database")

        # if speaker not in self.speaker_list:
        #     self.memory_dict[speaker]={'real':0.25,'animal':0.25,'cartoon':0.25,'people':0.25}
        #     self.speaker_list.append(speaker)
        print('image 1')
        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        print('image 2')
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        print('image 3',image)
        image = self.transform(image)
        print('image_id:',image_id)
        # print(image_id + '+real.jpg')

        try:
            real_image_b64 = self.txn_imgs.get("{}".format(image_id + '+real.jpg').encode('utf-8')).tobytes()
        except AttributeError as e:
            print(f"Error for image_id {image_id}: {e}")
            real_image_b64 = None
        real_image_b64 = real_image_b64.decode(encoding="utf8", errors="ignore")
        real_image = Image.open(BytesIO(base64.urlsafe_b64decode(real_image_b64)))  # already resized
        real_image = self.transform(real_image)

        people_image_b64 = self.txn_imgs.get("{}".format(image_id + '+people.jpg').encode('utf-8')).tobytes()
        people_image_b64 = people_image_b64.decode(encoding="utf8", errors="ignore")
        people_image = Image.open(BytesIO(base64.urlsafe_b64decode(people_image_b64)))  # already resized
        people_image = self.transform(people_image)

        animal_image_b64 = self.txn_imgs.get("{}".format(image_id + '+animal.jpg').encode('utf-8')).tobytes()
        animal_image_b64 = animal_image_b64.decode(encoding="utf8", errors="ignore")
        animal_image = Image.open(BytesIO(base64.urlsafe_b64decode(animal_image_b64)))  # already resized
        animal_image = self.transform(animal_image)

        cartoon_image_b64 = self.txn_imgs.get("{}".format(image_id + '+cartoon.jpg').encode('utf-8')).tobytes()
        cartoon_image_b64 = cartoon_image_b64.decode(encoding="utf8", errors="ignore")
        cartoon_image = Image.open(BytesIO(base64.urlsafe_b64decode(cartoon_image_b64)))  # already resized
        cartoon_image = self.transform(cartoon_image)
        '''
        # print(image_id)

        raw_text = raw_text.split('[SEP]')[:-1]
        # print(raw_text)
        # exit()
        new_text = []
        num = 0

        raw_text = '[SEP]'.join(raw_text)
        raw_text = raw_text.split('\t')[1:]
        raw_text = '[SEP]'.join(raw_text)
        # similar_list = [item for sublist in similar for item in sublist + [0]]
        # print(raw_text,len(raw_text))
        # print(similar_list,len(similar_list))
        # exit()
        new_text = raw_text
        text_list = []
        # keyword_list = keyword.split(',')
        # k_gesture,k_posture,k_facial,k_verbal = keyword.split(',')
        # print(k_gesture,k_posture,k_facial,k_verbal)
        # exit()
        # keyword = tokenize([_preprocess_text(keyword)], context_length=20)[0]
        # text_tensor = torch.empty(0)
        # flag = 1

        # for text in new_text:
        # print(tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0].size())
        # if flag:
        #     text_tensor = tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0]
        #     flag = 0
        # text_tensor = torch.stack([text_tensor,tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0]],dim = 0)
        # text_list.append(torch.unsqueeze(tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0],dim=0))
        # text_list.append(
        #     tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0], dim=0)
        # print(torch.unsqueeze(tokenize([_preprocess_text(text)], context_length=self.max_txt_length)[0],dim=0).size())
        # print(text_list)
        number = len(text_list)
        for num in range(15 - len(text_list)):
            text_list.append(torch.unsqueeze(torch.zeros(52), dim=0))

        text_tensor = torch.cat(text_list, dim=0)
        # print(text_tensor.size(),len(new_text))
        # print('raw_text:', raw_text)
        # print('comet:', comet)
        raw_text = raw_text + '[SEP]' + comet
        # exit()
        text = tokenize([_preprocess_text(raw_text)], intent_zh, context_length=self.max_txt_length)  # [0]
        '''
        k_gesture, k_posture, k_facial, k_verbal = keyword.split(',')
        k_gesture, k_text = tokenize([_preprocess_text(k_gesture)], None,context_length=15)
        k_posture, k_text = tokenize([_preprocess_text(k_posture)], None, context_length=15)
        k_facial, k_text = tokenize([_preprocess_text(k_facial)], None, context_length=15)
        k_verbal, k_text = tokenize([_preprocess_text(k_verbal)], None, context_length=15)
        print('k_ges:',k_gesture)
        '''
        # print('2:',image_id)
        k_gesture = self.imgid2kges[image_id]
        k_posture = self.imgid2kpos[image_id]
        k_facial = self.imgid2kface[image_id]
        k_verbal = self.imgid2kver[image_id]
        # intent_zh, intent_text = tokenize([_preprocess_text(intent_zh)],intent_zh, context_length=self.max_txt_length)#[0]
        # eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        # text_list = np.array(text_list)
        similar = []

        # print('keyword:',decoded_keyword,decoded_text)
        # exit()
        # for each in text:
        #     # text = keyword + '[SEP]' + each
        #     print(each,keyword_text)
        #     words1 = list(jieba.cut(each))
        #     words2 = list(jieba.cut(keyword))
        #     # new_text.append(text)
        #     for w1 in words1:
        #         temp_list = []
        #         max_similarity = 0
        #         for w2 in words2:
        #             # 获取WordNet中的同义词集（synset）
        #             synsets1 = wordnet.synsets(w1)
        #             synsets2 = wordnet.synsets(w2)
        #             # 计算两个同义词集之间的相似度（这里使用path_similarity）
        #             if synsets1 and synsets2:
        #                 similarity = synsets1[0].path_similarity(synsets2[0])
        #                 max_similarity = max(max_similarity, similarity)
        #
        #         similarities.append(max_similarity)
        #     similar.append(temp_list)
        # print(keyword)
        # exit()
        # real_image, cartoon_image, people_image,animal_image,
        sent_text = ''
        return image_id, image, text_id, raw_text, text, sent_text, intent_zh, keyword, speaker, number, \
               comet, intent_id, k_gesture, k_posture, k_facial, k_verbal#, \
               # self.id2intent, self.intent2id, self.intent2token, \
               # self.imgid2image, self.imgid2kges, self.imgid2kpos, self.imgid2kface, self.imgid2kver, self.imgid2intent  # , eos_index


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(
        __file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = LMDBDataset(
        db_path,
        split="train" if is_train else "valid",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    )

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs).
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args,
            is_train=True,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    if args.val_data:
        data["valid"] = get_dataset(
            args,
            is_train=False,
            max_txt_length=max_txt_length,
            epoch_id=epoch_id)

    return data

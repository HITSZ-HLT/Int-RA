# -*- coding: utf-8 -*-
'''
This script serializes images and image-text pair annotations into LMDB files,
which supports more convenient dataset loading and random access to samples during training 
compared with TSV and Jsonl data files.
'''

import argparse
import os
from tqdm import tqdm
import lmdb
import json
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=False, help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
            (e.g. train,valid,test)"
    )
    parser.add_argument(
        "--mode", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
                (e.g. boba, kuaile,quanguo,siban)"
    )
    parser.add_argument(
        "--lmdb_dir", type=str, default=None, help="specify the directory which stores the output lmdb files. \
            If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    num=8
    # assert os.path.isdir(args.data_dir), "The data_dir does not exist! Please check the input args..."
    args.data_dir = '/data10T/wangbingbing/Chinese-CLIP/Dataset/total/'#+args.mode
    mode = args.mode#'wi'
    # read specified dataset splits
    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))

    # build LMDB data files
    if args.lmdb_dir is None:
        args.lmdb_dir = os.path.join(args.data_dir, "lmdb_"+mode+'_'+str(num))
    for split in specified_splits:
        # open new LMDB files
        lmdb_split_dir = os.path.join(args.lmdb_dir, split)
        if os.path.isdir(lmdb_split_dir):
            print("We will overwrite an existing LMDB file {}".format(lmdb_split_dir))
        os.makedirs(lmdb_split_dir, exist_ok=True)
        lmdb_img = os.path.join(lmdb_split_dir, "imgs")
        env_img = lmdb.open(lmdb_img, map_size=1024**4)
        txn_img = env_img.begin(write=True)
        lmdb_pairs = os.path.join(lmdb_split_dir, "pairs")
        env_pairs = lmdb.open(lmdb_pairs, map_size=1024**4)
        txn_pairs = env_pairs.begin(write=True)

        # write LMDB file storing (image_id, text_id, text, context) pairs
        pairs_annotation_path = os.path.join(args.data_dir, split+"_contexts_total_key.jsonl")
        # pairs_annotation_path = os.path.join(args.data_dir, split+"_contexts_comet_key.jsonl")
        with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
            write_idx = 0
            for line in tqdm(fin_pairs):
                print(line)
                line = line.strip()
                obj = json.loads(line)
                print(obj)
                for field in ("textid", "text", "image_ids",'style','form',
                              'coarse_intent','fined_intent','intent_zh','keyword','speaker','comet'):
                    assert field in obj, "Field {} does not exist in line {}. \
                        Please check the integrity of the text annotation Jsonl file."
                for image_id in obj["image_ids"]:
                    if args.mode not in image_id:
                        continue
                    print('obj:',obj['text'])
                    if len(obj['text'].split('[SEP]'))!=num:
                        continue
                    dump = pickle.dumps((image_id, obj['textid'], obj['text'],
                                         obj['style'],obj['form'],
                                         obj['coarse_intent'],obj['fined_intent'],obj['intent_zh'],
                                         obj['keyword'],obj['speaker'],obj['comet'])) # encoded (image_id, text_id, text)
                    if 'User' not in obj['speaker']:
                        print(obj['speaker'], image_id,obj['response'])
                        exit()
                    txn_pairs.put(key="{}".format(write_idx).encode('utf-8'), value=dump)
                    write_idx += 1
                    if write_idx % 5000 == 0:
                        txn_pairs.commit()
                        txn_pairs = env_pairs.begin(write=True)
            txn_pairs.put(key=b'num_samples',
                    value="{}".format(write_idx).encode('utf-8'))
            txn_pairs.commit()
            env_pairs.close()
        print("Finished serializing {} {} split pairs into {}.".format(write_idx, split, lmdb_pairs))

        # write LMDB file storing image base64 strings
        base64_path = os.path.join(args.data_dir, "{}_imgs.tsv".format(split))
        # cartoon_path = os.path.join(args.data_dir, "{}_imgs_cartoon.tsv".format(split))
        # real_path = os.path.join(args.data_dir, "{}_imgs_real.tsv".format(split))
        # animal_path = os.path.join(args.data_dir, "{}_imgs_animal.tsv".format(split))
        # people_path = os.path.join(args.data_dir, "{}_imgs_people.tsv".format(split))
        image_id_list, b64_list = [], []
        with open(base64_path, "r", encoding="utf-8") as fin_imgs:
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                if args.mode not in image_id:
                    print('mode:',args.mode,image_id)
                    continue
                image_id_list.append(image_id)
                b64_list.append(b64)
        '''
        with open(cartoon_path, "r", encoding="utf-8") as fin_imgs:
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                image_id_list.append(image_id)
                b64_list.append(b64)

        with open(real_path, "r", encoding="utf-8") as fin_imgs:
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                image_id_list.append(image_id)
                b64_list.append(b64)

        with open(people_path, "r", encoding="utf-8") as fin_imgs:
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                image_id_list.append(image_id)
                b64_list.append(b64)

        with open(animal_path, "r", encoding="utf-8") as fin_imgs:
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                image_id_list.append(image_id)
                b64_list.append(b64)
        '''
        write_idx = 0
        for i in range(len(image_id_list)):
            image_id = image_id_list[i]
            b64 = b64_list[i]
            txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
            write_idx += 1
            if write_idx % 1000 == 0:
                txn_img.commit()
                txn_img = env_img.begin(write=True)
        txn_img.put(key=b'num_images',
                    value="{}".format(write_idx).encode('utf-8'))
        txn_img.commit()
        env_img.close()
        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img))
        '''
        with open(cartoon_path, "r", encoding="utf-8") as fin_imgs:
            write_idx = 0
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)
            txn_img.put(key=b'num_images',
                    value="{}".format(write_idx).encode('utf-8'))
            txn_img.commit()
            env_img.close()
        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img))

        with open(real_path, "r", encoding="utf-8") as fin_imgs:
            write_idx = 0
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)
            txn_img.put(key=b'num_images',
                    value="{}".format(write_idx).encode('utf-8'))
            txn_img.commit()
            env_img.close()
        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img))

        with open(people_path, "r", encoding="utf-8") as fin_imgs:
            write_idx = 0
            for line in tqdm(fin_imgs):
                line = line.strip()
                image_id, b64 = line.split("\t")
                txn_img.put(key="{}".format(image_id).encode('utf-8'), value=b64.encode("utf-8"))
                write_idx += 1
                if write_idx % 1000 == 0:
                    txn_img.commit()
                    txn_img = env_img.begin(write=True)
            txn_img.put(key=b'num_images',
                    value="{}".format(write_idx).encode('utf-8'))
            txn_img.commit()
            env_img.close()
        print("Finished serializing {} {} split images into {}.".format(write_idx, split, lmdb_img))
        '''

    print("done!")
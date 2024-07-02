import os
import glob
import shutil
import openai
import chromadb
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn as nn
import pickle
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import dashscope
from dashscope import TextEmbedding
import shutil
import os
import random
import pandas as pd

def generate_jd_cv_pair(file_path):
    pairs = []
    df = pd.read_csv(pairs_path)
    # print(len(df)) #4856
    for idx, row in df.iterrows():
        
        jd_uuid = row["JD_uuid"]
        if not pd.isna(row["CVN_uuid"]):
            cvn_uuid_list = row["CVN_uuid"].split(',')
        else:
            cvn_uuid_list = []
        if not pd.isna(row["CV1_uuid"]):
            cvn_uuid_list.append(row["CV1_uuid"])

        if jd_uuid not in jd2idx:
            jd_idx = len(jd2idx)
            jd2idx[jd_uuid] = jd_idx
            idx2jd[jd_idx] = jd_uuid
        else:
            jd_idx = jd2idx[jd_uuid]

        for cv_uuid in cvn_uuid_list:
            if cv_uuid not in cv2idx:
                cv_idx = len(cv2idx)
                cv2idx[cv_uuid] = cv_idx
                idx2cv[cv_idx] = cv_uuid
            else:
                cv_idx = cv2idx[cv_uuid]

            # pairs.append(((jd_idx, cv_idx), 1))
            pairs.append((jd_idx, cv_idx))
        
        print(f'process {idx}-th ja_ncv pairs')
    print("the number of pairs: ", len(pairs)) #18286
    return pairs

def generate_pairs(save_path, pairs_path):
    jd2idx, cv2idx = {'OOV': 0}, {'OOV':0}
    idx2jd, idx2cv = {0: 'OOV'}, {0: 'OOV'}
    generated_pairs = generate_jd_cv_pair(pairs_path) #18286


    pos_pairs = []
    neg_pairs = []
    pos_cnt, neg_cnt = 0, 0
    for i in range(1, len(jd2idx)):
        for j in range(i, len(cv2idx)):
            print(i, j)
            if (i, j) in generated_pairs:
                pos_pairs.append(((i, j), 1))
                pos_cnt += 1
            else:
                neg_pairs.append(((i, j), 0))
                neg_cnt += 1


    with open(os.path.join(save_path,'dicts.pkl'), 'wb') as f:
        pickle.dump(jd2idx, f)
        pickle.dump(idx2jd, f)
        pickle.dump(cv2idx, f)
        pickle.dump(idx2cv, f)

    with open(os.path.join(save_path, 'all_pairs.pkl'), 'wb') as f:
        pickle.dump(pos_pairs, f)
        pickle.dump(neg_pairs, f)

    print(f"generate {pos_cnt} positive samples")
    print(f"generate {neg_cnt} negtive samples")

def sample_pairs(save_path):
    random.seed(0)
    with open(os.path.join(save_path, 'all_pairs.pkl'), 'rb') as f:
        pos_pairs = pickle.load(f)
        neg_pairs = pickle.load(f)
    random.shuffle(neg_pairs)
    sampled_neg_pairs = random.sample(neg_pairs, 3 * len(pos_pairs))  #54858
    sampled_pairs = sampled_neg_pairs + pos_pairs
    random.shuffle(sampled_pairs)
    with open(os.path.join(save_path, 'all_pairs_1_pos_3_neg_pairs.pkl'), 'wb') as f:
        pickle.dump(sampled_pairs, f)
    print("Number of all sampling pairs: ", len(sampled_pairs))
    print('End of sampling')
    return sample_pairs


if __name__ == '__main__':
    save_path = '/root/havi/src/preprocess/data/'
    pairs_path = '/root/autodl-fs/wang/falcon/JD_CV.csv'


    with open(os.path.join(save_path, 'all_pairs.pkl'), 'rb') as f:
        pos = pickle.load(f)
        neg = pickle.load(f)
        
        
    jd2idx, cv2idx = {'OOV': 0}, {'OOV':0}
    idx2jd, idx2cv = {0: 'OOV'}, {0: 'OOV'}
    generated_pairs = generate_jd_cv_pair(pairs_path) #18286


    pos_pairs = []
    neg_pairs = []
    pos_cnt, neg_cnt = 0, 0
    for i in range(1, len(jd2idx)):
        for j in range(i, len(cv2idx)):
            print(i, j)
            if (i, j) in generated_pairs:
                pos_pairs.append(((i, j), 1))
                pos_cnt += 1
            else:
                neg_pairs.append(((i, j), 0))
                neg_cnt += 1


    with open(os.path.join(save_path,'dicts.pkl'), 'wb') as f:
        pickle.dump(jd2idx, f)
        pickle.dump(idx2jd, f)
        pickle.dump(cv2idx, f)
        pickle.dump(idx2cv, f)

    with open(os.path.join(save_path, 'all_pairs.pkl'), 'wb') as f:
        pickle.dump(pos_pairs, f)
        pickle.dump(neg_pairs, f)

    print(f"generate {pos_cnt} positive samples")
    print(f"generate {neg_cnt} negtive samples")



        



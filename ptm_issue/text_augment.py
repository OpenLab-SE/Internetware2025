import random
import re
from math import inf

import nltk
import numpy as np
import pandas as pd
import os
from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
from nltk import word_tokenize

warnings.filterwarnings('ignore')
ratios = [0.15, 0.2, 0.25]
path = r'my_data/train'
to_path = r'my_data/train'
seed = 42
modelPath = r'roberta-large'
model = RobertaForMaskedLM.from_pretrained(modelPath)
BERTtokenizer = RobertaTokenizer.from_pretrained(modelPath)
fill_mask = pipeline('fill-mask', model=model, tokenizer=BERTtokenizer)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def BERTAugment(text, ratio):
    final = word_tokenize(text)
    if len(final) == 0:
        return 0, " "
    else:
        words = []
        for id, it in enumerate(final):
            if not re.search('[^a-zA-Z_1-9 ]', it):
                words.append(id)
        if int(len(words) * ratio) < 1:
            return 0, text
        else:
            masks = list(sorted(random.sample(words, int(len(words) * ratio))))
            words4text = ['<mask>' if i in masks else final[i] for i in range(len(final))]
            text = ' '.join(words4text)
            # res是二维矩阵
            try:
                res = fill_mask(text)
            except:
                return 0, text
            # 对于每个mask，以及相应的预测的结果（5），我们将mask替换为预测结果，写入final，然后得到
            if len(masks) == 1:
                r = res
                flag = 0
                for i in range(len(r) - 1):
                    if r[i]['token_str'].replace(' ', '').lower() == final[masks[0]].lower():
                        flag = i + 1
                    else:
                        break
                # print(final[masks[0]], r[flag]['token_str'])
                final[masks[0]] = r[flag]['token_str']
                return 1, ' '.join(final)
            for mask, r in zip(masks, res):
                flag = 0
                for i in range(len(r) - 1):
                    if r[i]['token_str'].replace(' ', '').lower() == final[mask].lower():
                        flag = i + 1
                    else:
                        break
                # print(final[mask], r[flag]['token_str'])
                final[mask] = r[flag]['token_str']
            return 1, ' '.join(final)


def trainAugment(file):
    # 需要数据增强的文件
    df = pd.read_csv(os.path.join(path, file))
    # 记录每个类别下的样本
    dic = {}
    maxLabelNum = 0
    maxLabel = 0
    for i in df['labels'].unique():
        dd = df[df['labels'] == i].reset_index().drop('index', axis=1)
        dic[i] = dd
        if len(dd) >= maxLabelNum:
            maxLabelNum = len(dd)
            maxLabel = int(i)

    ls = list(dic.keys())
    ls.remove(maxLabel)
    for i in ls:
        dd = dic[i]
        samplesNum = maxLabelNum - len(dd)
        varNum = 0
        for ir, row in dd.iterrows():
            if varNum == samplesNum:
                break
            # 比例：0.2、0.15、0.25
            out = BERTAugment(str(row['Multi_Sentences']), 0.2)
            flag, text = out
            varNum += flag
            if flag == 1:
                row['Multi_Sentences'] = text
                df.loc[len(df)] = row
    df.reset_index().to_csv(os.path.join(to_path, file + '_TRAIN_Aug.csv'), index=False)


def trainAugment_2(file):
    # 需要数据增强的文件
    df = pd.read_csv(os.path.join(to_path, file + '_TRAIN_Bef.csv'))
    # 记录每个类别下的样本
    dic = {}
    minLabelNum = float(inf)
    minLabel = 0
    for i in df['label'].unique():
        dd = df[df['label'] == i].reset_index().drop('index', axis=1)
        dic[int(i)] = dd
        if len(dd) <= minLabelNum:
            minLabelNum = len(dd)
            minLabel = int(i)

    # 按照最小类进行采样
    ls = list(dic.keys())
    # ls.remove(minLabel)
    for i in ls:
        dd = dic[i].sample(n=minLabelNum).reset_index().drop('index', axis=1)
        for ir, row in dd.iterrows():
            # 比例：0.2、0.15、0.25
            for ratio in ratios:
                out = BERTAugment(str(row['Multi_Sentences']), ratio)
                flag, text = out
                if flag == 1:
                    row['Multi_Sentences'] = text
                    df.loc[len(df)] = row

    df.reset_index().to_csv(os.path.join(to_path, file + '_TRAIN_Aug_under.csv'), index=False)


def evalAugment(file, mode):
    # 需要数据增强的文件
    df = pd.read_csv(os.path.join(to_path, file + f'_{mode}_Bef.csv')).reset_index()
    for i in df['label'].unique():
        dd = df[df['label'] == i].reset_index()
        # dd = df[df['label'] == i].reset_index().drop('index', axis=1)
        for ir, row in tqdm(dd.iterrows(), total=len(dd), desc=mode + f' Augment(class {i})...'):
            for ratio in ratios:
                out = BERTAugment(str(row['Multi_Sentences']), ratio)
                _, text = out
                row['Multi_Sentences'] = text
                df.loc[len(df)] = row
        df.to_csv(os.path.join(to_path, file + '_TEST_Aug.csv'), index=False)
        # df.to_csv(os.path.join(to_path, file + '_DEV_Aug.csv'), index=False)
#         dataframe.rename(columns = {"old_name": "new_name"})
# dataframe.rename(columns = {"old1": "new1", "old2":"new2"},  inplace=True)


if __name__ == "__main__":
    # evalAugment('complete2', 'TEST')
    # trainAugment('complete2')
    # trainAugment_2('couple3')
    evalAugment('couple3', 'TEST')

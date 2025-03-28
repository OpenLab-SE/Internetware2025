import copy
import os
os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
import nltk
import random
import re
import traceback
import warnings
from math import inf

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util.text.tokenizer import Tokenizer


import nltk
nltk.download('punkt')
import numpy as np
import pandas as pd
from nlpaug.util import Action
from nltk import word_tokenize
# from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from tqdm import tqdm
# from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
from transformers import BertForMaskedLM, BertTokenizer, pipeline

os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings('ignore')
RATIOS = [0.15, 0.2, 0.25]
PATH = r'my_data/train'
TO_TRAIN_PATH = r'my_data/train'
TO_VALID_PATH = r'my_data/valid'
TO_TEST_PATH = r'my_data/test'
# MODELPATH = 'roberta-base'
# model = RobertaForMaskedLM.from_pretrained(MODELPATH)
# BERTtokenizer = RobertaTokenizer.from_pretrained(MODELPATH)
# MODELPATH = r'models/bert-base-uncased'
MODELPATH = r'/home/notebook/data/group/privacy/models/pretrained/models/bert-base-uncased'
T5MODELPATH = r'/home/notebook/data/group/privacy/models/pretrained/models/t5-base'
WMT19EN2DEMODELPATH=r'/home/notebook/data/group/privacy/models/pretrained/models/wmt19-en-de'
WMT19DE2ENMODELPATH=r'/home/notebook/data/group/privacy/models/pretrained/models/wmt19-de-en'

model = BertForMaskedLM.from_pretrained(MODELPATH)
BERTtokenizer = BertTokenizer.from_pretrained(MODELPATH)
fill_mask = pipeline('fill-mask', model=model, tokenizer=BERTtokenizer, device="cuda:0")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
max_length = model.config.max_position_embeddings  # 模型的最大输入长度

def BERTAugment(text, ratio):
    # 如果输入文本超过最大序列长度，可以考虑截断或分段

    final = word_tokenize(text)
    # final = BERTtokenizer.tokenize(text)
 
    if len(final) == 0:
        return 0, " "
    else:
        words = []
        for id, it in enumerate(final):
            if not re.search('[^a-zA-Z_1-9 ]', it):
                words.append(id)
        if int(len(words) * ratio) <= 1:
            return 0, text
        else:
            masks = list(sorted(random.sample(words, int(len(words) * ratio))))
            # words4text = ['<mask>' if i in masks else final[i] for i in range(len(final))]
            words4text = ['[MASK]' if i in masks else final[i] for i in range(len(final))]
            text = ' '.join(words4text)
            # res是二维矩阵
            try:
                # tokens = fill_mask.tokenizer(text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
                # res = fill_mask(**tokens)
                text_1 = BERTtokenizer.encode(text, max_length=max_length-2, truncation=True, return_tensors='pt', add_special_tokens=False)
                text_2 = BERTtokenizer.decode(text_1[0])
                res = fill_mask(text_2)
            except Exception as e:
                traceback.print_exc()
                print(len(words))
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


def augment(df, file, stage="train"):
    # 记录每个类别下的样本
    dic = {}
    maxLabelNum = 0
    maxLabel = 0
    for i in df['labels'].unique():
        dd = df[df['labels'] == i].reset_index().drop('index', axis=1)
        dic[i] = dd
        if len(dd) >= maxLabelNum:
            maxLabelNum = len(dd)
            maxLabel = i

    ls = list(dic.keys())
    update_ratio  = 1.0 * maxLabelNum / sum([len(v) for k,v in dic.items() if k != maxLabel])
    print(f"len of df is {len(df)}")
    for k,v in dic.items():
        print(f"num of {k} is {len(v)}")

    print(f"======update_ratio: {update_ratio}")
    ls.remove(maxLabel)
    for i in ls:
        dd = dic[i]
        # samplesNum = maxLabelNum - len(dd)
        varNum = len(dd)
        
        while True:
            if varNum >= int(len(dd) * update_ratio):
                    break
            for ir, row in dd.iterrows():
                if varNum >= int(len(dd) * update_ratio):
                    break
                # 比例：0.2、0.15、0.25
                out = BERTAugment(str(row['description']), 0.2)
                flag, text = out
                varNum += flag
                if flag == 1:
                    # print(f"ir: {ir}, flag:{flag} len(dd): {len(dd)}, target:{int(len(dd) * update_ratio)}, varNum: {varNum}")
                    new_row = copy.deepcopy(row)
                    new_row['description'] = text
                    df = df.append(new_row, ignore_index = True)

        print(f"before augment, num of {i} is: {len(dd)}")
        print(f"after augment, num of {i} is: {varNum}")
    if stage == "train":
        df.reset_index().to_excel(os.path.join(TO_TRAIN_PATH, file + '_TRAIN' + '_Aug.xlsx'), index=False)
        print(f"save augment train file to {file + '_TRAIN' + '_Aug.xlsx'}")
    elif stage == "test":
        df.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
        print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")
    elif stage == "valid":
        df.reset_index().to_excel(os.path.join(TO_VALID_PATH, file + '_VALID' + '_Aug.xlsx'), index=False)
        print(f"save augment valid file to {file + '_VALID' + '_Aug.xlsx'}")

def augmentEqual(df, file, stage="train", aug_ratio=1.0):
    """
    aug_ratio: 将较少的类增强至最多类别的比例
    """
    # 记录每个类别下的样本
    dic = {}
    maxLabelNum = 0
    maxLabel = 0
    for i in df['labels'].unique():
        dd = df[df['labels'] == i].reset_index().drop('index', axis=1)
        dic[i] = dd
        if len(dd) >= maxLabelNum:
            maxLabelNum = len(dd)
            maxLabel = i

    ls = list(dic.keys())
    print(f"len of df is {len(df)}")
    for k,v in dic.items():
        print(f"num of {k} is {len(v)}")

    ls.remove(maxLabel)
    for i in ls:
        dd = dic[i]
        # samplesNum = maxLabelNum - len(dd)
        varNum = len(dd)
        
        while True:
            if varNum >= int(maxLabelNum * aug_ratio):
                    break
            for ir, row in dd.iterrows():
                if varNum >= int(maxLabelNum * aug_ratio):
                    break
                # 比例：0.2、0.15、0.25
                out = BERTAugment(str(row['description']), 0.2)
                flag, text = out
                varNum += flag
                if flag == 1:
                    # print(f"ir: {ir}, flag:{flag} len(dd): {len(dd)}, target:{int(len(dd) * update_ratio)}, varNum: {varNum}")
                    new_row = copy.deepcopy(row)
                    new_row['description'] = text
                    df = df.append(new_row, ignore_index = True)

        print(f"before augment, num of {i} is: {len(dd)}")
        print(f"after augment, num of {i} is: {varNum}")
    if stage == "train":
        df.reset_index().to_excel(os.path.join(TO_TRAIN_PATH, file + '_TRAIN' + '_Aug.xlsx'), index=False)
        print(f"save augment train file to {file + '_TRAIN' + '_Aug.xlsx'}")
    elif stage == "test":
        df.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
        print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")
    elif stage == "valid":
        df.reset_index().to_excel(os.path.join(TO_VALID_PATH, file + '_VALID' + '_Aug.xlsx'), index=False)
        print(f"save augment valid file to {file + '_VALID' + '_Aug.xlsx'}")

GLOBAL_AUGMENTER = {
    "synonym_augmenter": naw.SynonymAug(aug_src='wordnet', aug_p=0.1),
    "contextual_word_embs_augmenter": naw.ContextualWordEmbsAug(model_path=MODELPATH, action="substitute", device="cuda:0"),
    "abstractive_summarization_augmenter": nas.AbstSummAug(model_path=T5MODELPATH, max_length=512, device="cuda:0"),
    "back_translation_augmenter": naw.BackTranslationAug(
            from_model_name=WMT19EN2DEMODELPATH,
            to_model_name=WMT19DE2ENMODELPATH,
            device="cuda:0",
            max_length=512)
}


# 先定义几种具体的数据增强方法
def synonym_augmenter(text):
    # aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.1)
    aug = GLOBAL_AUGMENTER["synonym_augmenter"]
    return aug.augment(text)

def random_delete_augmenter(text):
    aug = naw.RandomWordAug(action="delete", aug_p=0.1)
    return aug.augment(text)

def random_crop_augmenter(text):
    aug = naw.RandomWordAug(action="crop", aug_p=0.1)
    return aug.augment(text)

def random_swap_augmenter(text):
    aug = naw.RandomWordAug(action="swap", aug_p=0.1)
    return aug.augment(text)

# 单词级别增强
def contextual_word_embs_augmenter(text):
    # 使用BERT模型进行上下文相关的单词替换
    # aug = naw.ContextualWordEmbsAug(model_path=MODELPATH, action="substitute", device="cuda:0")
    aug = GLOBAL_AUGMENTER["contextual_word_embs_augmenter"]
    return aug.augment(text)

# 字符级别增强
def random_char_augmenter(text):
    # 随机替换，删除，插入或交换字符
    aug = nac.RandomCharAug(action="insert")
    return aug.augment(text)

# 句子级别增强
def abstractive_summarization_augmenter(text):
    # 使用T5模型生成摘要
    # aug = nas.AbstSummAug(model_path=T5MODELPATH, max_length=512, device="cuda:0")
    aug = GLOBAL_AUGMENTER["abstractive_summarization_augmenter"]
    return aug.augment(text)

def back_translation_augmenter(text):
    # aug = naw.BackTranslationAug(
    #         from_model_name=WMT19EN2DEMODELPATH,
    #         to_model_name=WMT19DE2ENMODELPATH,
    #         device="cuda:0",
    #         max_length=512)
    aug = GLOBAL_AUGMENTER["back_translation_augmenter"]
    return aug.augment(text)

# 定义通用的数据平衡+增强函数
def augmentEqual_NLPAug(df, file, stage="train", augmenters=None):
    assert augmenters is not None, "augmenters should be provided"
    
    dic = {}
    max_label_num = 0
    max_label = None
    
    # 找出最大的类别和对应数量
    for label in df['labels'].unique():
        label_df = df[df['labels'] == label]
        dic[label] = label_df
        if len(label_df) > max_label_num:
            max_label_num = len(label_df)
            max_label = label
            
    print(f"Length of dataframe is {len(df)}")
    print(f"Largest class is '{max_label}' with {max_label_num} samples.")

    # 数据增强，使每个类别数据量匹配最大类别
    for label, label_df in dic.items():
        # 计算需要增强多少数据
        diff = max_label_num - len(label_df)

        new_rows = []
        print(f"Augmenting class '{label} in {stage} dataset'...")
        while len(new_rows) < diff:
            # 在数据不足时继续增强
            for _, row in label_df.iterrows():
                augmented_text = row['description']
                # pass the nan text
                if augmented_text is None:
                    continue
                
                # 应用所有增强方法
                for augmenter in augmenters:
                    try:
                        augmented_text = augmenter(augmented_text)
                        new_rows.append(copy.deepcopy(row))
                        new_rows[-1]['description'] = augmented_text

                        if len(new_rows) >= diff:
                            break
                    except Exception:
                        print(f"{augmenter} failed on {augmented_text}")

                if len(new_rows) >= diff:
                    break
                    
        # 追加新的数据行到DataFrame
        df = df.append(new_rows, ignore_index=True)

    # 导出增强后的数据
    if stage == "train":
        df.reset_index().to_excel(os.path.join(TO_TRAIN_PATH, file + '_TRAIN' + '_Aug.xlsx'), index=False)
        print(f"save augment train file to {file + '_TRAIN' + '_Aug.xlsx'}")
    elif stage == "test":
        df.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
        print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")
    elif stage == "valid":
        df.reset_index().to_excel(os.path.join(TO_VALID_PATH, file + '_VALID' + '_Aug.xlsx'), index=False)
        print(f"save augment valid file to {file + '_VALID' + '_Aug.xlsx'}")

# 对测试集进行 投票式增强
def augmentTestVote(df, file, aug_num=2, max_attempts=10):
    augmented_df = pd.DataFrame()

    for ir, row in tqdm(df.iterrows(), total=len(df), desc='Augment Test Vote...'):
        # 先添加原始样本
        success_count = 0
        attempt_count = 0
        temp_rows = []  # 用于存储成功增强的临时样本

        while success_count < aug_num and attempt_count < max_attempts:
            out = BERTAugment(str(row['description']), 0.2)  # 这里的比例可以根据需要调整
            flag, text = out
            attempt_count += 1
            if flag == 1:
                success_count += 1
                new_row = copy.deepcopy(row)
                new_row['description'] = text
                temp_rows.append(new_row)

        # 在达到增强次数或尝试次数上限后，将成功增强的样本附加到DataFrame中
        if success_count == aug_num:
            augmented_df = augmented_df.append(row, ignore_index=True)
            for temp_row in temp_rows:
                augmented_df = augmented_df.append(temp_row, ignore_index=True)

    augmented_df.reset_index(drop=True, inplace=True)
    augmented_df.to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + f'_AugVote{aug_num}.xlsx'), index=False)
    print(f"save augment test file to {file}" + '_TEST' + f"_AugVote{aug_num}.xlsx")


def trainAugment(file):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx')) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    augment(df_train, file, stage="train")
    augment(df_test, file, stage="test")

# 仅增强训练集
def trainAugmentWithoutTest(file):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx')) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    augment(df_train, file, stage="train")
    # augment(df_test, file, isTrain=False)
    df_test.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
    print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")

# 训练集增强到等比例，测试集不增强
def trainEqualAugmentWithoutTest(file):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx')) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    augmentEqual(df_train, file, stage="train")
    df_test.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
    print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")

# 训练集、验证集 增强到等比例，测试集不增强
def trainValidEqualAugmentWithoutTest(file):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx'))
    df = df.dropna(axis=0,how='any') # drop all rows that have any NaN values

    split_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=128)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split_train_test.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    split_train_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=128)
    labels = df_train['labels']
    train_index, valid_index = None, None
    for _train_index, _valid_index in split_train_valid.split(df_train, labels):
        train_index = _train_index
        valid_index = _valid_index

    df = df_train.copy()
    df_train = df.iloc[train_index]
    df_valid = df.iloc[valid_index]

    augmenters = [
        # synonym_augmenter,
        # random_delete_augmenter,
        # random_crop_augmenter,
        # random_swap_augmenter,
        contextual_word_embs_augmenter,
        # random_char_augmenter,
        # abstractive_summarization_augmenter,
        # back_translation_augmenter
    ]

    # augmentEqual(df_train, file, stage="train", aug_ratio=1.0)
    # augmentEqual(df_valid, file, stage="valid")
    augmentEqual_NLPAug(df_train, file, stage="train", augmenters=augmenters)
    # augmentEqual_NLPAug(df_valid, file, stage="valid", augmenters=augmenters)
    # df_train.reset_index().to_excel(os.path.join(TO_TRAIN_PATH, file + '_TRAIN' + '_Aug.xlsx'), index=False)
    df_valid.reset_index().to_excel(os.path.join(TO_VALID_PATH, file + '_VALID' + '_Aug.xlsx'), index=False)
    print(f"save augment test file to {file + '_VALID' + '_Aug.xlsx'}")
    df_test.reset_index().to_excel(os.path.join(TO_TEST_PATH, file + '_TEST' + '_Aug.xlsx'), index=False)
    print(f"save augment test file to {file + '_TEST' + '_Aug.xlsx'}")

# 训练集增强到等比例，测试集每条样本增强到 1+ aug_num 条并连续排列
def trainEqualAugmentWithTestVoteAugment(file, aug_num=2):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx')) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    # augmentEqual(df_train, file, stage="train")
    augmentTestVote(df_test, file, aug_num=aug_num)


# 仅增强测试集
def testAugmentWithoutTrain(file):
    # 需要数据增强的文件
    # df = pd.read_csv(os.path.join(to_PATH, file + '_TRAIN_Bef.csv'))
    # df = pd.read_excel(os.path.join(PATH, file + '.xlsx'),sheet_name="Sheet2")
    df = pd.read_excel(os.path.join(PATH, file + '.xlsx')) 
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    labels = df['labels']
    train_index, test_index = None, None
    for _train_index, _test_index in split.split(df, labels):
        train_index = _train_index
        test_index = _test_index

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    # augment(df_train, file, isTrain=True)
    df_train.reset_index().to_excel(os.path.join(TO_TRAIN_PATH, file + '_TRAIN' + '_Aug.xlsx'), index=False)
    print(f"save augment train file to {file + '_TRAIN' + '_Aug.xlsx'}")
    augment(df_test, file, stage="test")


def trainAugment_2(file):
    # 需要数据增强的文件
    df = pd.read_csv(os.path.join(TO_TRAIN_PATH, file + '.csv'))
    # 记录每个类别下的样本
    dic = {}
    minLabelNum = float(inf)
    minLabel = 0
    for i in df['labels'].unique():
        dd = df[df['labels'] == i].reset_index().drop('index', axis=1)
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
            for ratio in RATIOS:
                out = BERTAugment(str(row['description']), ratio)
                flag, text = out
                if flag == 1:
                    row['description'] = text
                    df.loc[len(df)] = row

    df.reset_index().to_csv(os.path.join(TO_TRAIN_PATH, file + '_TRAIN_Aug_under.csv'), index=False)


def evalAugment(file,mode):
    # 需要数据增强的文件
    df = pd.read_csv(os.path.join(TO_TRAIN_PATH, file + f'_{mode}_Bef.csv')).reset_index()
    for i in df['label'].unique():
        dd = df[df['label'] == i].reset_index()
        # dd = df[df['label'] == i].reset_index().drop('index', axis=1)
        for ir, row in tqdm(dd.iterrows(), total=len(dd), desc=mode + f' Augment(class {i})...'):
            for ratio in RATIOS:
                out = BERTAugment(str(row['description']), ratio)
                _, text = out
                row['description'] = text
                df.loc[len(df)] = row
        df.to_csv(os.path.join(TO_TRAIN_PATH, file + '_TEST_Aug.csv'), index=False)
        # df.to_csv(os.path.join(TO_TRAIN_PATH, file + '_DEV_Aug.csv'), index=False)
#         dataframe.rename(columns = {"old_name": "new_name"})
# dataframe.rename(columns = {"old1": "new1", "old2":"new2"},  inplace=True)


if __name__ == "__main__":
    os.environ['http_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    os.environ['https_proxy'] = 'http://nbproxy.mlp.oppo.local:8888'
    # text = """Broke after creating the new Streamlit repo. Test runs aren't being recorded even though I have the `record key` environment variable set both in my personal CircleCI and in the Streamlit org CircleCI."""
    # text = """sad as ro sa fda dsl apple bannana <span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">Steps to repro:聽</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">&#10;</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">1. Run `examples/reference.py`聽</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">&#10;</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">2. When done, rerun it.聽</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">&#10;&#10;</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">**Expected:** on rerun, all elements fade out and then become opaque one by one even before the run is done.聽</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">&#10;</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">**Actual:** on rerun, all elements fade out and only become opaque when the entire run is done.聽</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">&#10;&#10;</span><span style="font-family: 瀹嬩綋;font-size: 15px;color: #000000;">I believe this bug was introduced with the Sidebar code.</span>"""
    # BERTAugment(text, 0.2)
    # exit(1)

    # evalAugment('complete2', 'TEST')
    print("----------------------------------------------------------------")
    # trainValidEqualAugmentWithoutTest('streamlit_new_clean')
    # trainValidEqualAugmentWithoutTest('Real-Time-Voice-Cloning_newlabel_clean')
    # trainValidEqualAugmentWithoutTest('pytorch-CycleGAN-and-pix2pix_newlabel_clean')
    # trainValidEqualAugmentWithoutTest('faceswap_newlabel_clean')
    # trainValidEqualAugmentWithoutTest('deepfacelab_newlabel_clean')
    # trainValidEqualAugmentWithoutTest('openpose_newlabel_clean')
    
    # trainValidEqualAugmentWithoutTest('caffe_newlabel_clean') # train+val:test  0.8:0.2 train:val 0.8:0.2
    # trainValidEqualAugmentWithoutTest('caffe_newlabel_clean') # train+val:test  0.75:0.25 train:val 0.8:0.2
    trainValidEqualAugmentWithoutTest('pytorch_newlabel_clean') # train+val:test  0.75:0.25 train:val 0.8:0.2
    # trainValidEqualAugmentWithoutTest('tensorflow_newlabel_clean')

    # trainAugment('pytorch-CycleGAN-and-pix2pix')
    # trainAugment('streamlit1')
    # trainAugment('faceswap_newlabel')
    # trainAugment('deepfacelab_newlabel')
    # trainAugment('deepfacelab')
    # trainAugment('faceswap')
    # trainAugment('streamlit')
    # trainAugmentWithoutTest('streamlit')
    # trainEqualAugmentWithTestVoteAugment('streamlit')
    # trainAugment('streamlit_newlabel')
    # trainAugment('streamlit_clean')
    # trainAugmentWithoutTest('streamlit_clean')
    # trainEqualAugmentWithoutTest('streamlit_new_clean')
    # trainValidEqualAugmentWithoutTest('streamlit_new_clean')
    
    # trainEqualAugmentWithoutTest('deepfacelab_newlabel_clean') # ok
    # trainEqualAugmentWithoutTest('EasyOCR_newlabel_clean') # fillna ok
    # trainValidEqualAugmentWithoutTest('EasyOCR_newlabel_clean') # fillna ok
    # trainValidEqualAugmentWithoutTest('EasyOCR_newlabel_with_comments_clean') # fillna ok
    # trainEqualAugmentWithoutTest('faceswap_newlabel_clean') # ok
    # trainEqualAugmentWithoutTest('jetson-inference_newlabel_clean') #error
    # trainEqualAugmentWithoutTest('Real-Time-Voice-Cloning_newlabel_clean') # 补空位ok
    # trainEqualAugmentWithoutTest('recommenders_newlabel_clean') # ok
    # trainEqualAugmentWithoutTest('TTS_newlabel_clean') ok
    # trainEqualAugmentWithTestVoteAugment('streamlit_new_clean')
    # testAugmentWithoutTrain('streamlit_clean_1')
    # trainAugment('Real-Time-Voice-Cloning')
    # trainAugment('Real-Time-Voice-Cloning_newlabel')
    print("----------------------------------------------------------------")
    # trainAugment('Real-Time-Voice-Cloning')
    # trainAugment('EasyOCR')
    # trainAugment('recommenders1')
    # trainAugment('streamlit1')
    # trainAugment_2('couple3')
    # evalAugment('couple3', 'TEST')

    del os.environ['http_proxy']   #用完需要del代理，否则训练的所有流量都走代理访问，有安全风险
    del os.environ['https_proxy']

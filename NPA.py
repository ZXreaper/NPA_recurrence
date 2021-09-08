# -*- coding: utf-8 -*-
"""
======================
@author  : Zhang Xu
@time    : 2021/9/7:11:15
@email   : zxreaper@foxmail.com
@content : 复现NPA的代码
======================
"""
import csv
import random
import nltk
from nltk.tokenize import word_tokenize
import datetime
import time
import random
import itertools
import numpy as np
import pickle
from numpy.linalg import cholesky

def newsample(nnn,ratio):
    if ratio > len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1), ratio)
    else:
        return random.sample(nnn, ratio)

def preprocess_user_file(file='ClickData_sample.tsv', npratio=4):
    '''
    处理用户数据文件。
    :param file:        文件名
    :param npratio:     negative sampling 的比例。默认1份正样本，4份负样本
    :return:
    '''
    userid_dict = {}
    with open(file) as f:
        userdata = f.readlines()
    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        if userid not in userid_dict:
            userid_dict[userid] = len(userid_dict)

    all_train_id = []
    all_train_pn = []
    all_label = []

    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []

    all_user_pos = []
    all_test_user_pos = []

    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        if len(line) == 4:
            impre = [x.split('#TAB#') for x in line[2].split('#N#')]    # impre 就是一个impression，即一个看到的界面
        if len(line) == 3:
            impre = [x.split('#TAB#') for x in line[2].split('#N#')]

        trainpos = [x[0].split() for x in impre]        # 正
        trainneg = [x[1].split() for x in impre]        # 负

        poslist = list(itertools.chain(*(trainpos)))    # 正样本news列表
        neglist = list(itertools.chain(*(trainneg)))    # 负样本news列表

        if len(line) == 4:
            testimpre = [x.split('#TAB#') for x in line[3].split('#N#')]
            testpos = [x[0].split() for x in testimpre]
            testneg = [x[1].split() for x in testimpre]

            for i in range(len(testpos)):
                sess_index = []
                sess_index.append(len(all_test_pn))
                posset = list(set(poslist))
                allpos = [int(p) for p in random.sample(posset, min(50, len(posset)))[:50]]
                allpos += [0] * (50 - len(allpos))

                for j in testpos[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(1)
                    all_test_id.append(userid_dict[userid])
                    all_test_user_pos.append(allpos)

                for j in testneg[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(0)
                    all_test_id.append(userid_dict[userid])
                    all_test_user_pos.append(allpos)
                sess_index.append(len(all_test_pn))
                all_test_index.append(sess_index)

        for impre_id in range(len(trainpos)):
            for pos_sample in trainpos[impre_id]:

                pos_neg_sample = newsample(trainneg[impre_id], npratio)
                pos_neg_sample.append(pos_sample)
                temp_label = [0] * npratio + [1]
                temp_id = list(range(npratio + 1))
                random.shuffle(temp_id)

                shuffle_sample = []
                shuffle_label = []
                for id in temp_id:
                    shuffle_sample.append(int(pos_neg_sample[id]))
                    shuffle_label.append(temp_label[id])

                posset = list(set(poslist) - set([pos_sample]))
                allpos = [int(p) for p in random.sample(posset, min(50, len(posset)))[:50]]
                allpos += [0] * (50 - len(allpos))
                all_train_pn.append(shuffle_sample)
                all_label.append(shuffle_label)
                all_train_id.append(userid_dict[userid])
                all_user_pos.append(allpos)

    all_train_pn = np.array(all_train_pn, dtype='int32')
    all_label = np.array(all_label, dtype='int32')
    all_train_id = np.array(all_train_id, dtype='int32')
    all_test_pn = np.array(all_test_pn, dtype='int32')
    all_test_label = np.array(all_test_label, dtype='int32')
    all_test_id = np.array(all_test_id, dtype='int32')
    all_user_pos = np.array(all_user_pos, dtype='int32')
    all_test_user_pos = np.array(all_test_user_pos, dtype='int32')
    return userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index

def preprocess_news_file(file='DocMeta_sample.tsv'):
    with open(file) as f:
        newsdata = f.readlines()

    # news的结构是：{news的标号:[文章分类,文章具体分类,[文章标题的tockenize]], }
    news = {}
    for newsline in newsdata:
        line = newsline.strip().split('\t')
        news[line[1]] = [line[2], line[3], word_tokenize(line[6].lower())]

    # word_dict_raw的结构是{'单词': [该单词的标号， 出现次数]}
    # 当单词之前没有出现过的时候，该单词的标号应该等于现在总的词数+1，即len(word_dict_raw), 出现次数为1
    # 初始化的时候，初始化一个‘PADDING’，标号为0， 次数为999999， 还不知道为什么？
    word_dict_raw = {'PADDING': [0, 999999]}

    for docid in news:
        for word in news[docid][2]:     # 取出docid这篇文章标题的每个词（tockenizer处理后的）
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]

    # word_dict的结构是 {'单词': [单词新的标号, 出现次数]}
    word_dict = {}
    for i in word_dict_raw:
        if word_dict_raw[i][1] >= 2:    # 为什么出现次数大于等于2的要被提取出来。
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    print(len(word_dict), len(word_dict_raw))


    # news_words的结构是: {[第一篇news的向量，长度为30]，[第二篇news的向量，长度为30]，...}。 第一个单词全部是0，即长度为30的零向量
    # news_index的结构是: {news的id: news的标号, ...}
    news_words = [[0] * 30]
    news_index = {'0': 0}
    for newsid in news:
        word_id = []
        news_index[newsid] = len(news_index)
        for word in news[newsid][2]:
            if word in word_dict:
                word_id.append(word_dict[word][0])
        word_id = word_id[:30]                                      # 截取。一个标题最多30个单词。
        news_words.append(word_id + [0] * (30 - len(word_id)))      # 填充。不足30个单词的标题，补0。
    news_words = np.array(news_words, dtype='int32')
    return word_dict, news_words, news_index

def get_embedding(word_dict):
    '''

    :param word_dict:  word_dict结构是 {'单词'：[单词的标号，单词出现的次数], ... }
    :return:
    '''
    embedding_dict = {}
    cnt = 0
    with open('./glove.840B.300d.txt', 'rb') as f:
        '''
            glove的形状是
                    标号     单词/符号     向量（len = 300）
                     0          .           [...]
                     1          the         [...]  
        '''
        linenb = 0
        while True:
            line = f.readline()
            if len(line) == 0:          # 到文件末尾了
                break
            line = line.split()         # 原txt以空格分隔
            word = line[0].decode()
            linenb += 1
            if len(word) != 0:          # 如果遇到单词是空格，就跳过
                vec = [float(x) for x in line[1:]]
                if word in word_dict:
                    embedding_dict[word] = vec
                    if cnt % 1000 == 0:
                        print(cnt, linenb, word)
                    cnt += 1

    embedding_matrix = [0]*len(word_dict)
    cand = []
    for i in embedding_dict:
        embedding_matrix[word_dict[i][0]] = np.array(embedding_dict[i], dtype='float32')
        cand.append(embedding_matrix[word_dict[i][0]])
    cand = np.array(cand, dtype='float32')
    mu = np.mean(cand, axis=0)
    Sigma = np.cov(cand.T)
    norm = np.random.multivariate_normal(mu, Sigma, 1)      # 多元正态分布
    for i in range(len(embedding_matrix)):
        if type(embedding_matrix[i]) == int:                # 正常情况下embedding_matrix[i]的维度应该是长度为300的向量。什么情况下（或者说为什么）会是int？
                                                            # 因为embedding_matrix之前被初始化成一个1维的向量，因此当这个单词没有在前面被制成一个向量的时候，该单词对应的位置就是0，因此就会是int类型
            embedding_matrix[i] = np.reshape(norm, 300)     # 如果是int，那么就将它填充成上面构建的正态分布，长度为300的向量
    embedding_matrix[0] = np.zeros(300, dtype='float32')
    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    print(embedding_matrix.shape)
    return embedding_matrix

# 评价标准
# -----------------------------------------------------------------
def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)
# -----------------------------------------------------------------

userid_dict, all_train_pn, all_label, all_train_id, all_test_pn, all_test_label, all_test_id, all_user_pos, all_test_user_pos, all_test_index = preprocess_user_file()

word_dict, news_words, news_index = preprocess_news_file()

embedding_mat = get_embedding(word_dict)

# 产生 train batch data
def generate_batch_data_train(all_train_pn,all_label,all_train_id,batch_size):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y=all_label
    batches = [inputid[range(batch_size*i, min(len(y), batch_size*(i+1)))] for i in range(len(y)//batch_size+1)]

    while (True):
        for i in batches:
            candidate = news_words[all_train_pn[i]]
            candidate_split=[candidate[:,k,:] for k in range(candidate.shape[1])]
            browsed_news=news_words[all_user_pos[i]]
            browsed_news_split=[browsed_news[:,k,:] for k in range(browsed_news.shape[1])]
            userid=np.expand_dims(all_train_id[i],axis=1)
            label=all_label[i]
            yield (candidate_split +browsed_news_split+[userid], label)

# 产生 test batch size
def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            userid = np.expand_dims(all_test_id[i], axis=1)
            label = all_label[i]

            yield ([candidate] + browsed_news_split + [userid], label)


'''模型训练'''

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *

npratio = 4               # negative sampling 的比例
results = []

MAX_SENT_LENGTH = 30      # 最长的文章的单词数量
MAX_SENTS = 50            # 最多几篇文章

# 构建 qw 、 qd
# ----------------------------------------------------------------------------------------------------------------------
user_id = Input(shape=(1,), dtype='int32')
user_embedding_layer = Embedding(len(userid_dict), 50, trainable=True)
user_embedding = user_embedding_layer(user_id)
user_embedding_word = Dense(200, activation='relu')(user_embedding)
user_embedding_word = Flatten()(user_embedding_word)
user_embedding_news = Dense(200, activation='relu')(user_embedding)
user_embedding_news = Flatten()(user_embedding_news)
# ----------------------------------------------------------------------------------------------------------------------

# news encoder
# ----------------------------------------------------------------------------------------------------------------------
news_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
embedded_sequences = embedding_layer(news_input)
embedded_sequences = Dropout(0.2)(embedded_sequences)

cnnouput = Convolution1D(nb_filter=400, filter_length=3, padding='same', activation='relu', strides=1)(embedded_sequences)
cnnouput = Dropout(0.2)(cnnouput)

# news encoder中的 personalized attention部分
attention_a = Dot((2, 1))([cnnouput, Dense(400, activation='tanh')(user_embedding_word)])
attention_weight = Activation('softmax')(attention_a)
news_rep = keras.layers.Dot((1, 1))([cnnouput, attention_weight])
newsEncoder = Model([news_input, user_id], news_rep)
# ----------------------------------------------------------------------------------------------------------------------

# clicked paper
all_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_news_rep = [newsEncoder([news, user_id]) for news in all_news_input]
browsed_news_rep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsed_news_rep], axis=1)

attention_news = keras.layers.Dot((2, 1))([browsed_news_rep, Dense(400, activation='tanh')(user_embedding_news)])
attention_weight_news = Activation('softmax')(attention_news)
user_rep = keras.layers.Dot((1, 1))([browsed_news_rep, attention_weight_news])

# candidates paper
candidates = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]
candidate_vecs = [newsEncoder([candidate, user_id]) for candidate in candidates]

logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))

model = Model(candidates + all_news_input + [user_id], logits)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

candidate_one = keras.Input((MAX_SENT_LENGTH,))
candidate_one_vec = newsEncoder([candidate_one, user_id])
score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one] + all_news_input + [user_id], score)

for ep in range(3):
    traingen = generate_batch_data_train(all_train_pn, all_label, all_train_id, 100)
    model.fit_generator(traingen, epochs=1, steps_per_epoch=len(all_train_id) // 100)
    testgen = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, 100)
    click_score = model_test.predict_generator(testgen, steps=len(all_test_id) // 100, verbose=1)
    from sklearn.metrics import roc_auc_score

    all_auc = []
    all_mrr = []
    all_ndcg = []
    all_ndcg2 = []
    for m in all_test_index:
        if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
            all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
            all_ndcg.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
            all_ndcg2.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
    results.append([np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2)])
    print(np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2))
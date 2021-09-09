# -*- coding: utf-8 -*-
"""
======================
@author  : Zhang Xu
@time    : 2021/9/8:16:29
@email   : zxreaper@foxmail.com
@content : tensorflow subclassing 复现 NPA
======================
"""
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *
from keras import backend as K

npratio = 4

MAX_SENT_LENGTH = 30      # 一篇news的单词数量
MAX_SENTS = 50            # 一个用户的点击的news的数量

# news encoder
# 输入：user id, 1篇news的信息
# 输出：news representation
class NewsEncoder(tf.keras.Model):

    def __init__(self):
        super(NewsEncoder, self).__init__(name='NewsEncoder')

        # user_id 部分
        self.userid_input_layer = Input()
        self.userid_embedding_layer = Embedding()
        self.userid_dense_layer = Dense()
        self.userid_flatten_layer = Flatten()

        # news 部分
        self.news_input_layer = Input()
        self.news_embedding_layer = Embedding()
        self.news_conv_layer = Conv1D()
        self.news_dropout_layer_1 = Dropout(0.2)
        self.news_dropout_layer_2 = Dropout(0.2)

        # personalized attention 部分
        self.pa_dense_layer = Dense()
        self.pa_2_1_dot_layer = Dot()
        self.pa_softmax_layer = Activation('softmax')
        self.pa_1_1_dot_layer = Dot()

    def call(self, inputs):
        '''多输入：输入 user_id、 news_input'''
        '''输入单个用户的 user id 和 一篇 news 的信息'''
        user_id, news_input = inputs[0], inputs[1]

        # qw
        x1 = self.userid_input_layer(user_id)
        x1 = self.userid_embedding_layer(x1)
        x1 = self.userid_dense_layer(x1)
        qw = self.userid_flatten_layer(x1)

        # news representation
        x2 = self.news_input_layer(news_input)
        x2 = self.news_embedding_layer(x2)
        x2 = self.news_dropout_layer_1(x2)
        x2 = self.news_conv_layer(x2)
        x2 = self.news_dropout_layer_2(x2)

        # personalized attention
        qw = self.pa_dense_layer(qw)
        attention_a = self.pa_2_1_dot_layer([x2, qw])
        attention_weight = self.pa_softmax_layer(attention_a)
        news_rep = self.pa_1_1_dot_layer([x2, attention_weight])

        return news_rep


# NPA
# 输入：user id 和 该用户所有的 clicked news（N篇） 和 candidate news（K篇）
# 输出：对K篇 candidate news 做出预测，分别给出点击的概率
class NPA(tf.keras.Model):

    def __init__(self):
        super(NPA, self).__init__(name='NPA')

        # user id 部分
        self.userid_input_layer = Input()
        self.userid_embedding_layer = Embedding()
        self.userid_dense_layer = Dense()
        self.userid_flatten_layer = Flatten()

        # clicked news 部分
        self.clickednews_input_layer = [Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
        self.clickednews_encoder = [NewsEncoder() for _ in range(MAX_SENTS)]
        self.clickednews_dense_layer = Dense()
        self.clickednews_2_1_dot_layer = Dot((2, 1))
        self.clickednews_softmax_layer = Activation('softmax')
        self.clickednews_1_1_dot_layer = Dot((1, 1))

        # candidate news 部分
        self.candidatenews_input_layer = [Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]
        self.candidatenews_encoder = [NewsEncoder() for _ in range(1 + npratio)]

        # click prediction
        self.cp_dot_layer = dot()
        self.cp_concatenate = concatenate()
        self.cp_activation_layer = Activation('softmax')


    def call(self, inputs):
        user_id, clicked_news, candidate_news = inputs[0], inputs[1], inputs[2]

        # qd
        x1 = self.userid_input_layer(user_id)
        x1 = self.userid_embedding_layer(x1)
        x1 = self.userid_dense_layer(x1)
        qd = self.userid_flatten_layer(x1)

        # clicked news
        clicked_news_vec = [0]*MAX_SENTS
        for i in range(len(clicked_news)):
            xx = self.clickednews_input_layer[i](clicked_news[i])
            clicked_news_vec[i] = self.clickednews_encoder[i]([user_id, xx])
        clicked_news_rep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in clicked_news_vec], axis=1)

        # qd 与 click_news_rep 进行 personalized attention
        news_temp_dense = self.clickednews_dense_layer(qd)
        attention_news = self.clickednews_2_1_dot_layer([clicked_news_rep, news_temp_dense])
        attention_news_weight = self.clickednews_softmax_layer(attention_news)
        user_rep = self.clickednews_1_1_dot_layer([clicked_news_rep, attention_news_weight])

        # candidate news
        candidate_news_vec = [0]*(1+npratio)
        for i in range(len(candidate_news)):
            xx = self.candidatenews_input_layer[i](candidate_news[i])
            candidate_news_vec[i] = self.candidatenews_encoder[i]([user_id, xx])

        # click prediction
        # candidate news representation 与 user representation 进行 dot 和 softmax
        logits = [self.cp_dot_layer([user_rep, candidate_news], axes=-1) for candidate_news in candidate_news_vec]
        logits = self.cp_activation_layer(self.cp_concatenate(logits))

        return logits
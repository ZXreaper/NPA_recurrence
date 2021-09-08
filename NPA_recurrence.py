# -*- coding: utf-8 -*-
"""
======================
@author  : Zhang Xu
@time    : 2021/9/8:16:29
@email   : zxreaper@foxmail.com
@content : 
======================
"""
import tensorflow as tf
from tensorflow.keras import *
from tensorflow.keras.layers import *

# news encoder
class NewsEncoder(tf.keras.Model):

    def __init__(self, user_embedding_word):
        super(NewsEncoder, self).__init__(name='NewsEncoder')

        self.user_embedding_word = user_embedding_word          # qw
        self.news_input_layer = Input()
        self.embedding_layer = Embedding()
        self.cnn_layer = Conv1D()

    def call(self, inputs):
        x = self.news_input_layer(inputs)
        x = self.embedding_layer(x)
        x = self.cnn_layer(x)
        x = Dropout(0.2)(x)
        # Personalized Attention
        attention_qw_cnn = Dot((2, 1))([x, Dense(400, activation='tanh')(self.user_embedding_word)])
        attention_qw_cnn_weight = Attention('softmax')(attention_qw_cnn)
        news_rep = Dot((1, 1))([x, attention_qw_cnn_weight])
        return news_rep

# user encoder 直接放在模型中去实现

# NPA
class NPA(tf.keras.Model):

    def __init__(self):
        self.user_id_input_layer = Input(shape=(1,), dtype='int32')
        self.user_embedding_layer = Embedding()
        self.user_dense_layer = Dense()


    def call(self, userid, clickednews, candidatenews):
        '''输入一： userid 构建 qw，qd'''
        user_id = self.user_id_input_layer(userid)
        user_embedding = self.user_embedding_layer(user_id)
        user_embedding_word = self.user_dense_layer(user_embedding)
        # qw
        user_embedding_word = Flatten()(user_embedding_word)
        user_embedding_news = self.user_dense_layer(user_embedding)
        # qd
        user_embedding_news = Flatten()(user_embedding_news)

        '''输入二： 输入clicked news'''
        output_clicked_news = NewsEncoder(user_embedding_word)(clickednews)

        '''qd与output_clicked_news的 personalized attention'''


        '''输入三：输入candidatenews'''
        output_candidate_news = NewsEncoder(user_embedding_news)(candidatenews)


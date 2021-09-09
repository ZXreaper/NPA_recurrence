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

# news encoder缺少Personalized Attention部分，放在后面NPA模型中实现
class NewsEncoder(tf.keras.Model):

    def __init__(self):
        super(NewsEncoder, self).__init__(name='NewsEncoder')

        self.news_input_layer = Input()
        self.embedding_layer = Embedding()
        self.cnn_layer = Conv1D()
        self.dropout_layer = Dropout(0.2)

    def call(self, inputs):
        x = self.news_input_layer(inputs)
        x = self.embedding_layer(x)
        x = self.dropout_layer(x)
        x = self.cnn_layer(x)
        x = self.dropout_layer(x)
        return x


# NPA
class NPA(tf.keras.Model):

    def __init__(self):
        self.user_id_input_layer = Input(shape=(1,), dtype='int32')
        self.user_embedding_layer = Embedding()
        self.user_dense_layer = Dense()
        self.flatten_layer = Flatten()
        self.news_encoder = NewsEncoder()
        self.dot_2_1_layer = Dot((2, 1))
        self.dense_layer = Dense()
        self.softmax_layer = Activation('softmax')
        self.dot_1_1_layer = Dot((1,1))


    def call(self, userid, clickednews, candidatenews):
        '''输入一： userid 构建 qw，qd'''
        user_id = self.user_id_input_layer(userid)
        user_embedding = self.user_embedding_layer(user_id)
        user_embedding_word = self.user_dense_layer(user_embedding)
        # qw
        user_embedding_word = self.flatten_layer(user_embedding_word)
        user_embedding_news = self.user_dense_layer(user_embedding)
        # qd
        user_embedding_news = self.flatten_layer(user_embedding_news)

        '''输入二： 输入clicked news'''
        output_clicked_news = self.news_encoder(clickednews)

        '''qw与output_clicked_news的 personalized attention'''
        temp_dense = self.dense_layer(user_embedding_word)
        attention_a = self.dot_2_1_layer([output_clicked_news,temp_dense])
        attention_weight = self.softmax_layer(attention_a)
        news_rep = self.dot_1_1_layer([output_clicked_news, attention_weight])

        '''qd与news_rep的 personalized attention'''



        '''输入三：输入candidatenews'''
        output_candidate_news = NewsEncoder(user_embedding_news)(candidatenews)


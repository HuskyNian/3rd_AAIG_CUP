import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow_addons.metrics import F1Score

# 因天池提供的基础镜像只有tensorflow，若使用pytorch请自行安装


def train_sample(train_path):
    """
    读取训练数据并返回numpy数组，这里只采用了embedding features
    """
    data = pd.read_csv(train_path, header=None)
    data = data.loc[data[5]>=0]
    x = data[4].str.split(" ", expand=True).values.astype(np.float32)
    y = data[5].values
    return x, y


def build_model():
    """
    此处若要使用其它数据输入，请自行修改天池镜像中的数据处理部分
    构建mlp模型，注意输入为(batch_size, 75)，输出为(batch_size, 1)
    线上评测要求CPU训练40min内（共100万训练样本），CPU推理每个样本0.5s内
    """
    input = Input((152,))
    x = BatchNormalization()(input)
    #x = Dense(64,"relu")(input)
    for u in [64,128]:
        x = Dense(u, "relu")(x)
    x = Dropout(0.5)(Dense(64,"relu")(x))
    output = Dense(1, "sigmoid")(x)
    model = Model(input, output)
    f1score = F1Score(num_classes=2,average='micro',threshold=0.5)
    model.compile(Adam(learning_rate=0.0005), "binary_crossentropy", [f1score,"accuracy"])
    model.summary()
    return model


def train(train_path, model_dir, save_name):
    """
    天池的脚本会直接调这个函数，不要改train_path, model_dir, save_name
    我们只用管训练，只要模型的输入输出形状一致，推理是自动完成的
    注意：必须采用.pb格式保存到model_dir + "/frozen_model"！
    """
    x, y = train_sample(train_path)
    model = build_model()
    checkpoint = ModelCheckpoint(
        model_dir + "/frozen_model", "val_f1_score", save_best_only=True
    )
    model_save_path = os.path.join(model_dir,save_name)
    if not os.path.exists('/host/model/frozen_model/save_model.pb'):
        print('this is first time')
    else:
        print('this is second time')
    model.fit(
        x,
        y,
        batch_size=128,
        epochs=25,
        verbose=2,
        validation_split=0.1,
        callbacks=[checkpoint],
    )
    os.remove(model_dir + "/frozen_model/keras_metadata.pb")
    # 注意：若不删除这个文件有几率导致flink ai flow无法load模型！

#if __name__ == '__main__':  
#    model_dir = './model'
#    save_name = 'saved_model'
#    train_path = '/tcdata/train0.csv'

#    train(train_path, model_dir, save_name)

# -*- coding: utf-8 -*-
# 模型训练
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import numpy as np
from keras.utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from att import Attention
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from operator import itemgetter

from load_data import get_train_test_pd
from bert.extract_feature import BertVector
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# 读取文件并进行转换
train_df, dev_df = get_train_test_pd()
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=128)#检查batch_size对GPU使用率的影响 """batch_size=128,"""
print('begin encoding')
f= lambda text: bert_model.encode([text])["encodes"][0]

train_df['x'] = train_df['text'].apply(f)
dev_df['x'] = dev_df['text'].apply(f)
print('end encoding')

# 读取关系对应表
with open('./data/rel_dict.json', 'r', encoding='utf-8') as f:
    label_id_dict = json.loads(f.read())
print(label_id_dict)
id2rel = {}
for i in label_id_dict:
    id2rel[label_id_dict[i]] = i
print(id2rel)
# 训练集和测试集
x_train = np.array([vec for vec in train_df['x']])
x_dev = np.array([vec for vec in dev_df['x']])
y_train = np.array([vec for vec in train_df['label']])
y_dev = np.array([vec for vec in dev_df['label']])
# print('x_train: ', x_train.shape)

# 将类型y值转化为ont-hot向量
num_classes = 11
y_train = to_categorical(y_train, num_classes)
y_dev = to_categorical(y_dev, num_classes)
# 模型结构：BERT + 双向GRU + Attention + FC
inputs = Input(shape=(128, 768, ))
gru = Bidirectional(GRU(128, dropout=0.1, return_sequences=True))(inputs)
attention = Attention(32)(gru)
output = Dense(num_classes, activation='softmax')(attention)
model = Model(inputs, output)

# 模型可视化
# from keras.utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, mode='max')

# 如果原来models文件夹下存在.h5文件，则全部删除
# model_dir = 'models/models_apptype'
# if os.listdir(model_dir):
#     for file in os.listdir(model_dir):
#         os.remove(os.path.join(model_dir, file))

# 保存最新的val_acc最好的模型文件
filepath="models_append_all/{epoch:02d}-{val_accuracy:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')

# 模型训练以及评估
history = model.fit(x_train, y_train, validation_data=(x_dev, y_dev), batch_size=16, epochs=30, callbacks=[early_stopping, checkpoint])
# model.save('people_relation.h5')

print('在测试集上的效果：', model.evaluate(x_dev, y_dev))

# 改变了关系类型顺序，wait release
sorted_label_id_dict = sorted(label_id_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]

# 输出每一类的classification report
y_pred = model.predict(x_dev, batch_size=32)#y_pred = model.predict(x_dev, batch_size=128)
true=[]
pre=[]
for i in range(len(y_dev.argmax(axis=1))):
    true.append(id2rel[y_dev.argmax(axis=1)[i]] )
    pre.append(id2rel[y_pred.argmax(axis=1)[i]] )
print(classification_report(true ,pre, target_names=values))

# 绘制loss和acc图像
plt.subplot(2, 1, 1)
epochs = len(history.history['loss'])
plt.plot(range(epochs), history.history['loss'], label='loss')
plt.plot(range(epochs), history.history['val_loss'], label='val_loss')
plt.legend()

plt.subplot(2, 1, 2)
epochs = len(history.history['accuracy'])
plt.plot(range(epochs), history.history['accuracy'], label='acc')
plt.plot(range(epochs), history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig("loss_acc.png")

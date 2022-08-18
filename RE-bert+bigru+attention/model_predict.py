# -*- coding: utf-8 -*-
# 模型预测
from operator import itemgetter
import os, json
import numpy as np
from bert.extract_feature import BertVector
from keras.models import load_model
from att import Attention
import sklearn
# 加载训练效果最好的模型
model_dir = 'models_append_all'
files = os.listdir(model_dir)
os.environ['CUDA_VISIBLE_DEVICES']="0"
models_path = [os.path.join(model_dir, _) for _ in files]
best_model_path = sorted(models_path, key=lambda x: str(x.split('-')[-1].replace('.h5', '')), reverse=True)
print(best_model_path)
best_model_path='models_append_all/08-0.9341.h5'
model = load_model(best_model_path, custom_objects={"Attention": Attention})

# 示例语句及预处理
# text1 = '徐寿#华蘅芳#3年后，徐寿和华蘅芳同心协力，制造出了我国第一艘机动木质轮船，“长五十余尺，每一时能行四十余里”。'
# per1, per2, doc = text1.split('#')
# text = '$'.join([per1, per2, doc.replace(per1, len(per1)*'#').replace(per2, len(per2)*'#')])
# print(text)


with open('data_append_all/test.txt', 'r', encoding='utf-8') as f:
    texts = f.readlines()
with open('data/rel_dict.json', 'r', encoding='utf-8') as f:
    rel_dict = json.load(f)
predict_list=[]
true_list=[]
# 利用BERT提取句子特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=128)
for text in texts:
    if text[1]!=' ':
        true_label = int(text[0:2])
        text=text[3:]
    else:
        true_label = int(text[0:1])
        text = text[2:]
    # print(true_label,text)
    vec = bert_model.encode([text])["encodes"][0]
    x_train = np.array([vec])
    # 模型预测并输出预测结果
    predict=(model.predict(x_train))
    y = np.argmax(predict[0])
    id_rel_dict = {v:k for k,v in rel_dict.items()}
    # print(id_rel_dict)
    # print('原文: %s' % text)
    # print('真实值: %s' % id_rel_dict[true_label])
    # print('预测值: %s' % id_rel_dict[y])
    print(id_rel_dict[y])
    true_list.append(id_rel_dict[true_label])
    predict_list.append(id_rel_dict[y])

sorted_label_id_dict = sorted(rel_dict.items(), key=itemgetter(1))
values = [_[0] for _ in sorted_label_id_dict]
print(sklearn.metrics.classification_report(true_list, predict_list, target_names=values))
#coding=utf-8

from tempfile import template
import arguments
args = arguments.parse_args()

import os
os.chdir(os.path.dirname(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from transformers import T5Config, T5ForConditionalGeneration

from utils import *
import random
import numpy as np

seed_val = args.seed
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

prediction_dataset = ProductFromFile(file_path_input = args.query_file_path, \
        file_path_output = args.candidate_file_path, 
        template = args.template, 
        MAX_LEN_in = args.MAX_LEN_in, MAX_LEN_out = args.MAX_LEN_out, 
        Tokenizer_PATH = "./models/google/mt5-base")
prediction_dataloader = DataLoader(prediction_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
# T5_PATH = args.model_path # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
T5_PATH = '/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/models/google/mt5-base'
t5_config = T5Config.from_pretrained(T5_PATH)
model = T5ForConditionalGeneration.from_pretrained('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/jr_train/train/model/jr_v1/mt5-base-epoch-1-6', config=t5_config).to(DEVICE)

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# 预测测试集
print('Predicting labels for {:,} test sentences...'.format(prediction_dataset.__len__()))
model.eval()

# Tracking variables 
# query_word, candidate_word, template, predictions = [], [], [], []
all_losses = torch.empty(0)

# 预测
t0 = time.time()
for step, batch in enumerate(prediction_dataloader):
    if step % 40 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}'.format(step, prediction_dataset.__len__() // args.batch_size, elapsed))

    # 将数据加载到 gpu 中
    b_input_ids = batch[0].to(DEVICE)
    b_labels = batch[1].to(DEVICE)

    # 不需要计算梯度
    with torch.no_grad():
        # 前向传播，获取预测结果
        losses = model(input_ids=b_input_ids, labels=b_labels, reduction='none').loss

    # 将结果加载到 cpu 中
    losses = losses.detach().cpu()
    all_losses = torch.cat((all_losses, losses))


save_to_file(query_file = args.query_file_path, candidate_file = args.candidate_file_path, template = args.template, \
    all_losses = all_losses, save_file = args.save_file, K = args.topK)

print('    DONE.')


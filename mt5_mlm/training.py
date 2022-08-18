#coding=utf-8

import arguments
args = arguments.parse_args()

import os
# os.chdir(os.path.dirname(__file__))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

input_ids, output_ids = load_tensor(f'/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/semeval/')

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
train_dataset = TensorDataset(input_ids, output_ids)
train_dataloader = DataLoader(
            train_dataset,  # 训练样本
            num_workers = 0,
            sampler = RandomSampler(train_dataset), # 随机小批量
            batch_size = args.batch_size # 以小批量进行训练
        )

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # My envirnment uses CPU
T5_PATH = "/home/hadoop-aipnlp/cephfs/data/donghande/code_kg/MLM_torch/models/google/mt5-base" # "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"
t5_config = T5Config.from_pretrained(T5_PATH)
model = T5ForConditionalGeneration.from_pretrained(T5_PATH, config=t5_config).to(DEVICE)

from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = 1e-4, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
from transformers import get_linear_schedule_with_warmup
total_steps = args.epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)

import time
import datetime
import tqdm

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # 四舍五入到最近的秒
    elapsed_rounded = int(round((elapsed)))
    
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

training_stats = []
total_t0 = time.time()

for epoch_i in range(0, args.epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        # 每经过40次迭代，就输出进度信息
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:},    Current loss: {:}.'.format(step, len(train_dataloader), elapsed, loss.item()))


        b_input_ids = batch[0].to(DEVICE)
        b_labels = batch[1].to(DEVICE)
        model.zero_grad()       

        loss = model(input_ids=b_input_ids, labels=b_labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

        if (step + 1) % 10000 == 0: model.save_pretrained(f'/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/semeval/model/mt5-{step}')


    avg_train_loss = total_train_loss / len(train_dataloader)            
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
    if epoch_i%1==0:
        model.save_pretrained(f"/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/semeval/model/mt5-{epoch_i}")

    # print("")
    # print("Running Validation...")

    # t0 = time.time()

    # # 设置模型为评估模式
    # model.eval()

    # # Tracking variables 
    # total_eval_accuracy = 0
    # total_eval_loss = 0
    # nb_eval_steps = 0

    # # Evaluate data for one epoch
    # for batch in validation_dataloader:
        
    #     # 将输入数据加载到 gpu 中
    #     b_input_ids = batch[0].to(device)
    #     b_input_mask = batch[1].to(device)
    #     b_labels = batch[2].to(device)
        
    #     # 评估的时候不需要更新参数、计算梯度
    #     with torch.no_grad():        
    #         (loss, logits) = model(b_input_ids, 
    #                                token_type_ids=None, 
    #                                attention_mask=b_input_mask,
    #                                labels=b_labels)
            
    #     # 累加 loss
    #     total_eval_loss += loss.item()

    #     # 将预测结果和 labels 加载到 cpu 中计算
    #     logits = logits.detach().cpu().numpy()
    #     label_ids = b_labels.to('cpu').numpy()

    #     # 计算准确率
    #     total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # # 打印本次 epoch 的准确率
    # avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    # print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # # 统计本次 epoch 的 loss
    # avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # # 统计本次评估的时长
    # validation_time = format_time(time.time() - t0)
    
    # print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    # print("  Validation took: {:}".format(validation_time))

    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            # 'Valid. Loss': avg_val_loss,
            # 'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            # 'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

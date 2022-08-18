JR='./jr_train/test/'
import random
e1_type=[]
e1=[]
e2_type=[]
e2=[]
relation=[]
content=[]
with open(JR+'ori.txt','r',encoding='utf-8') as r_content:
   all=r_content.readlines()
   for i in all:
        x=i.split('\t')
        e1_type.append(x[0]),
        e1.append(x[1])
        e2_type.append(x[2]),
        e2.append(x[3])
        relation.append(x[4])
        content.append(x[6].strip('\n'))
print("数据加载完毕！！！")
with open('jr_train/test/mask.txt','w',encoding='utf-8') as w:
    for i in range(1,len(content)):
        w.write(e1[i]+'('+e1_type[i]+')'+'与'+e2[i]+'('+e2_type[i]+')'+'之间的关系为'+' <extra_id_0>'+'[SEP]'+content[i])
        w.write('\x01<extra_id_0>'+relation[i]+' <extra_id_1>\n')
# with open('/home/hadoop-aipnlp/cephfs/data/donghande/code_kg/MLM_torch/datasets/ugc_v2/raw/ugc','r',encoding='utf-8') as r:
#     print(r.readlines()[:100])

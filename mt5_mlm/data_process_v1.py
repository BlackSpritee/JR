JR='/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/jr/news_contents_train.txt'
import random
with open(JR,'r',encoding='utf-8') as r:
   lines=r.readlines()
   print(len(lines))
print("数据加载完毕！！！")
rel_w_list = [['持股','持有', '减持', '增持', '拥有', '减少', '流入', '流出', '拟减', '缩减', ],
              ['股份股权转让','转让','出售', '受让',  '发行', '股份'],
              ['任职','新任', '现任','担任','法人''创始人','董事长','CEO','总经理','首席创新官','法定代表人','董事会秘书','高级合伙人'],
              ['惩罚','开罚单','处罚','拘留','整改','罚款','终止上市','退市'],
              ['业务','业务为','业务覆盖','办理业务'],
              ['合作', '签署', '协议', '订立', '意向书', '携手', '联合', '一致', '行动人','合伙人'],
              ['子公司', '旗下','附属公司'],
              ['隶属','领域', '行业'],
              ['投资', '增资', '募资', '购买', '出资', '买入', '认购', '收购', '并购','注资'],
              ['债务','质押', '债务', '贷款'],
              ['起诉','诉讼','指控']
              ]

relation_word=[]
for i in rel_w_list:
    for j in i:
        relation_word.append(j)

with open('jr/mask_train.txt','w',encoding='utf-8') as w:
    for i,line in enumerate(lines):
            if i%100==0:
                print(i)
            temp=line
            for word in relation_word:
                for index in range(len(line)):
                    if word==line[index:index+len(word)]:
                        temp=temp[0:index]+' <extra_id_0>'+temp[index+len(word):]
                        w.write(temp.strip()+('\x01<extra_id_0>'+word+' <extra_id_1>\n'))
                        temp=line

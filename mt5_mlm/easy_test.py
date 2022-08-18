# rel_w_list = [['持股','持有', '减持', '增持', '拥有', '减少', '流入', '流出', '拟减', '缩减', ],
#               ['股份股权转让','转让','出售', '受让',  '发行', '股份'],
#               ['任职','新任', '现任','担任'],
#               ['惩罚','开罚单','处罚','拘留','整改','罚款','终止上市','退市'],
#               ['业务','业务为','业务覆盖','办理业务'],
#               ['合作', '签署', '协议', '订立', '意向书', '携手', '联合', '一致', '行动人','合伙人'],
#               ['子公司', '旗下','附属公司'],
#               ['隶属','领域', '行业'],
#               ['投资', '增资', '募资', '购买', '出资', '买入', '认购', '收购', '并购','注资'],
#               ['债务','质押', '债务', '贷款'],
#               ['起诉','诉讼','指控']
#               ]
# num=0
# with open('jr/relation_word','a+',encoding='utf-8') as w:
#     for i in rel_w_list:
#         for j in  i:
#             num+=1
# print(num)
#
# with open('/home/hadoop-aipnlp/cephfs/data/donghande/code_kg/MLM_torch/datasets/ugc_v2/raw/ugc','r') as r:
#     print(r.readlines()[:100])
#
# import  json
# with open('datasets/10.json','r',encoding='utf-8') as r:
#     j=r.readlines()
#     j=j[0]
#     js=json.loads(j)
#     print(js[1]["tags"])

# with open('/home/hadoop-aipnlp/cephfs/data/donghande/code_kg/MLM_torch/datasets/ugc_v2/raw/ugc','r') as r:
#     a=r.readline()
# print(a)
#
# rel_w_list = [['持股','持有', '减持', '增持', '拥有', '减少', '流入', '流出', '拟减', '缩减', ],
#               ['股份股权转让','转让','出售', '受让',  '发行', '股份'],
#               ['任职','新任', '现任','担任'],
#               ['惩罚','开罚单','处罚','拘留','整改','罚款','终止上市','退市'],
#               ['业务','业务为','业务覆盖','办理业务'],
#               ['合作', '签署', '协议', '订立', '意向书', '携手', '联合', '一致', '行动人','合伙人'],
#               ['子公司', '旗下','附属公司'],
#               ['隶属','领域', '行业'],
#               ['投资', '增资', '募资', '购买', '出资', '买入', '认购', '收购', '并购','注资'],
#               ['债务','质押', '债务', '贷款'],
#               ['起诉','诉讼','指控']
#               ]
# relation_word=[]
# for i in rel_w_list:
#     for j in i:
#         relation_word.append(j)
# print(relation_word)
with open('/home/hadoop-aipnlp/cephfs/data/baichuanyang/project/code_kg/code_kg/MLM_torch/jr/mask_train.txt','r',encoding='utf-8') as r:
    r=r.readlines()[:1000]
print(r)

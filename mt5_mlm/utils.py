import pandas as pd
import torch
from transformers import T5Tokenizer 
from torch.utils.data import IterableDataset, DataLoader

class ProductFromFile(IterableDataset):                                          
                                                                                
    def __init__(self, file_path_input: str, file_path_output: str, template: str, 
        MAX_LEN_in: int, MAX_LEN_out: int, Tokenizer_PATH = str):                                                                          
        super(ProductFromFile).__init__()                                        
        self.file_path_input = file_path_input
        self.file_path_output = file_path_output
        self.template = template

        self.info_input = self._get_file_info(file_path_input)
        self.info_output = self._get_file_info(file_path_output)

        self.MAX_LEN_in = MAX_LEN_in
        self.MAX_LEN_out = MAX_LEN_out
        self.tokenizer = T5Tokenizer.from_pretrained(Tokenizer_PATH)
        # self.tokenizer.add_tokens(['持股', '持有', '减持', '增持', '拥有', '减少', '流入', '流出', '拟减', '缩减', '股份股权转让', '转让', '出售', '受让', '发行', '股份', '任职', '新任', '现任', '担任', '惩罚', '开罚单', '处罚', '拘留', '整改', '罚款', '终止上市', '退市', '业务', '业务为', '业务覆盖', '办理业务', '合作', '签署', '协议', '订立', '意向书', '携手', '联合', '一致', '行动人', '合伙人', '子公司', '旗下', '附属公司', '隶属', '领域', '行业', '投资', '增资', '募资', '购买', '出资', '买入', '认购', '收购', '并购', '注资', '债务', '质押', '债务', '贷款', '起诉', '诉讼', '指控'])
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()                        
        if worker_info is None:  # single worker     
            pass                                           
        else:  # multiple workers                                               
            raise NotImplementedError               
        sample_iterator = self._sample_generator()          
        return sample_iterator  
                                                                                
    def __len__(self):                                                          
        return (self.info_input['end'] - self.info_input['start']) * \
            (self.info_output['end'] - self.info_output['start'])
                                                                                
    def _get_file_info(self, file_path):                                                                          
        info = {  
            "path": file_path,                                                              
            "start": 1,                                                         
            "end": 0,                                                       
            "content": 0 # column in file
        }                                                                       
        with open(file_path, 'r') as fin:                                       
            for _ in enumerate(fin):                                            
                info['end'] += 1                                                                                            
        return info                                                   
                                                                                
    def _sample_generator(self):     
        input_c = self.info_input['content']
        output_c = self.info_output['content']

        with open(self.info_input['path'], 'r') as fin_in:
            for i, line_in in enumerate(fin_in):                                      
                if i < self.info_input['start']: continue
                items_in = line_in.strip().split('')[input_c]
                tensor_in = self.tokenizer([items_in + ' <extra_id_0>'+self.template], padding="max_length",
                            max_length=self.MAX_LEN_in, 
                            truncation=True, return_tensors='pt').input_ids[0]
                with open(self.info_output['path'], 'r') as fin_out:
                    for j, line_out in enumerate(fin_out): 
                        if j < self.info_output['start']: continue
                        if i >= self.info_input['end'] and j >= self.info_output['end']: return StopIteration()
                        items_out = line_out.strip().split('')[output_c]
                        tensor_out = self.tokenizer(['<extra_id_0>' + items_out + ' <extra_id_1>'], padding="max_length",
                            max_length=self.MAX_LEN_out, 
                            truncation=True, return_tensors='pt').input_ids[0]
                        sample = {0: tensor_in, 1: tensor_out, 2: items_in, 3: items_out, 4: self.template}
                        yield sample


def load_tensor(dir_name): 
    return torch.load(dir_name + 'semeval_Tensor_all.pth')

def save_to_file(query_file: str, candidate_file: str, template: str, all_losses: torch.Tensor, save_file: str, K: int): 
    query_list = []
    candidate_list = []
    template_list = []
    values_list = []

    query = pd.read_table(query_file).iloc[:, 0].tolist()
    candidate = pd.read_table(candidate_file).iloc[:, 0].tolist()

    all_loss = all_losses.view(-1, len(candidate))
    for i, loss in enumerate(all_loss): 
        values, indices = torch.topk(loss, k=K, largest=False)
        values = list(values.cpu().numpy())
        indices = list(indices.cpu().numpy())

        query_list.extend([query[i]] * K)
        candidate_list.extend(candidate[j] for j in indices)
        template_list.extend([template] * K)
        values_list.extend(values)

    res_df = pd.DataFrame({'query': query_list, 'candidate': candidate_list, 'template': template_list, 'loss': values_list})
    res_df.to_csv(save_file, index=False, mode = 'a', header = False)
    
if __name__ == '__main__': 
    load_tensor()
import argparse
# arguments setting
def parse_args(): 
    parser = argparse.ArgumentParser(description='MT5 for MLM in commodity knowledge graph. ')
    parser.add_argument('--dataset', type=str, default='v1', help='Choose from v1, v2')
    parser.add_argument('--batch_size', type=int, default =16,
                help='Batch size.')
    parser.add_argument('--epochs', type=int, default = 40,
                help='Epochs.')
    parser.add_argument('--gpu', type=int, default = 3,
                help='Batch size.')            
    parser.add_argument('--dim', type=int, default = 10, 
                help='The dimension of output.')
    parser.add_argument('--model_path', type=str, default = 'models/meituan_v1/mt5-base-epoch-0', #v2 epoch-3
                help='The path to save model.') 
    parser.add_argument('--query_file_path', type=str, default = 'jr/entity1',
                help='The file of query words.')
    parser.add_argument('--candidate_file_path', type=str, default = 'jr/relation',
                help='The file of candidate words.')  
    parser.add_argument('--template', type=str, default = '太平洋证券',
                help='The file of query word.')  
    parser.add_argument('--MAX_LEN_in', type=int, default = 128,
                help='The max length of input sentence.')     
    parser.add_argument('--MAX_LEN_out', type=int, default = 16,
                help='The max length of output sentence.')        
    parser.add_argument('--topK', type=int, default = 65,
                help='Return the top K min loss.')     
    parser.add_argument('--save_file', type=str, default = 'jr_train/train/res2.csv',
                help='The file to save result.')     
    parser.add_argument('--seed', type=int, default=0, help='global general random seed.')
    return parser.parse_args()
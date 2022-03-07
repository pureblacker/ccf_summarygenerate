from torch.utils.data import Dataset
import json,os,torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer
class MyDataset(Dataset):
    def __init__(self,data_path,src_maxlen,tar_maxlen):
        self.tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
        self.src_maxlen=src_maxlen
        self.tar_maxlen=tar_maxlen
        self.src_id=self.tokenizer.convert_tokens_to_ids("[Content]")
        self.tar_id=self.tokenizer.convert_tokens_to_ids("[Summary]")
        self.data_set=self.load_data(data_path)

    def load_data(self,data_path):
        filenames=os.listdir(data_path)
        self.data_set=[]
        for i in filenames:
            path=os.path.join(data_path,i)
            with open(path) as f:
                data=json.load(f)
                input_ids,token_type_ids=self.convert_to_ids(data)
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set
    
    def convert_to_ids(self, sample):  #sample: {'source':'....','summary':'.....'}
        input_ids=[]
        token_type_ids=[]
        src_tokens=self.tokenizer.tokenize(sample['source'])
        tar_tokens=self.tokenizer.tokenize(sample['summary'])
        if len(tar_tokens)>self.tar_maxlen:
            tar_tokens=tar_tokens[:self.tar_maxlen]
        if len(src_tokens)>self.src_maxlen:
            src_tokens=src_tokens[:self.src_maxlen]
        
        input_ids.append(self.tokenizer.cls_token_id)
        token_type_ids.append(self.src_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(src_tokens))
        token_type_ids.extend([self.src_id]*len(src_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.src_id)
        input_ids.extend(self.tokenizer.convert_tokens_to_ids(tar_tokens))
        token_type_ids.extend([self.tar_id] * len(tar_tokens))
        input_ids.append(self.tokenizer.sep_token_id)
        token_type_ids.append(self.tar_id)
        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.src_maxlen+self.tar_maxlen+3
        return input_ids, token_type_ids  #返回的是一个样本的东西

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        instance = self.data_set[idx]
        return instance
    
def collate_fn(batch_data):
    batch_size=len(batch_data)
    if batch_size==0:
        return {}
    input_ids_list, token_type_ids_list=[],[]
    for instance in batch_data:
        input_ids_temp=instance["input_ids"]
        token_type_ids_temp=instance["token_type_ids"]
        input_ids_list.append(torch.tensor(input_ids_temp,dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp,dtype=torch.long))

    return {
            "input_ids":pad_sequence(input_ids_list,batch_first=True,padding_value=0),
            "token_type_ids":pad_sequence(token_type_ids_list,batch_first=True,padding_value=0)
        }

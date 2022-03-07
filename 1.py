from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader
from data_set import MyDataset,collate_fn
train_data=MyDataset('data_set/dev',980,40)
train_dl = DataLoader(train_data, batch_size=2, shuffle=True,collate_fn=collate_fn)
    # 模型
# model=GPT2LMHeadModel.from_pretrained('gpt2')
# print(type(train_dl))
for idx,batch in enumerate(train_dl):
    print(batch,type(batch))

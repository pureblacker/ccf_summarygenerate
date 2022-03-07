from transformers import AdamW, GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader
from data_set import MyDataset, collate_fn
from set_args import set_args

def train( data_loader, model,batch_size=16, epochs=5, 
            lr=2e-5, warmup_steps=200):
    
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)
    
    sum_loss=0
    for epoch in range(epochs):
        print(f"running {epoch+1} epochs")
        for idx,txt in enumerate(data_loader):
            pass

def main(args):
    # 数据：
    args=set_args()
    train_data=MyDataset('dataset/train',args.src_maxlen,args.tar_maxlen)
    train_dl = DataLoader(train_data, batch_size=1, shuffle=True,collate_fn=collate_fn)
    # 模型
    model=GPT2LMHeadModel.from_pretrained('gpt2')
    #训练模型
    train(train_dl,model)
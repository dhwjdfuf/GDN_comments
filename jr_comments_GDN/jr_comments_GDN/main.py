import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from GDN import GDN
from train_test import *
from evaluate import *

class TimeDataset(Dataset): #train_df or test_df. dataset 하나의 모든것을 담는 클래스
    '''
        x torch.Size([310, 27, 15])
        y torch.Size([310, 27])
        attack torch.Size([310]). 
        
        data 하나 predict: (27,15) -> (27)
        그래서
        data batch에는 310개의 data가 있는거다. 그래서 (310,27,15). 
        
        아마 한 sensor라도 loss가 크면 그 timestamp가 attack=1 일거야.
    '''

    def __init__(self, data_df, mode, config):
        self.data_df=data_df
        self.config=config
        self.mode=mode

        self.feature, self.label, self.attack = self.process()
        #310 X 27 X 15   310 X 27
        

    
    def process(self):
        win=self.config['slide_win']
        stride=self.config['slide_stride']

      
        if self.mode=='test':
            attack_col=torch.tensor(self.data_df['attack'])
            self.data_df=self.data_df.drop(columns=['attack'])


        num_nodes=len(self.data_df.columns)
        timestamp_len=len(self.data_df.iloc[:,1])


        ran=range(win,timestamp_len,stride) if self.mode =='train' else range(win,timestamp_len)
        data_num=len(ran)
        feature=torch.zeros((data_num,num_nodes,win)) 
        label=torch.zeros((data_num,num_nodes))
        attack=torch.zeros((data_num))

        for cnt,i in enumerate(ran): #310, i는 window를 뽑는 시작점.
            mat_i=torch.zeros((num_nodes,win))
            label_i=torch.zeros((num_nodes))
            for j in range(num_nodes):
                column=torch.tensor(self.data_df.iloc[:,j])
                mat_i[j]=column[i-win:i] 
                label_i[j]=column[i]

                if j==0 and self.mode=='test':
                    attack[cnt]=attack_col[i]
                    

            feature[cnt]=mat_i
            label[cnt]=label_i


        return feature, label, attack

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx): #feature의 dim0만 리턴해주면 돼. 

        return self.feature[idx], self.label[idx], self.attack[idx]

def train_val_loader(train_dataset, batch, val_ratio=0.1): #이건 한 train set에서 val도 뽑아내는 거야. 
        dataset_len = int(len(train_dataset)) 

        train_use_len = int(dataset_len * (1 - val_ratio)) #train 개수
        val_use_len = int(dataset_len * val_ratio) #val 개수 

        val_start_index = random.randrange(train_use_len) #이게 upper bound니까.
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]]) #val 뺀 나머지 indice가 train indice~ 
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len] #val indice.
        val_subset = Subset(train_dataset, val_sub_indices) # train_dataset을 train과 val로 나누는거네. index로 구분하고. 
        #Subset은, dim0을 indices로 잘라서 붙여서 새 tensor을 만들었다고 생각해.
        


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True)
                            #DataLoader()이 class만 받아야 하나? 그런건 아니야. 그냥 train_subset[] 만 가능하면 돼. 그게 핵심이야.
                            #안거같다. DataLoader()은 batch리턴하는게 맞고,
                            # 100 X 2 X 2 dataset이면, batch는 첫째 dimension. 10 X 2 X 2로 10개를 준다. 
                            # 
                            # batch size == batch의 depth(channel)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False)

        return train_dataloader, val_dataloader

        


train_df=pd.read_csv('./data/train.csv',index_col='timestamp')
test_df=pd.read_csv('./data/test.csv',index_col='timestamp')
list_txt=open('./data/list.txt','r')

config={
    'slide_win': 15,
    'slide_stride': 5,
    'batch': 128,
    'dim': 64,
    'val_ratio': 0.1,
    'topk': 20, #including itself. 
    'out_layer_num': 1,
    'out_layer_inter_dim': 256,
    'decay': 0,
    'epoch': 200,
    'report': 'best' # or 'val'

}

nodes_list=[] #총 27개.
for node in list_txt:
    nodes_list.append(node.strip())

full_edges=[]

for i in range(len(nodes_list)):
    for j in range(len(nodes_list)):
        if i==j:
            continue
        full_edges.append([i,j])

full_edges=torch.tensor(full_edges).T 

train_dataset=TimeDataset(train_df,'train',config)
test_dataset=TimeDataset(test_df,'test',config)




train_dataloader, val_dataloader = train_val_loader(train_dataset, config['batch'], config['val_ratio'])
test_dataloader = DataLoader(test_dataset, batch_size=config['batch'], shuffle=False, num_workers=0)

model = GDN(full_edges, len(nodes_list), 
                embed_dim=config['dim'], 
                input_dim=config['slide_win'],
                out_layer_num=config['out_layer_num'],
                out_layer_inter_dim=config['out_layer_inter_dim'],
                topk=config['topk'],
            )




train_log=train(model, config,  train_dataloader, val_dataloader, nodes_list, test_dataloader, test_dataset, train_dataset, full_edges)


best_model=model



_, test_result= test(best_model,test_dataloader,full_edges)
_, val_result = test(best_model,val_dataloader, full_edges)

get_score(test_result,val_result,config['report'])






    
'''
뭐 여튼 돌리는 데에는 완벽하게 성공했다. 이제 뭘 할 지 제대로 적어보자. 

gpu? 코드랑은 관련 없고.

'''

        










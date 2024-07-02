# TODO
# gbdt / xgboost
# Transformer Encoder + DNN
# token + DNN
import torch
from torch import nn as nn



class Model(nn.Module):
    def __init__(self, jd_embedding, cv_embedding, embedding_dim = 1536, hidden_dim = 512):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.jd_embedding = nn.Embedding(len(jd_embedding), embedding_dim)
        jd_matrix = torch.tensor([list(jd_embedding.values())]).squeeze(0)
        self.jd_embedding.weight.data.copy_(jd_matrix)       
        self.jd_embedding.weight.requires_grad = False

        self.cv_embedding = nn.Embedding(len(cv_embedding), embedding_dim)
        cv_matrix = torch.tensor([list(cv_embedding.values())]).squeeze(0)
        self.cv_embedding.weight.data.copy_(cv_matrix)       
        self.cv_embedding.weight.requires_grad = False


        self.combine = nn.Linear(2 * embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear_2nd = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU()
    
    def forward(self, jd, cv):
        jd_embeds = self.jd_embedding((torch.tensor(jd, device=self.device, dtype=torch.int)))
        cv_embeds = self.cv_embedding((torch.tensor(cv, device=self.device, dtype=torch.int)))
        x = torch.cat([jd_embeds, cv_embeds], dim=0)
        x = self.combine(x)
        x = self.leaky_relu(x)
        x = self.linear(x)
        x = self.leaky_relu(x)
        x = self.linear_2nd(x)
        return x 


# loss_fn = nn.BCEWithLogitsLoss() 
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Model(jd_embedding, cv_embedding).to(device)
# predict = model(1, 2)

# print(predict)
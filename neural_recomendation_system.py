import torch
import torch.nn as nn 
import torch.optim as optim
import numpy as np
from torch.utils.data import dataset, dataloader

class NCFDataset(dataset):
    def __init__(self,user_ids,item_ids,ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self,idx):
        return(
            self.user_ids[idx],
            self.item_ids[idx],
            self.ratings[idx]
        )
        
class UserBasedNCF(nn.modules):
    def __init__(self,num_users,num_items,embeddings_dim=50):
        super(UserBasedNCF,self).__init__() 
        self.user_embeddings = nn.Embedding(num_users,embeddings_dim)
        self.item_embedding =  nn.Embedding(num_items,embeddings_dim)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embeddings_dim*2,embeddings_dim),
            nn.ReLU(),
            nn.Linear(embeddings_dim,embeddings_dim//2),
            nn.ReLU(),
            nn.Linear(embeddings_dim//2,1),
        )
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight,std=0.01)
        nn.init.normal_(self.item_embedding.weight,std=0.01)
    
    
    def forward(self,users,items):
        user_embed = self.user_embedding(users)
        item_embed = self.item_embedding(items)
        combined =torch.cat([user_embed,item_embed],dim=1)
        prediction =torch.sigmoid(self.mlp_layers(combined)).squeeze()
        return prediction
    
class ItemBasedNCF(nn.module):
    def __init__(self,num_users,num_items,embeddings_dim=50):
        super(ItemBasedNCF,self).__init__()
        self.user_embedding = nn.Embedding(num_users,embeddings_dim)
        self.item_embedding = nn.Embedding(num_items,embeddings_dim)
        
        self.gmf_interaction = nn.Linear(embeddings_dim,1)
        self.mlp_layers = nn.Sequential(
            nn.Linear(embeddings_dim*2,embeddings_dim),
            nn.ReLU(),
            nn.Linear(embeddings_dim,embeddings_dim//2),
            nn.ReLU(),
            nn.Linear(embeddings_dim//2,1)
        )
        
        self.fusion_layer = nn.Linear(2,1)
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.user_embeddings.weight,std=0.01)
        nn.init.normal_(self.item_embeddings.weight,std=0.01)
        
    def forward(self,users,items):
        user_embed = self.user_embedding(users)
        item_embed = self.item_embedding(items)
        gmf_output = user_embed * item_embed
        gmf_prediction = torch.sigmoid(self.gmf_interaction(gmf_output))
        combined = torch.cat([user_embed,item_embed],dim=1)
        mlp_prediction = torch.sigmoid(self.mlp_layers(combined))
        
        fused_prediction = torch.sigmoid(self.fusion_layer(torch.cat([gmf_prediction,mlp_prediction],dim=1))).squeeze()
        
        return fused_prediction
    
def train_ncf_model(model,train_loader,criterion,optimizer,epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for users, items , ratings in train_loader:
            optimizer.zero_grad()
            predictions = model(users,items)
            loss = criterion(predictions,ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0:
            print(f"Epoch [(epoch+1/ {epochs})],Loss:{total_loss/len(train_loader):.4f}")
    
def prepare_data():
    num_users = 1000
    num_items = 500
    
    user_ids = np.random.randint(0,num_users,5000)
    item_ids = np.random.ranint(0,num_items,5000)
    ratings = np.random.random(5000)
    
    dataset = NCFDataset(user_ids,item_ids,ratings)
    train_loader = dataloader(dataset,batch_size=64, shuffle=True)
    
    return num_users,num_items,train_loader                
        
        
def main():
    num_users,num_items,train_loader = prepare_data()
    print("Training user-based neural collaborative filtering")
    
    
    user_ncf_model = UserBasedNCF(num_users,num_items)
    user_ncf_optimizer = optim.Adam(user_ncf_model.parameters(),lr=0.001)
    user_ncf_criterion = nn.BCELoss()
    
    train_ncf_model(
        user_ncf_model,
        train_loader,
        user_ncf_criterion,
        user_ncf_optimizer
    )
    
    print(f"\nTraining Item-Based Neural Collaborative Filtering")
    item_ncf_model = ItemBasedNCF(num_users,num_items)
    item_ncf_optimizer = optim.Adam(item_ncf_model.parameters(),lr=0.001)
    item_ncf_criterion = nn.BCELoss()
    
    
        
        
        
        
        
        
        
        
        
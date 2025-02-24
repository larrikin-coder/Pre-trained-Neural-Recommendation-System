# import torch
# from tncf import *



# torch.save(model.state_dict(),'tnr_model.pth')
# model.load_state_dict(torch.load('tnr_model.pth'))

# model.eval()

# def recommend(user_id,item_id,domain_id,top_k=5):
#     user_tensor = torch.tensor([user_id])
#     item_tensor = torch.tensor(item_id)
#     domain_tensor = torch.tensor([domain_id]*len(item_id))
    
#     with torch.no_grad():
#         scores = model(user_tensor,item_tensor,domain_tensor)
        
#     top_indices = torch.argsort(scores,descending=True)[:top_k]
#     recommended_items = item_tensor[top_indices].tolist()
#     return recommended_items 
    
    
# user_id = 2
# item_candidates = [101,102,103,104,105]
# domain_id = 1

# recommendations = recommend(user_id,item_candidates,domain_id)
# print("Recommended Items:",recommendations)

import torch 
from tncf import TransferableRecommender
from torch.utils.data import Dataset, DataLoader
import numpy as np

NUM_USERS = 100
NUM_ITEMS = 50
NUM_DOMAINS = 2
EMBEDDING_DIM = 64
HIDDEN_LAYERS = [128,64,32]

model = TransferableRecommender(
    num_users=NUM_USERS,
    num_items=NUM_ITEMS,
    num_domains=NUM_DOMAINS,
    embedding_dim=EMBEDDING_DIM,
)


class SimpleDataset(Dataset):
    def __init__(self,num_samples):
        self.users = torch.randint(0,NUM_USERS,(num_samples,))   
        self.items = torch.randint(0,NUM_ITEMS,(num_samples,))
        self.domains = torch.randint(0,NUM_DOMAINS,(num_samples,)) 
        self.ratings = torch.randint(0,5,(num_samples,)).float()

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self,idx):
        return self.users[idx],self.items[idx],self.domains[idx],self.ratings[idx]

train_dataset = SimpleDataset(num_samples=1000)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

# def train(epochs=5):
#     for epoch in range(epochs):
#         total_loss = 0
#         for batch_idx,(users,items,domains,ratings) in enumerate(train_loader):
#             optimizer.zero_grad()
#             outputs = model(users,items,domains)
#             loss = criterion(outputs,ratings)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         avg_loss = total_loss/len(train_loader)
#         print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

def save_model(path='tnr_model.pth'):
    torch.save(model.state_dict(),path)
    print(f"Model saved to {path}")

def load_model(path='tnr_model.pth'):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

def recommend(user_id,item_ids,domain_id,top_k=5):
    if user_id >= NUM_USERS:
        raise ValueError(f"User ID {user_id} is out of range. Must be between 0 and {NUM_USERS-1}")
    if domain_id >= NUM_DOMAINS:
        raise ValueError(f"Domain ID {domain_id} is out of range. Must be between 0 and {NUM_DOMAINS-1}")
    if max(item_ids) >= NUM_ITEMS:
        raise ValueError(f"Item ID {max(item_ids)} is out of range. Must be between 0 and {NUM_ITEMS-1}")
    
    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).repeat(len(item_ids))
        item_tensor = torch.tensor(item_ids)
        domain_tensor = torch.tensor([domain_id]*len(item_ids))
        scores = model(user_tensor,item_tensor,domain_tensor)
        top_indices = torch.argsort(scores,descending=True)[:top_k]
        recommended_items = item_tensor[top_indices].tolist()
    return recommended_items

if __name__ == "__main__":
    print("training model...")
    # train()
    print("model trained")
    print("saving model...")
    save_model()
    print("model saved")

    load_model()
    user_id = 1
    item_candidates = [0,1,2,3,4,5,6,7,8,9]
    domain_id = 0
    top_k = 3
    recommendations = recommend(user_id,item_candidates,domain_id,top_k)
    print(f"Recommended items for user {user_id} in domain {domain_id}: {recommendations}")

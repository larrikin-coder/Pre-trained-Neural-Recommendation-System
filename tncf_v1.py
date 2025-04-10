import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict


class Dataset(Dataset):
    def __init__(self, data):
        self.names = data['name'].tolist()
        self.ratings = data['rating'].tolist()
        self.domains = data['domain'].tolist()
        self.unique_names = sorted(set(self.names))
        self.unique_domains = sorted(set(self.domains))
        self.name_to_index = {name:idx for idx ,name in enumerate(self.unique_names)}
        self.domain_to_index = {domain:idx for idx, domain in enumerate(self.unique_domains)}
        

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        name = self.names[idx]
        rating = self.ratings[idx]
        domain = self.domains[idx]
        name_index = self.name_to_index[name]
        domain_index = self.domain_to_index[domain]
        return torch.tensor(name_index, dtype=torch.long), \
               torch.tensor(rating, dtype=torch.float), \
               torch.tensor(domain_index, dtype=torch.long)


class NCFModel(nn.Module):
    def __init__(self,num_names,embedding_dim,num_domains):
        super(NCFModel,self).__init__()
        self.name_embedding = nn.Embedding(num_names,embedding_dim)
        self.domain_embedding = nn.Embedding(num_domains,embedding_dim)
        self.fc1 = nn.Linear(embedding_dim*2,64)
        self.fc2 = nn.Linear(64,1)
        self.dropout = nn.Dropout(0.2)       
        
    def forward(self,name_indices,domain_indices):
        name_embeds = self.name_embedding(name_indices)
        domain_embeds = self.domain_embedding(domain_indices)
        combined_embeds = torch.cat([name_embeds,domain_embeds],dim=1)
        x = F.relu(self.dropout(self.fc1(combined_embeds)))
        rating = torch.sigmoid(self.fc2(x)) * 5 
        return rating.squeeze(1)
    
def create_domain_mappings(data):
    unique_domains = sorted(set(data['domain']))
    domain_to_index = {domain: idx for idx, domain in enumerate(unique_domains)}
    return  domain_to_index, len(unique_domains)


def prepare_data(data,domain_to_index):
    data['domain'] = data['domain'].map(domain_to_index)
    return data


def train(model,train_loader,optimizer,criterion,device):
    model.train()
    total_loss = 0
    for name_indices, ratings, domain_indices in train_loader:
        name_indices = name_indices.to(device)
        ratings = ratings.float().to(device)
        domain_indices = domain_indices.to(device)
        optimizer.zero_grad()
        predictions = model(name_indices, domain_indices)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)



def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for name_indices,ratings,domain_indices in val_loader:
            name_indices = name_indices.to(device)
            ratings = ratings.float().to(device)
            domain_indices = domain_indices.to(device)
            predictions = model(name_indices, domain_indices)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()
        return total_loss / len(val_loader)

def get_predictions(model,data,domain_to_index,device):
    model.eval()
    names = data['name'].tolist()
    domains = data['domain'].tolist()
    unique_names = sorted(set(names))
    name_to_index_local = {name: idx for idx, name in enumerate(unique_names)}
    name_indices_tensor = torch.tensor([name_to_index_local[name] for name in names], dtype=torch.long).to(device)
    domain_indices_tensor = torch.tensor([domain_to_index[domain] for domain in domains], dtype=torch.long).to(device)
    with torch.no_grad():
        predictions = model(name_indices_tensor, domain_indices_tensor)
    return list(zip(names, predictions.cpu().numpy()))

def recommend_based_on_item(model, liked_item_name, liked_item_domain, dataset_obj, top_n=10, device='cpu'):
    model.eval()
    recommendations = []
    if liked_item_name not in dataset_obj.name_to_index or liked_item_domain not in dataset_obj.domain_to_index:
        print(f"Item '{liked_item_name}' or domain '{liked_item_domain}' not found.")
        return recommendations

    liked_name_index = dataset_obj.name_to_index[liked_item_name]
    liked_domain_index = dataset_obj.domain_to_index[liked_item_domain]

    # Get the embedding of the liked item and its domain
    with torch.no_grad():
        liked_name_embed = model.name_embedding(torch.tensor([liked_name_index]).to(device))
        liked_domain_embed = model.domain_embedding(torch.tensor([liked_domain_index]).to(device))
        liked_combined_embed = torch.cat([liked_name_embed, liked_domain_embed], dim=1)
        # Get the intermediate representation (after fc1) - might capture higher-level features
        liked_features = F.relu(model.fc1(liked_combined_embed)).cpu().numpy()

    # Compare the liked item's representation with all other items
    all_item_representations = {}
    for name in dataset_obj.unique_names:
        name_index = dataset_obj.name_to_index[name]
        for domain in dataset_obj.unique_domains:
            domain_index = dataset_obj.domain_to_index[domain]
            with torch.no_grad():
                name_embed = model.name_embedding(torch.tensor([name_index]).to(device))
                domain_embed = model.domain_embedding(torch.tensor([domain_index]).to(device))
                combined_embed = torch.cat([name_embed, domain_embed], dim=1)
                features = F.relu(model.fc1(combined_embed)).cpu().numpy()
                all_item_representations[(name, domain)] = features

    from sklearn.metrics.pairwise import cosine_similarity

    similarity_scores = {}
    for (name, domain), features in all_item_representations.items():
        if name != liked_item_name or domain != liked_item_domain:
            similarity = cosine_similarity(liked_features, features)[0][0]
            similarity_scores[(name, domain)] = similarity

    # Sort by similarity and get top N
    sorted_recommendations = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]

    return sorted_recommendations

if __name__ == "__main__":
    my_data = pd.read_csv("multi-domain synthetic dataset.csv")
    full_dataset = Dataset(my_data)
    train_data, val_data = train_test_split(my_data, test_size=0.2, random_state=42)
    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model parameters
    embedding_dim = 16
    num_names = len(full_dataset.unique_names)
    num_domains = len(full_dataset.unique_domains)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model, optimizer, and loss function
    model = NCFModel(num_names, embedding_dim, num_domains).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 20
    print(f"Training on multi-domain data for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Example recommendation based on a liked movie
    liked_movie = "The Confusion"
    liked_domain = "books"
    top_recommendations = recommend_based_on_item(model, liked_movie, liked_domain, full_dataset, top_n=5, device=device)

    print(f"\nTop recommendations based on liking '{liked_movie}' ({liked_domain}):")
    for (recommended_name, recommended_domain), similarity in top_recommendations:
        print(f"- {recommended_name} ({recommended_domain}) - Similarity: {similarity:.4f}")
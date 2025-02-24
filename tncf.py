import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Example Dataset (Replace with real multi-domain dataset)
class RecommendationDataset(Dataset):
    def __init__(self, users, items, ratings, domains):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)
        self.domains = torch.tensor(domains, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx], self.domains[idx]

# Example Data
users = [0, 1, 2, 3, 0, 1, 2]
items = [10, 11, 12, 13, 20, 21, 22]
ratings = [4.5, 5.0, 3.0, 4.0, 5.0, 3.5, 2.0]
domains = [0, 0, 0, 0, 1, 1, 1]  # 0: E-commerce, 1: Movies

dataset = RecommendationDataset(users, items, ratings, domains)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model Definition
class TransferableRecommender(nn.Module):
    def __init__(self, num_users, num_items, num_domains, embedding_dim=32):
        super(TransferableRecommender, self).__init__()
        # Shared Embedding Layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Domain Embeddings
        self.domain_embedding = nn.Embedding(num_domains, embedding_dim)

        # Shared Layers
        self.shared_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Domain-Specific Heads
        self.domain_heads = nn.ModuleList([
            nn.Linear(64, 1) for _ in range(num_domains)
        ])

    def forward(self, user_id, item_id, domain_id):
        # Shared Embeddings
        user_embed = self.user_embedding(user_id)
        item_embed = self.item_embedding(item_id)

        # Domain Embedding
        domain_embed = self.domain_embedding(domain_id)

        # Concatenate and pass through shared layers
        x = torch.cat([user_embed, item_embed, domain_embed], dim=-1)
        shared_output = self.shared_layer(x)

        # Domain-specific output
        # output = self.domain_heads[int(domain_id.item())](shared_output)
        output = [self.domain_heads[int(d)](shared_output[i]) for i, d in enumerate(domain_id)]
        output = torch.stack(output)  # Convert list to tensor

        return output.squeeze()

# Hyperparameters
num_users = max(users) + 1
num_items = max(items) + 1
num_domains = max(domains) + 1
embedding_dim = 32

# Instantiate Model
model = TransferableRecommender(num_users, num_items, num_domains, embedding_dim)

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop: Pre-training
print("Starting Pre-training...")
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for user_id, item_id, rating, domain_id in dataloader:
        optimizer.zero_grad()
        predictions = model(user_id, item_id, domain_id)
        loss = criterion(predictions, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save Pre-trained Model
torch.save(model.state_dict(), "pretrained_tnr.pth")

# Fine-tuning on a New Domain
print("\nStarting Fine-tuning on Domain 1...")
domain_1_data = [x for x in zip(users, items, ratings, domains) if x[3] == 1]
fine_tune_dataset = RecommendationDataset(
    [x[0] for x in domain_1_data],
    [x[1] for x in domain_1_data],
    [x[2] for x in domain_1_data],
    [x[3] for x in domain_1_data]
)
fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=4, shuffle=True)

# Load Pre-trained Weights
model.load_state_dict(torch.load("pretrained_tnr.pth"))

# Fine-tune on Domain 1
fine_tune_epochs = 5
for epoch in range(fine_tune_epochs):
    model.train()
    total_loss = 0
    for user_id, item_id, rating, domain_id in fine_tune_dataloader:
        optimizer.zero_grad()
        predictions = model(user_id, item_id, domain_id)
        loss = criterion(predictions, rating)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Fine-tune Epoch {epoch+1}/{fine_tune_epochs}, Loss: {total_loss:.4f}")



"""
create synthetic dataset for movies music books 
normalize the dataset 
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import torch


def load_and_preprocess_datasets():
    movies_df = pd.read_csv('module version/imdb_movies.csv')
    # print(movies_df.columns)  
    books_df = pd.read_csv('module version/data.csv')
    # print(books_df.columns)
    movies_df = movies_df[['names','score']].rename(columns={'names': 'name', 'score': 'rating'})
    books_df = books_df[['title','average_rating']].rename(columns={'title':'name','average_rating': 'rating'})
    
    movies_df.dropna(subset=['name', 'rating'], inplace=True)
    books_df.dropna(subset=['name', 'rating'], inplace=True)
    
    movies_df['domain'] = 'movies'
    books_df['domain'] = 'books'
    
    return movies_df,books_df


def create_multi_domain_dataset(movies_df, books_df, n_samples=1000):
    domain_sample_size = n_samples // 2
    
    movies_sample = movies_df.sample(n=domain_sample_size, random_state=1)
    books_sample = books_df.sample(n=domain_sample_size, random_state=1)
    
    combined_df = pd.concat([movies_sample, books_sample], ignore_index=True)
    scaler = MinMaxScaler()
    
    combined_df['rating'] = scaler.fit_transform(combined_df[['rating']])
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df
    
    
def create_torch_dataset(combined_df):
    def __init__(self,df):
        self.names = df['name'].values
        self.ratings = df['rating'].values
        self.domains = pd.factorize(df['domain'])[0]
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        return{
            "name": self.names[idx],
            "rating": torch.tensor(self.ratings[idx]),
            "domain": torch.tensor(self.domains[idx])
        }

if __name__ == "__main__":
    movies_df,books_df = load_and_preprocess_datasets()
    combined_df = create_multi_domain_dataset(
        movies_df,
        books_df,
        n_samples = 1000
    )

    print("\nDataset Statistics:")
    print(f"Total samples: {len(combined_df)}")
    print("\nSamples per domain:")
    print(combined_df['domain'].value_counts())
    print("\nRating distribution:")
    print(combined_df['rating'].describe())

    torch_dataset = create_torch_dataset(combined_df)
    combined_df.to_csv("multi-domain synthetic dataset.csv",index=False)
    print("dataset saved !!")
    
    
     
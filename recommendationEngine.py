import os
import subprocess
try:
    import pandas as pd 
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    import numpy as np
except:
    directory=os.path.join(os.getcwd(),'requirements.cmd')
    subprocess.call(directory)
    
    
class RecommendationEngine:
    def __init__(self,dataset_path,item_col,content_col=None,rating_col=None):
        self.dataset_path = dataset_path
        self.content_col = content_col
        self.item_col = item_col
        self.rating_col = rating_col
        print(dataset_path)
        try:
            self.df = pd.read_csv(dataset_path)
        except pd.errors.ParserError:
            print("Parsing Error: The file must be csv")
        self.df[content_col] = self.df[content_col].fillna('')
        self.df[rating_col] = self.df[rating_col].fillna("")
        self.cosine_similarity_matrix = None
        self.euclidean_distance_matrix = None
        self.indexes = None
        self.engine()
        
    def engine(self):
        #de duping the item_col in the dataset
        self.df =self.df.drop_duplicates(subset=[self.item_col]).reset_index(drop=True)
        
        tokenizer = TfidfVectorizer(stop_words="english")
        #cosine similarity matrix
        tokenizer_matrix = tokenizer.fit_transform(self.df[self.content_col])
        self.cosine_similarity_matrix = linear_kernel(tokenizer_matrix,tokenizer_matrix)
        
        # euclidean distance matrix
        # tokenizer_rating = tokenizer.fit_transform(self.df[self.rating_col])
        # euclidean_distance = linear_kernel(tokenizer_rating,tokenizer_rating)
        # self.euclidean_distance_matrix = np.diagonal(euclidean_distance)
        # self.euclidean_distance_matrix = np.sqrt(self.euclidean_distance_matrix[:,None]+self.euclidean_distance_matrix[None,:]-2*euclidean_distance)
        # print(self.euclidean_distance_rating)
        
        if self.rating_col is not None:
            self.df[self.rating_col] = pd.to_numeric(self.df[self.rating_col],errors="coerce")
            self.df[self.rating_col] = self.df[self.rating_col].fillna(0)
            rating_array = self.df[self.rating_col].values
            self.euclidean_distance_matrix = np.sqrt(((rating_array[:,np.newaxis]-rating_array)**2))
        else:
            self.euclidean_distance_matrix = None
                
        self.indexes = pd.Series(self.df.index,index=self.df[self.item_col]).drop_duplicates()
        # print("After Deduping",self.indexes)
        
    def contentBasedRecommendations(self,item_name,no_recommendations=10):
        if item_name not in self.indexes:
            raise ValueError(f"Item {item_name} not found in the database")
        if self.content_col is None:
            raise ValueError("Argument content_col not passed")
        indx = self.indexes[item_name]
        #using cosine simmilarity for text based recommendations
        similarityScores = list(enumerate(self.cosine_similarity_matrix[indx]))
        
        
        #similarity scores 
        similarityScores = [(idx, score[0] if isinstance(score, np.ndarray) else float(score))for idx, score in similarityScores]
        # print(similarityScores)   
        similarityScores = sorted(similarityScores,key=lambda x:x [1],reverse=True)
        # print(similarityScores)
        # recommendations = similarityScores[1:no_recommendations+1]
        recommendations = [i[0] for i in similarityScores[1:no_recommendations+1]]
        print(recommendations)
        return self.df[self.item_col].iloc[recommendations].tolist()
    
    def ratingBasedRecommendations(self,item_name,no_recommendations=10):
        if item_name not in self.indexes:
            raise ValueError(f"Item {item_name} not found in the database")
        if self.rating_col is None:
            raise ValueError("Argument rating_col not passed")
        indx = self.indexes[item_name]
        print(indx)
        #calculating euclidean distance 
        euclideanScores = list(enumerate(self.euclidean_distance_matrix[indx]))
        euclideanScores = sorted(euclideanScores,key=lambda x:x [1],reverse=True)
        # recommendations = similarityScores[1:no_recommendations+1]
        recommendations = [i[0] for i in euclideanScores[1:no_recommendations+1]]
        return self.df[self.item_col].iloc[recommendations].tolist()

    
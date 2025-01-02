from recommendationEngine import RecommendationEngine

engine = RecommendationEngine(
    dataset_path='./amazon_products_india.csv',
    content_col='name',
    rating_col='rating',
    item_col='name')

# engine = RecommendationEngine(
#     dataset_path='./netflix_titles.csv',
#     content_col='description',
#     rating_col='rating',
#     item_col='title'
# )



# print(engine.contentBasedRecommendations('Power Rangers Zeo'))

# print(engine.contentBasedRecommendations('Mysore-Sandal-Soaps-Pack-Bars'))
print(engine.ratingBasedRecommendations('Mysore-Sandal-Soaps-Pack-Bars'))
# print(engine.ratingBasedRecommendations('Power Rangers Zeo'))

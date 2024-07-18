import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense # type: ignore

# Step 1: Generate a sample dataset
def generate_sample_data():
    users = np.arange(1, 101)  # 100 users
    items = np.arange(1, 101)  # 100 items
    data = []

    for user in users:
        num_items = np.random.randint(5, 20)  # Each user interacts with 5 to 20 items
        items_purchased = np.random.choice(items, num_items, replace=False)
        for item in items_purchased:
            rating = np.random.randint(1, 6)  # Random rating between 1 and 5
            data.append([user, item, rating])

    df = pd.DataFrame(data, columns=['userId', 'itemId', 'rating'])
    return df

# Generate data
data = generate_sample_data()

# Step 2: Exploratory Data Analysis (EDA)
# Plot the distribution of ratings
sns.histplot(data['rating'], bins=5, kde=False)
plt.xlabel('Rating')
plt.ylabel('Count')
plt.title('Distribution of Product Ratings')
plt.show()

# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Step 3: Collaborative Filtering using Singular Value Decomposition (SVD)
# Create user-item matrix
user_item_matrix = data.pivot(index='userId', columns='itemId', values='rating').fillna(0)
U, sigma, Vt = svds(user_item_matrix, k=50)  # Perform SVD
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns, index=user_item_matrix.index)

# Step 4: Content-Based Filtering using Cosine Similarity
item_features = np.random.rand(len(data['itemId'].unique()), 20)  # Random item features
cosine_sim = cosine_similarity(item_features, item_features)
item_indices = pd.Series(data['itemId'].unique())

def get_recommendations(item_id, cosine_sim=cosine_sim):
    idx = item_indices[item_indices == item_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    item_indices = [i[0] for i in sim_scores]
    return data['itemId'].unique()[item_indices]

# Get recommendations for item_id 1
print("Recommendations for item 1:", get_recommendations(1))

# Step 5: Evaluation Metrics
def get_rmse(predictions, ground_truth):
    predictions = predictions[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(predictions, ground_truth))

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_matrix = train_data.pivot(index='userId', columns='itemId', values='rating').fillna(0)
test_matrix = test_data.pivot(index='userId', columns='itemId', values='rating').fillna(0)
U_train, sigma_train, Vt_train = svds(train_matrix, k=50)
sigma_train = np.diag(sigma_train)
predicted_train_ratings = np.dot(np.dot(U_train, sigma_train), Vt_train)
predicted_train_ratings_df = pd.DataFrame(predicted_train_ratings, columns=train_matrix.columns, index=train_matrix.index)
train_rmse = get_rmse(predicted_train_ratings_df.values, train_matrix.values)
test_rmse = get_rmse(predicted_train_ratings_df.values, test_matrix.values)
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Step 6: Neural Network-based Recommendation Model using TensorFlow
class RecommenderNet(Model):
    def __init__(self, num_users, num_items, embedding_size):
        super(RecommenderNet, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.item_embedding = Embedding(num_items, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=tf.keras.regularizers.l2(1e-6))
        self.user_flatten = Flatten()
        self.item_flatten = Flatten()
        self.dot = Dot(axes=1)

    def call(self, inputs):
        user_vector = self.user_flatten(self.user_embedding(inputs[0]))
        item_vector = self.item_flatten(self.item_embedding(inputs[1]))
        return self.dot([user_vector, item_vector])

# Prepare data for the neural network
user_ids = data['userId'].unique().astype(np.int32)
item_ids = data['itemId'].unique().astype(np.int32)
num_users = len(user_ids)
num_items = len(item_ids)
embedding_size = 50
model = RecommenderNet(num_users, num_items, embedding_size)
model.compile(optimizer='adam', loss='mse')
history = model.fit([data['userId'], data['itemId']], data['rating'], epochs=5, batch_size=64, validation_split=0.2)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import silhouette_score
import seaborn as sns
from minisom import MiniSom  # Import MiniSom for SOM

# Load your dataset
try:
    df = pd.read_csv('online_retail_II.csv', encoding='ISO-8859-1')
except UnicodeDecodeError:
    # If there is an encoding error, try a different encoding
    df = pd.read_csv('online_retail_II.csv', encoding='utf-8', errors='replace')

# Display columns
print(df.columns)

# Create TotalSpend column
df['TotalSpend'] = df['Quantity'] * df['Price']

# Box plot to show the distribution of 'TotalSpend' and 'Quantity'
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['TotalSpend'])
plt.title('Boxplot of TotalSpend')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Quantity'])
plt.title('Boxplot of Quantity')
plt.show()

# Scatter plot to visualize the relationship between 'TotalSpend' and 'Quantity'
plt.figure(figsize=(10, 6))
plt.scatter(df['Quantity'], df['TotalSpend'], alpha=0.5)
plt.title('Scatter plot of Quantity vs TotalSpend')
plt.xlabel('Quantity')
plt.ylabel('TotalSpend')
plt.show()

# Scatter plot to visualize the relationship between 'TotalSpend' and 'Invoice' (to check for any pattern)
plt.figure(figsize=(10, 6))
plt.scatter(df['Invoice'], df['TotalSpend'], alpha=0.5)
plt.title('Scatter plot of Invoice vs TotalSpend')
plt.xlabel('Invoice')
plt.ylabel('TotalSpend')
plt.show()

# Pairplot to visualize the relationships between 'TotalSpend' and 'Quantity'
sns.pairplot(df[['Quantity', 'TotalSpend']], diag_kind='kde')
plt.suptitle('Pairplot of Quantity and TotalSpend', y=1.02)
plt.show()

# Data Cleaning
# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Handle missing values
df.dropna(subset=['Quantity', 'Price'], inplace=True)  # Drop rows where Quantity or Price is NaN

# 3. Remove negative values in Quantity and Price
df = df[(df['Quantity'] >= 0) & (df['Price'] >= 0)]

# 5. Optional: Convert Invoice to a categorical variable if it's not numeric
df['Invoice'] = df['Invoice'].astype('category').cat.codes

# Outlier treatment function
def remove_outliers(df):
    return df[(df['TotalSpend'] >= df['TotalSpend'].quantile(0.01)) & (df['TotalSpend'] <= df['TotalSpend'].quantile(0.99))]

# Remove outliers
df = remove_outliers(df)

# Sample a subset of the dataset if it's too large
df_sample = df.sample(frac=0.1, random_state=42)  # Take 10% of the data

# View first few rows of the cleaned dataset
print(df_sample.head())

# Select features for clustering
X_numeric = df_sample[['TotalSpend']]
X = df_sample[['TotalSpend', 'Invoice']]

# Normalize the data (scaling the features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Update the scaled values in the DataFrame
df_sample[['TotalSpend', 'Invoice']] = X_scaled

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters
kmeans.fit(X_scaled)

# Get the predicted cluster labels
df_sample['KMeansCluster'] = kmeans.labels_

# Save the trained KMeans model for later use
joblib.dump(kmeans, 'kmeans_model.pkl')

# Optionally, visualize the KMeans clustering results
plt.scatter(df_sample['TotalSpend'], df_sample['Invoice'], c=df_sample['KMeansCluster'], cmap='viridis')
plt.xlabel('TotalSpend')
plt.ylabel('Invoice')
plt.title('KMeans Clustering')
plt.show()

# DBSCAN Clustering
dbscan = DBSCAN(eps=0.7, min_samples=5)  # eps is the maximum distance between two points to be considered neighbors
dbscan_labels = dbscan.fit_predict(X_scaled)

# Add the DBSCAN labels to the DataFrame
df_sample['DBSCANCluster'] = dbscan_labels

# Save the trained DBSCAN model for later use
joblib.dump(dbscan, 'dbscan_model.pkl')

# Visualize the DBSCAN results
plt.scatter(df_sample['TotalSpend'], df_sample['Invoice'], c=df_sample['DBSCANCluster'], cmap='plasma')
plt.xlabel('TotalSpend')
plt.ylabel('Invoice')
plt.title('DBSCAN Clustering')
plt.show()

# GMM Clustering (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, random_state=42)  # 3 components
gmm.fit(X_scaled)

# Get the predicted cluster labels
df_sample['GMMCluster'] = gmm.predict(X_scaled)

# Visualize the GMM clustering results
plt.scatter(df_sample['TotalSpend'], df_sample['Invoice'], c=df_sample['GMMCluster'], cmap='coolwarm')
plt.xlabel('TotalSpend')
plt.ylabel('Invoice')
plt.title('Gaussian Mixture Model Clustering')
plt.show()

# SOM (Self Organizing Map)
som = MiniSom(x=10, y=10, input_len=2, sigma=1.0, learning_rate=0.5)  # 10x10 SOM grid
som.random_weights_init(X_scaled)  # Initialize the SOM with random weights
som.train_random(X_scaled, num_iteration=100)  # Train the SOM

# Get the winning neurons for each data point
win_map = som.win_map(X_scaled)

# Get SOM clusters by finding the closest neurons
som_labels = [som.winner(x) for x in X_scaled]

# Add SOM cluster labels to the DataFrame
df_sample['SOMCluster'] = [som_labels.index(x) for x in som_labels]

# Save the trained GMM and SOM models for later use
joblib.dump(gmm, 'gmm_model.pkl')
joblib.dump(som, 'som_model.pkl')

# Load the trained models
kmeans = joblib.load('kmeans_model.pkl')
dbscan = joblib.load('dbscan_model.pkl')
gmm = joblib.load('gmm_model.pkl')
som = joblib.load('som_model.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')

# New data prediction
new_data = pd.DataFrame({
    'TotalSpend': [500, 1500],
    'Invoice': [12345, 67890]
})

# scale only TotalSpend
new_data_scaled = scaler.transform(new_data[['TotalSpend', 'Invoice']])

# KMeans Prediction
kmeans_predictions = kmeans.predict(new_data_scaled)

# DBSCAN Prediction
dbscan_predictions = dbscan.fit_predict(new_data_scaled)

# GMM Prediction
gmm_predictions = gmm.predict(new_data_scaled)

# SOM Prediction
som_predictions = [som.winner(x) for x in new_data_scaled]

# Print predictions
print("KMeans Predictions:", kmeans_predictions)
print("DBSCAN Predictions:", dbscan_predictions)
print("GMM Predictions:", gmm_predictions)
print("SOM Predictions:", som_predictions)

# Calculate silhouette score for KMeans
kmeans_silhouette = silhouette_score(X_scaled, df_sample['KMeansCluster'])
print(f"KMeans Silhouette Score: {kmeans_silhouette}")

# Check the number of unique labels in DBSCAN (excluding noise label -1)
unique_dbscan_labels = df_sample['DBSCANCluster'].unique()
if len(unique_dbscan_labels) > 1:
    dbscan_silhouette = silhouette_score(X_scaled, df_sample['DBSCANCluster'][df_sample['DBSCANCluster'] != -1])
    print(f"DBSCAN Silhouette Score: {dbscan_silhouette}")
else:
    print("DBSCAN did not form multiple clusters. Silhouette Score cannot be calculated.")

# GMM Silhouette Score
gmm_silhouette = silhouette_score(X_scaled, df_sample['GMMCluster'])
print(f"GMM Silhouette Score: {gmm_silhouette}")

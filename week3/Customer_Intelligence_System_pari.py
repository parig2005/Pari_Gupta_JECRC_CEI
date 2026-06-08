# =====================================================
# CUSTOMER INTELLIGENCE SYSTEM
# Classification + Ensemble Learning + Clustering
# =====================================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# STEP 1: LOAD DATA
# =====================================================

df = pd.read_csv("Country-data.csv")

print("\nDataset Shape:", df.shape)
print(df.head())

# =====================================================
# STEP 2: DATA PREPROCESSING
# =====================================================

country_names = df["country"]

X = df.drop("country", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# STEP 3: K-MEANS CLUSTERING
# =====================================================

wcss = []

for k in range(2, 11):
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    km.fit(X_scaled)
    wcss.append(km.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(2,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.show()

# Optimal clusters
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = cluster_labels

print("\nCluster Distribution:")
print(df["Cluster"].value_counts())

# Silhouette Score
score = silhouette_score(
    X_scaled,
    cluster_labels
)

print("\nSilhouette Score:", score)

# =====================================================
# STEP 4: DBSCAN CLUSTERING
# =====================================================

dbscan = DBSCAN(
    eps=1.5,
    min_samples=5
)

db_labels = dbscan.fit_predict(X_scaled)

df["DBSCAN_Cluster"] = db_labels

print("\nDBSCAN Clusters:")
print(df["DBSCAN_Cluster"].value_counts())

# =====================================================
# STEP 5: VISUALIZATION
# =====================================================

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))

sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=cluster_labels,
    palette="Set1"
)

plt.title("K-Means Customer Segments")
plt.show()

# =====================================================
# STEP 6: CLASSIFICATION
# Predict Customer Segment
# =====================================================

y = cluster_labels

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# RANDOM FOREST
# =====================================================

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n========== RANDOM FOREST ==========")

print(
    "Accuracy:",
    accuracy_score(y_test, rf_pred)
)

print(
    classification_report(
        y_test,
        rf_pred
    )
)

# =====================================================
# XGBOOST
# =====================================================

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective="multi:softmax",
    num_class=3,
    random_state=42
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

print("\n========== XGBOOST ==========")

print(
    "Accuracy:",
    accuracy_score(y_test, xgb_pred)
)

print(
    classification_report(
        y_test,
        xgb_pred
    )
)

# =====================================================
# STEP 7: MODEL COMPARISON
# =====================================================

rf_acc = accuracy_score(y_test, rf_pred)
xgb_acc = accuracy_score(y_test, xgb_pred)

comparison = pd.DataFrame({
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [rf_acc, xgb_acc]
})

print("\nModel Comparison")
print(comparison)

sns.barplot(
    data=comparison,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.show()

# =====================================================
# STEP 8: FEATURE IMPORTANCE
# =====================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feature_importance,
    x="Importance",
    y="Feature"
)

plt.title("Random Forest Feature Importance")
plt.show()

# =====================================================
# STEP 9: CUSTOMER SEGMENT INSIGHTS
# =====================================================

segment_summary = df.groupby("Cluster").mean(
    numeric_only=True
)

print("\nSegment Summary")
print(segment_summary)

# Save output

df.to_csv(
    "Customer_Intelligence_Output.csv",
    index=False
)

print("\nProject Completed Successfully!")

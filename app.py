import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import base64
from flask import Flask, render_template
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score



app = Flask(__name__)

def run_clustering_and_get_outputs():
    # Load Dataset
    df = pd.read_csv("Mall_Customers.csv")

    # Encode kolom Gender jadi angka
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # Pilih fitur yang akan digunakan
    features = ["CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
    X = df[features]

    # Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Menentukan jumlah cluster dengan metode Elbow
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plot hasil Metode Elbow jadi gambar base64
    fig, ax = plt.subplots()
    ax.plot(k_range, inertia, marker='o')
    ax.set_xlabel('Jumlah Cluster (K)')
    ax.set_ylabel('Inertia')
    ax.set_title('Metode Elbow untuk Menentukan K Optimal')
    elbow_buf = io.BytesIO()
    plt.savefig(elbow_buf, format='png')
    elbow_buf.seek(0)
    elbow_img = base64.b64encode(elbow_buf.getvalue()).decode()
    plt.close(fig)

    # Menggunakan K optimal (misalnya: K=3)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    # Buat plot clustering (Annual Income vs Spending Score)
    fig, ax = plt.subplots()
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis', ax=ax)
    ax.set_title('Hasil Clustering Mall Customer')
    ax.set_xlabel('Annual Income (k$)')
    ax.set_ylabel('Spending Score (1-100)')
    cluster_buf = io.BytesIO()
    plt.savefig(cluster_buf, format='png')
    cluster_buf.seek(0)
    cluster_img = base64.b64encode(cluster_buf.getvalue()).decode()
    plt.close(fig)

    # Menghitung silhouette score untuk setiap nilai k
    silhouette_scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)

    # Plot silhouette score
    fig, ax = plt.subplots()
    ax.plot(range(2, 10), silhouette_scores, marker='o', color='green')
    ax.set_xlabel('Jumlah Cluster (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score untuk Menentukan K Optimal')
    silhouette_buf = io.BytesIO()
    plt.savefig(silhouette_buf, format='png')
    silhouette_buf.seek(0)
    silhouette_img = base64.b64encode(silhouette_buf.getvalue()).decode()
    plt.close(fig)

    # Konversi dataframe hasil cluster ke HTML tabel
    cluster_table = df[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].to_html(classes='data', header="true", index=False)

    return cluster_table, elbow_img, cluster_img, silhouette_img

@app.route('/')
def index():
    cluster_table, elbow_img, cluster_img, silhouette_img = run_clustering_and_get_outputs()
    return render_template('index.html',
                           cluster_table=cluster_table,
                           elbow_img=elbow_img,
                           cluster_img=cluster_img,
                           silhouette_img=silhouette_img)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

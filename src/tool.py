from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from time import time

def Hungarian_matching(y, y_pred):
  cm = confusion_matrix(y, y_pred)
  row_ind, col_ind = linear_sum_assignment(-cm)
  mapping = dict(zip(col_ind, row_ind))
  y_pred_aligned = np.array([mapping[label] for label in y_pred])
  return y_pred_aligned

def k_mean_classify(pca_list, X, y, FIG_DIR):
  results = []
  for name, pca in pca_list:
    start = time()
    if pca == None:
      X_pca = X
    else:
      X_pca = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=10, random_state=42).fit(X_pca)
    end = time()

    y_pred = kmeans.labels_
    inertia = kmeans.inertia_
    y_pred = Hungarian_matching(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)

    results.append(
        {
            "name": name,
            "accuracy": accuracy,
            "ari": ari,
            "nmi": nmi,
            "inertia": inertia,
            "time": end - start
        }
    )

    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(FIG_DIR / f"{name}_cm.png")
    plt.show()

    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_pca)
    center_pca = pca2.transform(kmeans.cluster_centers_)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca2[:,0],X_pca2[:,1],c=y_pred, s=5, cmap="rainbow")
    plt.scatter(center_pca[:,0],center_pca[:,1],c="black",marker="X",s=30,alpha=0.5)
    plt.title(f"{name} K-means Clusters")
    plt.savefig(FIG_DIR / f"{name}_k-means_clusters.png")
    plt.show()

  results_df = pd.DataFrame(results)
  return results_df

def spectral_clustering(pca_list, X, y, FIG_DIR):
  results = []
  for name, pca in pca_list:
    start = time()
    if pca == None:
      X_pca = X
    else:
      X_pca = pca.fit_transform(X)
    spectral = SpectralClustering(n_clusters=10, affinity="nearest_neighbors", assign_labels="kmeans", random_state=42).fit(X_pca)
    end = time()

    y_pred = spectral.fit_predict(X_pca)
    y_pred = Hungarian_matching(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)

    results.append(
        {
            "name": name,
            "accuracy": accuracy,
            "ari": ari,
            "nmi": nmi,
            "inertia": None,
            "time": end - start
        }
    )

    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title(f"{name} Confusion Matrix (Spectral Clustering)")
    plt.savefig(FIG_DIR / f"{name}_cm.png")
    plt.show()

    pca2 = PCA(n_components=2)
    X_pca2 = pca2.fit_transform(X_pca)
    plt.figure(figsize=(8,6))
    plt.scatter(X_pca2[:,0],X_pca2[:,1],c=y_pred, s=5, cmap="rainbow")
    plt.title(f"{name} Spectral Clustering")
    plt.savefig(FIG_DIR / f"{name}_spectral_clustering.png")
    plt.show()

  results_df = pd.DataFrame(results)
  return results_df

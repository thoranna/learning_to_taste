import numpy as np
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from packages.tste.tste import tste
from umap import UMAP
from imblearn.over_sampling import RandomOverSampler
from scripts.prediction import DICTIONARIES
import statistics
from utils.set_seed import RANDOM_SEED


def reduce_data1(data, method, ids):
    if method == 'MDS':
        optimal_mds_lis = []
        n_inits = [5, 7, 10]
        max_iters = [300, 400, 500, 600]
        eps_values = [1e-3, 1e-4, 1e-5]
        for _, d in DICTIONARIES.items():
            labels = [d[int(key)] for key in ids]
            if isinstance(labels[0], str):
                le = LabelEncoder()
                labels = le.fit_transform(labels)
            optimal_mds = find_optimal_mds(data, labels, n_inits,
                                           max_iters, eps_values)
            optimal_mds_lis.append(optimal_mds)
        optimal_mds = statistics.mode(optimal_mds_lis)
        return optimal_mds.fit_transform(data)
    elif method == 't-STE':
        return tste(np.array(data))

def reduce_data2(data, method, ids):
    if method == 'TSNE':
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED)
        return tsne.fit_transform(data)
    elif method == 'PCA':
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        return pca.fit_transform(data)
    elif method == 'Umap':
        umap = UMAP(n_components=2, random_state=RANDOM_SEED)
        return umap.fit_transform(data)

def find_optimal_mds(dist_matrix, labels, n_inits, max_iters, eps_values, cv=5):
    best_score = -np.inf
    best_params = None

    # Initialize the RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    # Iterate through the parameter combinations
    for n_init in n_inits:
        for max_iter in max_iters:
            for eps in eps_values:
                # Create MDS with the current parameters
                mds = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=RANDOM_SEED, n_init=n_init, max_iter=max_iter, eps=eps)

                # Obtain embeddings using MDS
                embeddings = mds.fit_transform(dist_matrix)

                # Oversample the embeddings
                embeddings_resampled, labels_resampled = ros.fit_resample(embeddings, labels)

                # Create a KNeighborsClassifier
                knn = KNeighborsClassifier()

                # Evaluate classifier performance using cross-validation
                scores = cross_val_score(knn, embeddings_resampled, labels_resampled, cv=cv)
                mean_score = np.mean(scores)

                # Update best_score and best_params if a better score is found
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'n_init': n_init, 'max_iter': max_iter, 'eps': eps}

    # Create and return an MDS object with the optimal parameters
    optimal_mds = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=42, **best_params)
    return optimal_mds
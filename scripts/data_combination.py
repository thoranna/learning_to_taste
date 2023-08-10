import numpy as np
from sklearn.cross_decomposition import CCA
from packages.snack.embedding.snack import SNaCK
from packages.icp.icp import icp
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes

def combine_data(data1, data2, unique_ids1, unique_ids2, method, preprocessing_method1):
    if method == 'CCA':
        data1, common_experiment_ids1, data2, common_experiment_ids2 = align_embedding_matrices(data1, unique_ids1, data2, unique_ids2)
        cca = CCA(n_components=2)
        cca.fit(data1, data2)
        data1_embeddings_aligned, data2_embeddings_aligned = cca.transform(data1, data2)
        # Assign weights
        combined_embedding = np.hstack((data1_embeddings_aligned, data2_embeddings_aligned))
        if len(common_experiment_ids1) > len(common_experiment_ids2):
            return combined_embedding, common_experiment_ids2, data1, common_experiment_ids1, data2, common_experiment_ids2 
        else:
            return combined_embedding, common_experiment_ids1, data1, common_experiment_ids1, data2, common_experiment_ids2 
    elif method == 'ICP':
        data1, common_experiment_ids1, data2, common_experiment_ids2 = align_embedding_matrices(data1, unique_ids1, data2, unique_ids2)
        _, combined_embedding = icp(data1, data2)
        if len(common_experiment_ids1) > len(common_experiment_ids2):
            return combined_embedding, common_experiment_ids2, data1, common_experiment_ids1, data2, common_experiment_ids2 
        else:
            return combined_embedding, common_experiment_ids1, data1, common_experiment_ids1, data2, common_experiment_ids2 
    elif method == 'SNaCK':
        aligned_triplet_list, aligned_embedding_matrix, aligned_experiment_ids, _ = align_triplets_and_embedding_matrix(data1, data2, unique_ids2)
        aligned_triplet_list = np.array(aligned_triplet_list)
        N, _ = aligned_embedding_matrix.shape
        data2 -= np.mean(data2, 0)
        data2 /= np.max(np.abs(data2))
        snack = SNaCK(N)
        combined_embedding = snack.snack_embed(aligned_embedding_matrix.astype(np.float64), aligned_triplet_list)
        return combined_embedding, aligned_experiment_ids, None, None, None, None
    if method == 'procrustes':
        data1, common_experiment_ids1, data2, common_experiment_ids2 = align_embedding_matrices(data1, unique_ids1, data2, unique_ids2)
        
        # Compute the Procrustes transformation matrix
        R, scale = orthogonal_procrustes(data1, data2)
        
        # Transform data2 using the computed matrix
        data2_embeddings_aligned = np.dot(data2, R) * scale
        
        # Combine embeddings
        combined_embedding = np.hstack((data1, data2_embeddings_aligned))
        
        if len(common_experiment_ids1) > len(common_experiment_ids2):
            return combined_embedding, common_experiment_ids2, data1, common_experiment_ids1, data2_embeddings_aligned, common_experiment_ids2 
        else:
            return combined_embedding, common_experiment_ids1, data1, common_experiment_ids1, data2_embeddings_aligned, common_experiment_ids2

def align_embedding_matrices(embedding_matrix1, experiment_ids1, embedding_matrix2, experiment_ids2):
    # Ensure that both experiment_ids1 and experiment_ids2 are lists of integers
    experiment_ids1 = [int(exp_id) for exp_id in experiment_ids1]
    experiment_ids2 = [int(exp_id) for exp_id in experiment_ids2]
    
    # Find the common experiment IDs between the two lists
    common_experiment_ids = np.intersect1d(experiment_ids1, experiment_ids2)

    # Align the first embedding matrix and experiment IDs list
    aligned_embedding_matrix1 = []
    aligned_experiment_ids1 = []
    for exp_id in common_experiment_ids:
        if exp_id in experiment_ids1:
            idx = np.where(experiment_ids1 == exp_id)[0][0]
            aligned_embedding_matrix1.append(embedding_matrix1[idx])
            aligned_experiment_ids1.append(exp_id)

    aligned_embedding_matrix1 = np.array(aligned_embedding_matrix1)
    aligned_experiment_ids1 = np.array(aligned_experiment_ids1)

    # Align the second embedding matrix and experiment IDs list
    aligned_embedding_matrix2 = []
    aligned_experiment_ids2 = []
    for exp_id in common_experiment_ids:
        if exp_id in experiment_ids2:
            idx = np.where(experiment_ids2 == exp_id)[0][0]
            aligned_embedding_matrix2.append(embedding_matrix2[idx])
            aligned_experiment_ids2.append(exp_id)

    aligned_embedding_matrix2 = np.array(aligned_embedding_matrix2)
    aligned_experiment_ids2 = np.array(aligned_experiment_ids2)

    assert np.all(aligned_experiment_ids1 == aligned_experiment_ids2), "Aligned experiment IDs do not match."

    return aligned_embedding_matrix1, aligned_experiment_ids1, aligned_embedding_matrix2, aligned_experiment_ids2


def align_triplets_and_embedding_matrix(triplet_list, embedding_matrix, experiment_ids):
    # Create a set of unique experiment IDs from the triplets
    unique_triplet_ids = set()
    for triplet in triplet_list:
        unique_triplet_ids.update(triplet)

    # Find the common experiment IDs between the triplets and the embedding matrix
    common_experiment_ids = np.intersect1d(list(unique_triplet_ids), experiment_ids)

    # Align the embedding matrix and experiment IDs list
    aligned_embedding_matrix = []
    aligned_experiment_ids = []
    for exp_id in common_experiment_ids:
        if exp_id in experiment_ids:
            idx = np.where(experiment_ids == exp_id)[0][0]
            aligned_embedding_matrix.append(embedding_matrix[idx])
            aligned_experiment_ids.append(exp_id)

    aligned_embedding_matrix = np.array(aligned_embedding_matrix)
    aligned_experiment_ids = np.array(aligned_experiment_ids)

    # Create a dictionary to map old experiment IDs to new indices
    id_to_new_index = {exp_id: index for index, exp_id in enumerate(aligned_experiment_ids)}

    # Update the triplet list with the new experiment IDs and keep track of the changes
    aligned_triplet_list = []
    changed_triplets = []
    for triplet in triplet_list:
        if all(exp_id in aligned_experiment_ids for exp_id in triplet):
            new_triplet = tuple(id_to_new_index[exp_id] for exp_id in triplet)
            aligned_triplet_list.append(new_triplet)
            changed_triplets.append((triplet, new_triplet))

    return aligned_triplet_list, aligned_embedding_matrix, aligned_experiment_ids, changed_triplets
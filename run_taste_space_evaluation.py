import scripts.data_source1_utils as data_source1_utils
import scripts.data_source2_utils as data_source2_utils

import scripts.preprocessing as preprocessing
import scripts.data_combination as data_combination
import scripts.model_fitting_new as model_fitting
from packages.snack.embedding.snack import SNaCK
from collections import defaultdict

import random
import numpy as np

def generate_triplets(embeddings, idx_to_id):
    triplets = []
    # calculate the pairwise distance matrix
    dist_matrix = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
    n = embeddings.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j:  # i and j must be different
                for k in range(n):
                    if i != k and j != k:  # i, j, and k must all be different
                        if (dist_matrix[i, j] < dist_matrix[i, k]):  # i is closer to j than to k
                            triplet = (idx_to_id[i], idx_to_id[j], idx_to_id[k])
                            triplets.append(triplet)
    return triplets

def agreement_ratio(triplets_combined_embed, triplets_test_set_embed):
    # Convert the lists to sets for efficient lookup
    triplets_combined_set = set(triplets_combined_embed)
    triplets_test_set = set(triplets_test_set_embed)

    triplets_combined_set_intersect = set()
    for triplet in triplets_test_set:
        if triplet in triplets_combined_set:
            triplets_combined_set_intersect.add(triplet)
        elif (triplet[1], triplet[0], triplet[2]) in triplets_combined_set:
            triplets_combined_set_intersect.add((triplet[1], triplet[0], triplet[2]))

    # Count the number of triplets in triplets_combined_set that also appear in triplets_test_set
    agreement_count = len(triplets_combined_set_intersect & triplets_test_set)
    print("Agreement count: ", agreement_count)

    # Count the number of triplets in triplets_combined_set that do not appear in triplets_test_set
    disagreement_count = len(triplets_combined_set_intersect - triplets_test_set)
    print("Disagreement count: ", disagreement_count)

    # Calculate the agreement ratio
    ratio = agreement_count / (agreement_count + disagreement_count)

    return ratio

data_source1a, data_source1b = data_source1_utils.load_data()
data_source2 = data_source2_utils.load_data()

preprocessing_method1 = 'triplets'
data_combination_method = 'SNaCK'
model_to_use = 'distil_bert'

preprocessed_data1, unique_ids1, id_to_index = preprocessing.preprocess_data_source1(data_source1a, data_source1b, method=preprocessing_method1)
unique_ids1 = [int(item) for item in  unique_ids1]

preprocessed_data2 = preprocessing.preprocess_data_source2(data_source2)
embedding_matrix2, unique_ids2 = model_fitting.fit_model(model_to_use, preprocessed_data2)

aligned_triplet_list, aligned_embedding_matrix, aligned_experiment_ids, _ = data_combination.align_triplets_and_embedding_matrix(preprocessed_data1, embedding_matrix2, unique_ids1)

def split_triplets(triplet_list, test_ratio=0.3):
    # Set random seed for reproducabilty 
    random.seed(42)
    
    # Create a dictionary where the keys are frozensets of the triplets
    # and the values are lists of the original triplets
    triplet_dict = defaultdict(list)
    for triplet in triplet_list:
        triplet_dict[frozenset(triplet)].append(triplet)

    # Get a list of the unique keys (frozensets)
    unique_keys = list(triplet_dict.keys())

    # Shuffle the list
    random.shuffle(unique_keys)

    # Calculate the split index
    split_index = int(test_ratio * len(unique_keys))

    # Split the keys into test and train
    test_keys = unique_keys[:split_index]
    train_keys = unique_keys[split_index:]

    # Get the original triplets for the test and train sets
    test_triplets = [triplet for key in test_keys for triplet in triplet_dict[key]]
    train_triplets = [triplet for key in train_keys for triplet in triplet_dict[key]]

    # Create sets of all elements in the test and train sets
    test_elements = set([item for triplet in test_triplets for item in triplet])

    # Filter the train set to only include triplets that don't overlap with the test set
    train_triplets = [triplet for triplet in train_triplets if not any(item in test_elements for item in triplet)]

    return train_triplets, test_triplets

aligned_triplet_list_train, aligned_triplet_list_test = split_triplets(aligned_triplet_list, test_ratio=0.05)

# Latent = snack([train text, test text], [train flavour])
# Eval(triplet from latent, triplets form test flavour)

N, _ = aligned_embedding_matrix.shape
snack = SNaCK(N)
aligned_triplet_list_train = np.array(aligned_triplet_list_train)
combined_embedding = snack.snack_embed(aligned_embedding_matrix, aligned_triplet_list_train)
combined_embedding = combined_embedding.detach().numpy()
triplets_combined_embed = generate_triplets(combined_embedding, aligned_experiment_ids)
ratio = agreement_ratio(triplets_combined_embed, aligned_triplet_list_test)
print('The agreement / disagreement ratio is:', ratio)
import scripts.data_source1_utils as data_source1_utils
import scripts.data_source2_utils as data_source2_utils
import scripts.preprocessing as preprocessing
import scripts.data_combination as data_combination
import scripts.model_fitting_new as model_fitting
import scripts.fit_clip as fit_clip
from packages.snack.embedding.snack import SNaCK
from packages.tste.tste import tste
from collections import defaultdict
from sklearn.manifold import TSNE
from packages.icp.icp import icp
from sklearn.manifold import MDS
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import CCA
import sys
import random
import numpy as np
import multiprocessing as mp

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

def split_triplets(triplet_list, test_ratio=0.3, desired_number_of_train_triplets=10):
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

    # Get the original triplets for the test and train sets
    test_keys = unique_keys[:split_index]
    train_keys = unique_keys[split_index:]

    train_triplets = []
    while train_keys and len(train_triplets) < desired_number_of_train_triplets:
        key = train_keys.pop()
        train_triplets.extend(triplet_dict[key])
        test_keys = [k for k in test_keys if not any(item in key for item in k)]  # Remove overlapping triplets from test set

    # Get the remaining original triplets for the test set
    test_triplets = [triplet for key in test_keys for triplet in triplet_dict[key]]

    return train_triplets, test_triplets

def trim_embedding_matrix(original_embedding_matrix, tste_output):
    indices = np.arange(tste_output.shape[0])
    trimmed_matrix = original_embedding_matrix[indices]
    return trimmed_matrix

def remove_overlap(triplets1, triplets2):
    iterations = 10
    # Copy the original lists to avoid modifying them
    triplets1, triplets2 = list(triplets1), list(triplets2)

    for _ in range(iterations):
        # Select the first triplet from the first list
        triplet1 = triplets1[0]

        # Remove the selected triplet from the first list
        triplets1 = triplets1[1:]

        # Remove all overlapping triplets from the second list
        triplets2 = [triplet2 for triplet2 in triplets2 if not any(element in triplet1 for element in triplet2)]
    return triplets1, triplets2


# Method can be euclidean or triplets 
def fit_combine_machine_kernel(data2, data2_method):
    if data2_method == 'UMAP':
        umap = UMAP(n_components=2, random_state=42)
        processed_data2 = umap.fit_transform(data2)
    elif data2_method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42)
        processed_data2 = tsne.fit_transform(data2)

    return processed_data2


# Method can be euclidean or triplets 
def fit_combine(data1, data2, data1_method, data2_method, combination_method):
    if data1_method == 'MDS':
        mds = MDS(n_components=2, random_state=42)
        processed_data1 = mds.fit_transform(data1)
    elif data1_method == 't-STE':
        processed_data1 = tste(np.array(data1))

    if data2_method == 'UMAP':
        umap = UMAP(n_components=2, random_state=42)
        processed_data2 = umap.fit_transform(data2)
    elif data2_method == 't-SNE':
        tsne = TSNE(n_components=2, random_state=42)
        processed_data2 = tsne.fit_transform(data2)

    if combination_method == 'ICP':
        if data1_method == 't-STE':
            processed_data2 = trim_embedding_matrix(processed_data2, processed_data1)
        _, combined_embedding = icp(processed_data2, processed_data1)
    elif combination_method == 'CCA':
        cca = CCA(n_components=2)
        if data1_method == 't-STE':
            processed_data2 = trim_embedding_matrix(processed_data2, processed_data1)
        cca.fit(processed_data1, processed_data2)
        data1_embeddings_aligned, data2_embeddings_aligned = cca.transform(processed_data1, processed_data2)
        combined_embedding = np.hstack((data1_embeddings_aligned, data2_embeddings_aligned))
    return combined_embedding

# Latent = snack([train text, test text], [train flavour])
# Eval(triplet from latent, triplets form test flavour)

if __name__ == "__main__":
    # TEXT: 
    models_to_use_text = ['distil_bert', 't5_small', 'albert', 'bart', 'clip_text']
    # IMAGES: 
    models_to_use_images = ['vit_base', 'deit_small', 'resnet', 'clip_image']
    # IMAGES AND TEXT: 
    models_to_use_multi = ['clip']
    models_to_use = models_to_use_text + models_to_use_images + models_to_use_multi
    data1_processing_methods = ['t-STE', 'MDS']
    data2_processing_methods = ['UMAP', 't-SNE']
    combination_methods = ['CCA', 'ICP']

    original_stdout = sys.stdout
    # Load
    data_source1a, data_source1b = data_source1_utils.load_data()
    data_source2 = data_source2_utils.load_data()
    # Preprocess
    p_d1_triplets, unique_ids1, id_to_index = preprocessing.preprocess_data_source1(data_source1a, data_source1b, method='triplets')
    p_d1_dist, unique_ids1, id_to_index = preprocessing.preprocess_data_source1(data_source1a, data_source1b, method='euclidean')
    p_d2 = preprocessing.preprocess_data_source2(data_source2)

    print("Preprocessing done.. ")
    with open('output.txt', 'w') as f:
        sys.stdout = f
        print("Running the script.. ")
        for model_to_use in models_to_use:
            if model_to_use == 'clip':
                d2_embeds, unique_ids2 = fit_clip.fit_model(model_to_use, p_d2)
            else:
                d2_embeds, unique_ids2 = model_fitting.fit_model(model_to_use, p_d2)
            print("using model: ", model_to_use)
            for d1_p in data1_processing_methods:
                for c in combination_methods:
                    print("fitting.. ")
                    print(".. done!")
                    al_d1_dist, al_ids1, al_d2_dist_b, al_ids2 = data_combination.align_embedding_matrices(p_d1_dist, unique_ids1, d2_embeds, unique_ids2)
                    al_d1_triplets, al_d2_dist_a, al_ids, _ = data_combination.align_triplets_and_embedding_matrix(p_d1_triplets, d2_embeds, unique_ids2)
                    assert np.array_equal(al_ids1, al_ids2), "The arrays are not the same"
                    for d2_p in data2_processing_methods:
                        print("Using model: ", model_to_use)
                        print("Human kernel: ", d1_p)
                        print("Machine kernel: ", d2_p)
                        print("Combined: ", c)
                        if d1_p == 't-STE' and d2_p == 't-SNE':
                            # SNaCK
                            print("Looping in SNaCK. Now SNaCK = Combined")
                            N, _ = al_d2_dist_a.shape
                            snack = SNaCK(N)
                            train_triplets, test_triplets = split_triplets(al_d1_triplets, test_ratio=0.05)
                            al_d2_dist_a = al_d2_dist_a.astype(np.float64)
                            # Make sure dtypes are correct
                            train_triplets = np.array(train_triplets)
                            combined_embed = snack.snack_embed(al_d2_dist_a, train_triplets)
                            combined_embed = combined_embed.detach().numpy()
                            triplets_combined_embed = generate_triplets(combined_embed, al_ids)
                            ratio = agreement_ratio(triplets_combined_embed, test_triplets)
                            print("SNaCK TAR: ", ratio)
                            print("Switching to.. ", c)
                        if d1_p == 't-STE':
                            train_triplets, test_triplets = split_triplets(al_d1_triplets, test_ratio=0.05)
                            combined_embed = fit_combine(train_triplets, al_d2_dist_a, d1_p, d2_p, c)
                            triplets_combined_embed = generate_triplets(combined_embed, al_ids)
                        elif d1_p == 'MDS':
                            train_d1, test_d1, train_d1_labels, test_d1_labels = train_test_split(al_d1_dist, al_ids1, test_size=0.3, random_state=42)
                            train_d2, _ , train_d2_labels, _ = train_test_split(al_d2_dist_b, al_ids2, test_size=0.3, random_state=42)
                            combined_embed = fit_combine(train_d1, train_d2, d1_p, d2_p, c)
                            test_triplets = generate_triplets(test_d1, test_d1_labels)
                            triplets_combined_embed = generate_triplets(combined_embed, al_ids2)

                        ratio = agreement_ratio(triplets_combined_embed, test_triplets)
                        print("TAR: ", ratio)
                        # CHECKING FOR MACHINE KERNEL ONLY
                        m = fit_combine_machine_kernel(al_d2_dist_b, d2_p)
                        triplets_machine_embed = generate_triplets(m, al_ids2)
                        t1, t2 = remove_overlap(al_d1_triplets, triplets_machine_embed)
                        print("TAR MACHINE KERNEL: ", agreement_ratio(t1, t2))

        sys.stdout = original_stdout

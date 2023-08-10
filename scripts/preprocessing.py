import math
import numpy as np
from itertools import combinations
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

def reduce_graph_dimensionality(G, dimensions=64):
    """
    Use Node2Vec to reduce the graph's dimensionality.

    Parameters:
    - G: The input graph
    - dimensions: The desired dimensionality of the embeddings

    Returns:
    - embeddings: Dictionary of node embeddings
    - unique_ids: List of unique node IDs
    - id_to_index: Dictionary mapping from node ID to its index
    """
    
    # Create node embeddings using Node2Vec
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    # Fetch the embeddings for all nodes
    embeddings = {}
    for node in G.nodes():
        embeddings[node] = model.wv[str(node)]
    
    # Create a list of unique node IDs and a mapping from ID to index
    unique_ids = list(G.nodes())
    id_to_index = {node: idx for idx, node in enumerate(unique_ids)}

    return embeddings, unique_ids, id_to_index

RIBENA = False

# Duplicate keys in the dataset (wines that were duplicates and not annotated during the datacollection events as such)
duplicate_key_mapping = {
    26: 96,
    50: 10,
    5: 97
}

csv_data = pd.read_csv('data/wines_experiment.csv')
vintage_to_experiment_map = dict(zip(csv_data['vintage_id'], csv_data['experiment_wid']))

if not RIBENA:
    # Ribena ID
    value_to_remove = 67
    vintage_to_experiment_map = {k: v for k, v in vintage_to_experiment_map.items() if v != value_to_remove}

def preprocess_data_source1(data_source_napping, data_source_rounds, method):
    data_source_napping = remove_duplicate_ids(data_source_napping)
    data = combine_data_sources(data_source_napping, data_source_rounds)
    if method == "euclidean":
        distances = []
        for _, experiment_data in data.items():
            for _, round_data in experiment_data.items():
                for _, experiment_data in round_data.items():
                    point_distances = experiment_data['distances']
                    if len(point_distances) > 0:
                        point_distances = process_experiment_round(point_distances, method)
                        point_distances_normalized = normalize_values(point_distances)  
                        distances.append(point_distances_normalized)
        dist_matrix, unique_ids, id_to_index = create_distance_matrix(distances)
        return dist_matrix, unique_ids, id_to_index
    elif method == "triplets":
        triplets = []
        for _, experiment_data in data.items():
            for _, round_data in experiment_data.items():
                for _, experiment_data in round_data.items():
                    point_distances = experiment_data['distances']
                    triplet_batch = process_experiment_round(point_distances, method)
                    if not RIBENA and 67 in triplet_batch:
                        pass
                    else:
                        triplets.extend(triplet_batch)
        unique_triplet_ids = set()
        for triplet in triplets:
            unique_triplet_ids.update(triplet)
        if not RIBENA:
            unique_triplet_ids.remove(67)
        # sort unique_triplet_ids and create a mapping from old id to new id
        sorted_unique_triplet_ids = sorted(list(unique_triplet_ids))
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted_unique_triplet_ids)}
        # replace old ids with new ids in triplets
        for i, triplet in enumerate(triplets):
            if not RIBENA and 67 in triplet:
                pass
            else:
                triplets[i] = [id_mapping[old_id] for old_id in triplet]
        return triplets, sorted_unique_triplet_ids, None
    elif method == "graph":
        distances = []
        for _, experiment_data in data.items():
            for _, round_data in experiment_data.items():
                for _, experiment_data in round_data.items():
                    point_distances = experiment_data['distances']
                    if len(point_distances) > 0:
                        point_distances = process_experiment_round(point_distances, 'euclidean')
                        point_distances_normalized = normalize_values(point_distances)  
                        distances.append(point_distances)
        dist_matrix, unique_ids, id_to_index = create_distance_matrix(distances)
        G = nx.Graph()
        for i, id1 in enumerate(unique_ids):
            for j, id2 in enumerate(unique_ids):
                if i != j:
                    # Convert distance to similarity (you might adjust this conversion as needed)
                    similarity = 1 / (1 + dist_matrix[i][j])  
                    G.add_edge(id1, id2, weight=similarity)
        embeddings, unique_ids, id_to_index = reduce_graph_dimensionality(G)
        embedding_matrix = np.array([embeddings[node] for node in unique_ids])
        return embedding_matrix, unique_ids, id_to_index

def preprocess_data_source2(data):
    # Add a new column 'experiment_id' based on the provided mapping
    data['experiment_id'] = data['vintage_id'].map(vintage_to_experiment_map)
    # Replace the duplicate experiment_id values with the values from the duplicate_key_mapping
    data['experiment_id'].replace(duplicate_key_mapping, inplace=True)
    return data

def remove_duplicate_ids(data):
    new_data = {}
    for experiment_key, experiment_value in data['generated_data'].items():
        new_experiment = {}
        for round_key, round_value in experiment_value.items():
            new_round = {}
            for experiment_no_key, experiment_no_value in round_value.items():
                new_experiment_no = {}
                for original_key, value in experiment_no_value.items():
                    if int(original_key) in duplicate_key_mapping:
                        new_key = str(duplicate_key_mapping[int(original_key)])
                    else:
                        new_key = original_key
                    new_experiment_no[new_key] = value
                new_round[experiment_no_key] = new_experiment_no
            new_experiment[round_key] = new_round
        new_data[experiment_key] = new_experiment
    return new_data

def combine_data_sources(data_source1, data_source2):
    combined_data = {}
    for experiment in data_source1:
        combined_data[experiment] = {}
        for round_key in data_source1[experiment]:
            combined_data[experiment][round_key] = {}
            for experiment_no in data_source1[experiment][round_key]:
                # Create a new dictionary with 'round_id' and 'distances'
                combined_dict = {
                    "round_id": data_source2['generated_data'][experiment][round_key][experiment_no],
                    "distances": data_source1[experiment][round_key][experiment_no]
                }
                # Replace the original dictionary with the combined one
                combined_data[experiment][round_key][experiment_no] = combined_dict
    return combined_data

def normalize_values(dct):
    max_value = max(dct.values())
    normalized_dct = {key: value / max_value for key, value in dct.items()}
    return normalized_dct

def euclidean_distance(point1, point2):
    return math.sqrt((int(point1[0]) - int(point2[0])) ** 2 + (int(point1[1]) - int(point2[1])) ** 2)

def process_experiment_round(experiment_round_data, return_type):
    
    distances = {
        (id1, id2): euclidean_distance(coords1, coords2)
        for (id1, [coords1, _]), (id2, [coords2, _]) in combinations(experiment_round_data.items(), 2)
    }

    if return_type == "euclidean":
        return distances
    elif return_type == "triplets":
        triplets = [
            [int(id1), int(id2), int(id3)]
            for (id1, id2), dist_ij in distances.items()
            for (id3, _), dist_ik in distances.items()
            if id1 != id3 and dist_ij < dist_ik and id2 != id3
        ]
        return triplets


def create_distance_matrix(list_of_dicts):
    unique_ids = set()
    for d in list_of_dicts:
        for id_pair in d.keys():
            if not RIBENA:
                if '67' not in id_pair:
                    unique_ids.update(id_pair)
            else:
                unique_ids.update(id_pair)

    unique_ids = sorted(list(unique_ids))
    id_to_index = {id: index for index, id in enumerate(unique_ids)}

    n = len(unique_ids)
    distance_matrix = np.zeros((n, n))
    distance_lists = [[[] for _ in range(n)] for _ in range(n)]

    for d in list_of_dicts:
        for (id1, id2), distance in d.items():
            if not RIBENA:
                if id1 != '67' and id2 != '67':
                    index1, index2 = id_to_index[id1], id_to_index[id2]
                    distance_lists[index1][index2].append(distance)
                    distance_lists[index2][index1].append(distance)
            else:
                index1, index2 = id_to_index[id1], id_to_index[id2]
                distance_lists[index1][index2].append(distance)
                distance_lists[index2][index1].append(distance)

    for i in range(n):
        for j in range(i+1, n):
            distances = distance_lists[i][j]
            if distances:
                avg_distance = np.mean(distances)
                distance_matrix[i, j] = avg_distance
                distance_matrix[j, i] = avg_distance

    return distance_matrix, unique_ids, id_to_index
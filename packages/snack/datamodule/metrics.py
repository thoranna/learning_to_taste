import pickle
import numpy as np
import seaborn as sns
import torch
from torchmetrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

class Metrics():
    def __init__(self) -> None:
        pass
        
    def calculate_triplet_agreement(self, trips, trips_classes):
        """
        Description: Given the clicked triplets and their corresponding classes
        compute the number of positives that match anchor class and those that should have been anchor class
        :param trips:
        :param trips_classes:
        :return:
        """
        # how many triplets are in agreement with the best possible scenario
        should_be_anchor_class = np.sum(trips_classes[trips[:, 1]] == trips_classes[trips[:, 0]], axis=0)
        #should_not_be_anchor_class = np.sum(trips_classes[trips[:, 2]] != trips_classes[trips[:, 0]], axis=0)
        agreements = should_be_anchor_class #+ should_not_be_anchor_class
        was_not_be_anchor_class_but_should = np.sum(trips_classes[trips[:, 1]] != trips_classes[trips[:, 0]], axis=0)
        #was_anchor_class_but_should_not = np.sum(trips_classes[trips[:, 2]] == trips_classes[trips[:, 0]], axis=0)
        disagree = was_not_be_anchor_class_but_should #+ was_anchor_class_but_should_not
        return agreements, disagree

    def noise_to_signal_ratio(self, query_emb, ref_emb):
        # how much noise is in the syste, / can we measure
        query_var = np.var(query_emb, axis=0)
        query_ref_var = np.var(query_emb - ref_emb, axis=0)
        return np.sum(query_ref_var / query_var)

    def precision_recall_curve(self, preds, targets, num_classes=30):
        """
        Description: Given the clicked predictions of thoese that is similar 
        to the anchor class and the comments actual class, compute the precision 
        recall curve for a given parameter configuration.
        :param preds:
        :param targets:
        :param num_classes:
        :return:
        """
        pres = self.calculate_precision(preds, targets, num_classes)
        reca = self.calculate_recall(preds, targets, num_classes)
        return pres,reca
    
    def calculate_precision(self, preds, targets, num_classes=30):
        precision = Precision(average='none', num_classes=num_classes)
        return precision(preds, targets)

    def calculate_recall(self, preds, targets, num_classes=30):
        recall = Recall(average='none', num_classes=num_classes)
        return recall(preds, targets)

    def triplet_error(self, emb, trips):
        """
        Description: Given the embeddings and triplet constraints, compute the triplet error.
        :param emb:
        :param trips:
        :return:
        """
        d1 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 1], :]) ** 2, axis=1)
        d2 = np.sum((emb[trips[:, 0], :] - emb[trips[:, 2], :]) ** 2, axis=1)
        error_list = d2 > d1
        ratio = sum(error_list) / trips.shape[0]
        return ratio, error_list

    def knn_classification_error(self, true_emb, labels, args):
        # test how well local structure is preserved
        """
        Description: Compute the kNN classification error on a test set (70/30 split)  trained on
        the ground-truth embedding and the ordinal embedding.
        :param true_emb: ground-truth embedding
        :param labels: labels of the data points
        :return classification error on test data on ordinal embedding and ground-truth embedding.
        """
        if args.should_replace_label:
            idx = np.array(labels)!=args.replace_label
            labels = list(np.array(labels)[idx])
            true_emb = true_emb[idx]
        
        n_neighbors = int(np.log2(true_emb.shape[0]))

        x_train, x_test, y_train, y_test = train_test_split(true_emb, labels, train_size=0.7)
        # n_neighbors = {'n_neighbors':[1, 3, 5, 10, 15, 20, 25]}

        original_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)

        # ordinal_classifier = grid_search(kNN(), param_grid=n_neighbors, cv=3)
        # original_classifier = grid_search(kNN(), param_grid=n_neighbors, cv=3)
        original_classifier.fit(x_train, y_train)
        return original_classifier.score(x_test, y_test)

def post_hoc_metric_given_embedding_and_triplets(embedding, trips, labels, gt_triplets, args):
    pbm = Metrics()
    ratio, error_list = pbm.triplet_error(embedding, gt_triplets)
    knn_error = pbm.knn_classification_error(embedding, labels, args)
    index_to_class = np.array(labels)
    agreements, disagree = pbm.calculate_triplet_agreement(trips, index_to_class)
    n_classes = len(set(labels))
    snr = pbm.noise_to_signal_ratio(embedding[trips[:, 1]], embedding[trips[:, 0]])
    precision, recall =pbm.precision_recall_curve(torch.tensor(index_to_class[trips[:, 0]]),torch.tensor(index_to_class[trips[:, 1]]),n_classes)
    return [ratio, knn_error, agreements, disagree, snr, precision, recall]
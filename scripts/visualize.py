import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.preprocessing import LabelEncoder
from .prediction import DICTIONARIES

def visualize_embeddings(data1_embedding, data2_embedding, combined_embedding, data1_experiment_ids, data2_experiment_ids, common_experiment_ids, num, methods):
    data_sources = {
        "Human Annotation Embeddings": (data1_embedding, data1_experiment_ids, methods['data1']),
        "Review Embeddings": (data2_embedding, data2_experiment_ids, methods['data2']),
        "Combined Embeddings": (combined_embedding, common_experiment_ids, methods['combined'])
    }

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('results/images'):
        os.makedirs('results/images')

    le = LabelEncoder()
    
    for dk_idx, (dk, d) in enumerate(DICTIONARIES.items()):
        all_labels = []
        for i, (name, (data_source, ids, method)) in enumerate(data_sources.items()):
            labels = [d[int(key)] for key in ids]
            all_labels.extend(labels)

        all_labels = [int(label) if label.isdigit() else label for label in all_labels]
        le.fit(all_labels)

        unique_labels = list(le.classes_)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle(f"{dk}", fontsize=14)

        for i, (name, (data_source, ids, method)) in enumerate(data_sources.items()):
            labels = [d[int(key)] for key in ids]
            labels = [int(label) if label.isdigit() else label for label in labels]
            labels = le.transform(labels)

            if dk in ["Alcohol %", "Year", "Price", "Rating"]:
                sc = axes[i].scatter(data_source[:, 0], data_source[:, 1], c=labels, cmap='viridis', marker='o', s=50)
                if i == len(data_sources) - 1:  # Only add colorbar for last subplot
                    cax = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # [left, bottom, width, height]
                    cbar = plt.colorbar(sc, cax=cax, orientation='vertical')
                    cbar.ax.set_ylabel(dk)
                    cbar.set_ticks(np.arange(len(unique_labels)))
                    cbar.set_ticklabels(unique_labels)
            else:
                cmap = plt.cm.get_cmap('tab10', len(unique_labels))
                for label_idx, label in enumerate(unique_labels):
                    label_mask = np.array(labels) == label_idx
                    axes[i].scatter(data_source[label_mask, 0], data_source[label_mask, 1], color=cmap(label_idx), marker='o', s=50)

            axes[i].set_title(f"{name} - {method}")
            axes[i].set_xlabel('Component 1')
            axes[i].set_ylabel('Component 2')

            # Add legend for categorical values on each subplot
            if dk not in ["Alcohol %", "Year", "Price", "Region", "Grape", "Rating"]:
                handles = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(unique_labels)]
                if dk == "Country":  # Check if the category is 'Country'
                    # axes[i].legend(handles=handles, loc='upper right')
                    pass
                else:
                    axes[i].legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)

        plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust this value to provide space for super title and legend

        # Add legend for categorical values outside the subplots
        if dk not in ["Alcohol %", "Year", "Price", "Region", "Grape", "Rating"]:
            handles = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(unique_labels)]
            fig.legend(handles=handles, loc='center right', bbox_to_anchor=(0.95, 0.5))

        if not os.path.exists(f'results/images/image{num}'):
            os.makedirs(f'results/images/image{num}')
        plt.savefig(f'results/images/image{num}/image{dk}.png', bbox_inches='tight')
        plt.close()
import scripts.data_source1_utils as data_source1_utils
import scripts.data_source2_utils as data_source2_utils
import scripts.preprocessing as preprocessing
import scripts.dimensionality_reduction as dimensionality_reduction
import scripts.model_fitting_new as model_fitting
import scripts.visualize as visualize
import scripts.data_combination as data_combination
import scripts.prediction as prediction
import scripts.fit_clip as fit_clip
import csv
# Imports for resource allocations
import time
import psutil
import os

start = time.time()
pid = os.getpid()
py = psutil.Process(pid)
memory_use_before = py.memory_info()[0] / 2. ** 30  # Memory use in GB

def create_csv_row(
        preprocessing_method1,
        model,
        dimensionality_reduction_method2,
        data_combination_method,
        predictions,
        increment_report,
        averages,
        time,
        memory,
        cpu_p
    ):
    row = [preprocessing_method1, model, dimensionality_reduction_method2, data_combination_method]
    row.extend([predictions["single"]["data1"]["Country"]["accuracy"], predictions["single"]["data2"]["Country"]["accuracy"], predictions["single"]["combined"]["Country"]["accuracy"],
                predictions["single"]["data1"]["Region"]["accuracy"], predictions["single"]["data2"]["Region"]["accuracy"], predictions["single"]["combined"]["Region"]["accuracy"],
                predictions["single"]["data1"]["Grape"]["accuracy"], predictions["single"]["data2"]["Grape"]["accuracy"], predictions["single"]["combined"]["Grape"]["accuracy"],
                predictions["single"]["data1"]["Alcohol %"]["accuracy"], predictions["single"]["data2"]["Alcohol %"]["accuracy"], predictions["single"]["combined"]["Alcohol %"]["accuracy"],
                predictions["single"]["data1"]["Year"]["accuracy"], predictions["single"]["data2"]["Year"]["accuracy"], predictions["single"]["combined"]["Year"]["accuracy"],
                predictions["single"]["data1"]["Price"]["accuracy"], predictions["single"]["data2"]["Price"]["accuracy"], predictions["single"]["combined"]["Price"]["accuracy"],
                predictions["single"]["data1"]["Rating"]["accuracy"], predictions["single"]["data2"]["Rating"]["accuracy"], predictions["single"]["combined"]["Rating"]["accuracy"],
                predictions["multi"]["data1"]["accuracy"], predictions["multi"]["data2"]["accuracy"], predictions["multi"]["combined"]["accuracy"],
                predictions["multi"]["data1"]["f1_score"], predictions["multi"]["data2"]["f1_score"], predictions["multi"]["combined"]["f1_score"]])
    row.extend([increment_report["single"]["accuracy"]["increment"], increment_report["single"]["f1_score"]["increment"], increment_report["multi"]["accuracy"]["increment"], increment_report["multi"]["f1_score"]["increment"]])
    row.extend([averages["accuracy"], averages["f1_score"], averages["f1_score_d2"]])
    row.extend([time, memory, cpu_p])
    return row

# Define the combinations of methods to try
preprocessing_methods1 = ['euclidean',
                          'triplets']
dimensionality_reduction_methods2 = ['TSNE', 'PCA', 'Umap']
data_combination_methods = ['CCA', 'ICP', 'SNaCK']

# TEXT: 
models_to_use_text = ['distil_bert', 't5_small', 'albert', 'bart', 'clip_text']
# IMAGES: 
models_to_use_images = ['vit_base', 'deit_small', 'resnet', 'clip_image']
# IMAGES AND TEXT: 
models_to_use_multi = ['clip']
models_to_use = models_to_use_images + models_to_use_text + models_to_use_multi

# Load data
data_source1a, data_source1b = data_source1_utils.load_data()
data_source2 = data_source2_utils.load_data()

csv_header = ["preprocessing_method1", "model", "dimensionality_reduction_method2", "data_combination_method",
              "WINE_COUNTRY_data1", "WINE_COUNTRY_data2", "WINE_COUNTRY_combined",
              "WINE_REGION_data1", "WINE_REGION_data2", "WINE_REGION_combined",
              "WINE_GRAPE_data1", "WINE_GRAPE_data2", "WINE_GRAPE_combined",
              "WINE_ALCOHOL_PERCENTAGE_data1", "WINE_ALCOHOL_PERCENTAGE_data2", "WINE_ALCOHOL_PERCENTAGE_combined",
              "WINE_YEAR_data1", "WINE_YEAR_data2", "WINE_YEAR_combined",
              "WINE_PRICE_data1", "WINE_PRICE_data2", "WINE_PRICE_combined",
              "WINE_RATING_data1", "WINE_RATING_data2", "WINE_RATING_combined",
              "MULTI_LABEL_data1", "MULTI_LABEL_data2", "MULTI_LABEL_combined",
              "MULTI_LABEL_F1_SCORE_data1", "MULTI_LABEL_F1_SCORE_data2", "MULTI_LABEL_F1_SCORE_combined",
              "SINGLE_ACCURACY_IMPROVEMENT_IDX", "SINGLE_F1_IMPROVEMENT_IDX",
              "MULTI_ACCURACY_IMPROVEMENT_IDX", "MULTI_F1_IMPROVEMENT_IDX",
              "AVERAGE_COMBINED_ACCURACY", "AVERAGE_COMBINED_F1", "AVERAGE_D2_F1", 
              "time", "memory", "cpu_percent"]

csv_file_path = "results/results1000.csv"
#with open(csv_file_path, "w", newline="") as csvfile:
#    csv_writer = csv.writer(csvfile)
#    csv_writer.writerow(csv_header)

csv_file_path_rand = "results/results_random1000.csv"
#with open(csv_file_path_rand, "w", newline="") as csvfile:
#    csv_writer = csv.writer(csvfile)
#    csv_writer.writerow(csv_header)

# Iterate over combinations of methods
num = 0
for preprocessing_method1 in preprocessing_methods1:
    _start = time.time()
    _pid = os.getpid()
    _py = psutil.Process(_pid)
    _memory_use_before = _py.memory_info()[0] / 2. ** 30  # Memory use in GB
    print("preprocessing_method1: ", preprocessing_method1)
    if preprocessing_method1 == 'euclidean':
        preprocessed_data1, unique_ids1, index_to_id = preprocessing.preprocess_data_source1(data_source1a, data_source1b, method=preprocessing_method1)
        unique_ids1 = [int(item) for item in  unique_ids1]
        dimensionality_reduction_method1 = 'MDS'
        reduced_embeddings1 = dimensionality_reduction.reduce_data1(preprocessed_data1, method=dimensionality_reduction_method1, ids=unique_ids1)
    elif preprocessing_method1 == 'triplets':
        preprocessed_data1, unique_ids1, _ = preprocessing.preprocess_data_source1(data_source1a, data_source1b, method=preprocessing_method1)
        dimensionality_reduction_method1 = 't-STE'
        reduced_embeddings1 = dimensionality_reduction.reduce_data1(preprocessed_data1, method=dimensionality_reduction_method1, ids=unique_ids1)
    preprocessed_data2 = preprocessing.preprocess_data_source2(data_source2)
    for model_to_use in models_to_use:
        if model_to_use == 'clip':
            embedding_matrix2, unique_ids2 = fit_clip.fit_model(model_to_use, preprocessed_data2)
        else:
            embedding_matrix2, unique_ids2 = model_fitting.fit_model(model_to_use, preprocessed_data2)
        for dimensionality_reduction_method2 in dimensionality_reduction_methods2:
            print("dimensionality_reduction_method2: ", dimensionality_reduction_method2)
            reduced_embeddings2 = dimensionality_reduction.reduce_data2(embedding_matrix2, dimensionality_reduction_method2, unique_ids2)
            for data_combination_method in data_combination_methods:
                print("data combination method: ", data_combination_method)
                methods = {"data1": dimensionality_reduction_method1,
                           "data2": dimensionality_reduction_method2,
                           "combined": data_combination_method}
                # Only run SNaCK on triplets and data reduced by TSNE
                if data_combination_method == "SNaCK":
                    if preprocessing_method1 == "triplets" and dimensionality_reduction_method2 == "TSNE":
                        combined_embeddings, common_experiment_ids, _, _, _, _ = data_combination.combine_data(preprocessed_data1, embedding_matrix2, unique_ids1, unique_ids2, data_combination_method, preprocessing_method1)
                        combined_embeddings = combined_embeddings.detach().numpy()
                        predictions, random_predictions, increment_report, random_increment_report, averages = prediction.predict_classes(
                            reduced_embeddings1,
                            reduced_embeddings2,
                            combined_embeddings,
                            unique_ids1,
                            unique_ids2,
                            common_experiment_ids
                        )
                        visualize.visualize_embeddings(reduced_embeddings1,
                                                       reduced_embeddings2,
                                                       combined_embeddings,
                                                       unique_ids1,
                                                       unique_ids2,
                                                       common_experiment_ids,
                                                       num=num,
                                                       methods=methods)
                        # Write the results to the CSV file
                        _end = time.time()
                        _memory_use_after = py.memory_info()[0] / 2. ** 30  # Memory use in GB
                        _cpu_percent = py.cpu_percent(interval=None)
                        _time = _end - _start
                        _memory = _memory_use_after - _memory_use_before
                        csv_row = create_csv_row(preprocessing_method1,
                                                 model_to_use,
                                                 dimensionality_reduction_method2,
                                                 data_combination_method,
                                                 predictions,
                                                 increment_report,
                                                 averages["real"],
                                                 _time,
                                                 _memory,
                                                 _cpu_percent)
                        with open(csv_file_path, "a", newline="") as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow(csv_row)
                        csv_row = create_csv_row(preprocessing_method1,
                                                 model_to_use,
                                                 dimensionality_reduction_method2,
                                                 data_combination_method,
                                                 random_predictions,
                                                 random_increment_report,
                                                 averages["shuffled"],
                                                 _time,
                                                 _memory,
                                                 _cpu_percent)
                        with open(csv_file_path_rand, "a", newline="") as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow(csv_row)
                        num += 1
                    else:
                        continue
                else:
                    combined_embeddings, common_experiment_ids, data1, labels1, data2, labels2 = data_combination.combine_data(reduced_embeddings1,
                                                                                               reduced_embeddings2,
                                                                                               unique_ids1,
                                                                                               unique_ids2,
                                                                                               data_combination_method,
                                                                                               preprocessing_method1)
                    predictions, random_predictions, increment_report, random_increment_report, averages = prediction.predict_classes(
                        reduced_embeddings1,
                        reduced_embeddings2,
                        combined_embeddings,
                        unique_ids1,
                        unique_ids2,
                        common_experiment_ids
                    )
                    visualize.visualize_embeddings(reduced_embeddings1,
                                                   reduced_embeddings2,
                                                   combined_embeddings,
                                                   unique_ids1,
                                                   unique_ids2,
                                                   common_experiment_ids,
                                                   num=num,
                                                   methods=methods)
                    print("predictions: ", predictions)
                    # Write the results to the CSV file
                    _end = time.time()
                    _memory_use_after = py.memory_info()[0] / 2. ** 30  # Memory use in GB
                    _cpu_percent = py.cpu_percent(interval=None)
                    _time = _end - _start
                    _memory = _memory_use_after - _memory_use_before
                    csv_row = create_csv_row(preprocessing_method1,
                                             model_to_use,
                                             dimensionality_reduction_method2,
                                             data_combination_method,
                                             predictions,
                                             increment_report,
                                             averages["real"],
                                             _time,
                                             _memory,
                                             _cpu_percent)
                    with open(csv_file_path, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(csv_row)
                    csv_row = create_csv_row(preprocessing_method1,
                                                 model_to_use,
                                                 dimensionality_reduction_method2,
                                                 data_combination_method,
                                                 random_predictions,
                                                 random_increment_report,
                                                 averages["shuffled"],
                                                 _time,
                                                 _memory,
                                                 _cpu_percent)
                    with open(csv_file_path_rand, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(csv_row)
                    num += 1

end = time.time()
memory_use_after = py.memory_info()[0] / 2. ** 30  # Memory use in GB
cpu_percent = py.cpu_percent(interval=None)

print(f"Execution time: {end - start} seconds")
print(f"Memory used: {memory_use_after - memory_use_before} GB")
print(f"CPU percent used: {cpu_percent}%")
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from collections import Counter
from sklearn.dummy import DummyClassifier
import numpy as np
from utils.set_seed import RANDOM_SEED

RIBENA = False

def single_label(data_embedding, experiment_ids, d):
    labels = [d[int(key)] for key in experiment_ids]
    label_counts = Counter(labels)
    labels, ids, data_source = zip(*[(label, id, vector) for label, id, vector in zip(labels, experiment_ids, data_embedding) if label_counts[label] >= 1])

    labels = list(labels)
    ids = list(ids)
    data_source = np.array(data_source)

    if isinstance(labels[0], str):
        le = LabelEncoder()
        labels = le.fit_transform(labels)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    shuffled_scores = []
    f1_scores = []
    shuffled_f1_scores = []
    for train_index, test_index in kf.split(data_source, labels):
        X_train, X_test = data_source[train_index], data_source[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        ru = RandomOverSampler(sampling_strategy='not majority', random_state=RANDOM_SEED)
        X_res, y_res = ru.fit_resample(X_train, y_train)

        clf = SVC(random_state=42, class_weight='balanced')
        clf.fit(X_res, y_res)
        score = round(clf.score(X_test, y_test), 2)
        scores.append(score)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='micro')
        f1_scores.append(f1)
        
        # Establishing a dummy classifier as a baseline
        dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
        dummy_clf.fit(X_res, y_res)
        y_pred = dummy_clf.predict(X_test)
        shuffled_score = round(dummy_clf.score(X_test, y_test), 2)
        shuffled_scores.append(shuffled_score)
        shuffled_f1 = f1_score(y_test, y_pred, average='micro')
        shuffled_scores.append(shuffled_score)
        shuffled_f1_scores.append(shuffled_f1)

    avg_score = round(sum(scores) / len(scores), 2)
    avg_f1_score = round(sum(f1_scores) / len(f1_scores), 2)
    avg_shuffled_score = round(sum(shuffled_scores) / len(shuffled_scores), 2)
    avg_shuffled_f1_score = round(sum(shuffled_f1_scores) / len(shuffled_f1_scores), 2)
    return avg_score, avg_f1_score, avg_shuffled_score, avg_shuffled_f1_score

def get_multi_labels(dictionaries, ids):
    multi_labels = []
    for id in ids:
        labels = []
        for dictionary in dictionaries.values():
            labels.append(dictionary[int(id)])
        multi_labels.append(labels)
    return multi_labels

def multi_label(data_embedding, experiment_ids, dictionaries):
    labels = get_multi_labels(dictionaries, experiment_ids)
    mlb = MultiLabelBinarizer()
    transformed_labels = mlb.fit_transform(labels)
    print(transformed_labels)
    print(experiment_ids)
    print(data_embedding)
    
    filtered_data = [(label, id, vector) for label, id, vector in zip(transformed_labels, experiment_ids, data_embedding)]
    labels, _, data_source = zip(*filtered_data)
    
    data_source = np.array(data_source)
    labels = np.array(labels)
    
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    scores = []
    shuffled_scores = []
    f1_scores = []
    shuffled_f1_scores = []
    for train_index, test_index in kf.split(data_source):
        X_train, X_test = data_source[train_index], data_source[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        clf = OneVsRestClassifier(SVC(random_state=RANDOM_SEED, class_weight='balanced'))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = round(clf.score(X_test, y_test), 2)
        f1 = f1_score(y_test, y_pred, average='samples')
        scores.append(score)
        f1_scores.append(f1)
        
        # Create a dummy classifier as a baseline
        dummy_clf = DummyClassifier(strategy="stratified", random_state=RANDOM_SEED)
        dummy_clf.fit(X_train, y_train)
        y_pred = dummy_clf.predict(X_test)
        shuffled_score = round(dummy_clf.score(X_test, y_test), 2)
        shuffled_f1 = f1_score(y_test, y_pred, average='samples')
        shuffled_scores.append(shuffled_score)
        shuffled_f1_scores.append(shuffled_f1)

    avg_score = round(sum(scores) / len(scores), 2)
    avg_f1_score = round(sum(f1_scores) / len(f1_scores), 2)
    avg_shuffled_score = round(sum(shuffled_scores) / len(shuffled_scores), 2)
    avg_shuffled_f1_score = round(sum(shuffled_f1_scores) / len(shuffled_f1_scores), 2)
    
    return avg_score, avg_f1_score, avg_shuffled_score, avg_shuffled_f1_score


def predict_classes(data1_embedding, data2_embedding, combined_embedding, data1_experiment_ids, data2_experiment_ids, common_experiment_ids):
    report = {"single": {}, "multi": {}}
    shuffled_report = {"single": {}, "multi": {}}
    increment_report = {"single": {"accuracy": {}, "f1_score": {}}, "multi": {"accuracy": {}, "f1_score": {}}}
    shuffled_increment_report = {"single": {"accuracy": {}, "f1_score": {}}, "multi": {"accuracy": {}, "f1_score": {}}}
    data_sources_ids = {"data1": (data1_embedding, data1_experiment_ids),
                        "data2": (data2_embedding, data2_experiment_ids),
                        "combined": (combined_embedding, common_experiment_ids)}
    
    def single(name, data_source, ids, d, k, report, shuffled_report):
        avg_score, avg_f1_score, avg_shuffled_score, avg_shuffled_f1_score = single_label(data_source, ids, d)
        # Ensure report[name] exists and is a dictionary
        if name not in report:
            report[name] = {}
        if name not in shuffled_report:
            shuffled_report[name] = {}
        report[name][k] = {"accuracy": round(avg_score*100, 2), "f1_score": avg_f1_score}
        shuffled_report[name][k] = {"accuracy": round(avg_shuffled_score*100, 2), "f1_score": avg_shuffled_f1_score}
    
    def multi(name, data_source, ids, d, report, shuffled_report):
        avg_score, avg_f1_score, avg_shuffled_score, avg_shuffled_f1_score = multi_label(data_source, ids, d)
        report[name] = {"accuracy": round(avg_score*100, 2), "f1_score": avg_f1_score}
        shuffled_report[name] = {"accuracy": round(avg_shuffled_score*100, 2), "f1_score": avg_shuffled_f1_score}

    base_acc_1 = base_acc_2 = base_shuffled_acc_1 = base_shuffled_acc_2 = increment_acc_1 = shuffled_increment_acc_1 = 0
    base_f1_1 = base_f1_2 = base_shuffled_f1_1 = base_shuffled_f1_2  = increment_f1_1 = shuffled_increment_f1_1 = 0

    for k, d in DICTIONARIES.items():
        for name, (data_source, ids) in data_sources_ids.items():
            single(name, data_source, ids, d, k, report["single"], shuffled_report["single"])
            multi(name, data_source, ids, DICTIONARIES, report["multi"], shuffled_report["multi"])

    avg_combined_acc = 0
    avg_shuffled_combined_acc = 0

    avg_combined_f1 = 0
    avg_data2_f1 = 0
    avg_shuffled_combined_f1 = 0
    avg_shuffled_data2_f1 = 0
    for k, d in DICTIONARIES.items():
        # Compute incrementation for method 1 (single-label)
        base_acc_1 = max(report["single"]["data1"][k]["accuracy"], report["single"]["data2"][k]["accuracy"])
        base_shuffled_acc_1 = max(shuffled_report["single"]["data1"][k]["accuracy"], shuffled_report["single"]["data2"][k]["accuracy"])

        increment_acc_1 += report["single"]["combined"][k]["accuracy"] - base_acc_1
        shuffled_increment_acc_1 += shuffled_report["single"]["combined"][k]["accuracy"] - base_shuffled_acc_1

        avg_combined_acc += report["single"]["combined"][k]["accuracy"]
        avg_shuffled_combined_acc += shuffled_report["single"]["combined"][k]["accuracy"]

        # Compute incrementation for method 1 (single-label)
        base_f1_1 += max(report["single"]["data1"][k]["f1_score"], report["single"]["data2"][k]["f1_score"])
        base_shuffled_f1_1 += max(shuffled_report["single"]["data1"][k]["f1_score"], shuffled_report["single"]["data2"][k]["f1_score"])

        increment_f1_1 += report["single"]["combined"][k]["f1_score"] - base_f1_1
        shuffled_increment_f1_1 += shuffled_report["single"]["combined"][k]["f1_score"] - base_shuffled_f1_1

        avg_combined_acc += report["single"]["combined"][k]["f1_score"]
        avg_shuffled_combined_acc += shuffled_report["single"]["combined"][k]["f1_score"]

        avg_combined_f1 += report["single"]["combined"][k]["f1_score"]
        avg_shuffled_combined_f1 += shuffled_report["single"]["combined"][k]["f1_score"]

        avg_data2_f1 += report["single"]["data2"][k]["f1_score"]
        avg_shuffled_data2_f1 += shuffled_report["single"]["data2"][k]["f1_score"]
    
    for metric in ["f1_score", "accuracy"]:
        increment_report["single"][metric] = {"increment": round(increment_acc_1 * 100, 2)}
        shuffled_increment_report["single"][metric] = {"increment": round(shuffled_increment_acc_1 * 100, 2)}

    avg_combined_acc /= len(list(DICTIONARIES.keys()))
    avg_shuffled_combined_acc /= len(list(DICTIONARIES.keys()))
    avg_combined_f1 /= len(list(DICTIONARIES.keys()))
    avg_shuffled_combined_f1 /= len(list(DICTIONARIES.keys()))
    avg_data2_f1 /= len(list(DICTIONARIES.keys()))
    avg_shuffled_data2_f1 /= len(list(DICTIONARIES.keys()))

    # Compute incrementation for method 2 (multi-label)
    for metric in ["f1_score", "accuracy"]:
        base_acc_2 = max(report["multi"]["data1"][metric], report["multi"]["data2"][metric])
        base_shuffled_acc_2 = max(shuffled_report["multi"]["data1"][metric], shuffled_report["multi"]["data2"][metric])

        increment_acc_2 = report["multi"]["combined"][metric] - base_acc_2
        shuffled_increment_acc_2 = shuffled_report["multi"]["combined"][metric] - base_shuffled_acc_2
        
        if metric != "f1_score":
            increment_report["multi"][metric] = {"increment": round(increment_acc_2 * 100, 2)}
            shuffled_increment_report["multi"][metric] = {"increment": round(shuffled_increment_acc_2 * 100, 2)}
        else:
            increment_report["multi"][metric] = {"increment":increment_acc_2}
            shuffled_increment_report["multi"][metric] = {"increment": shuffled_increment_acc_2}
    print(increment_report['multi'])
    avg_score_report = {"real": {"accuracy": avg_combined_acc,
                                 "f1_score": avg_combined_f1,
                                 "f1_score_d2": avg_data2_f1},
                        "shuffled": {"accuracy": avg_shuffled_combined_acc,
                                     "f1_score": avg_shuffled_combined_f1,
                                     "f1_score_d2": avg_shuffled_data2_f1}}
    print("Increment report: ", increment_report)
    print("Shuffled increment report: ", shuffled_increment_report)
    print("Report: ", report)
    print("Shuffled report: ", shuffled_report)
    print("Incrementation report: ", increment_report)
    print("Shuffled incrementation report: ", shuffled_increment_report)
    return report, shuffled_report, increment_report, shuffled_increment_report, avg_score_report


# Duplicate keys in the dataset (wines that were duplicates and not annotated during the datacollection events as such)
duplicate_key_mapping = {
    26: 96,
    50: 10,
    5: 97
}

def replace_ids_in_dict(input_dict, duplicate_key_mapping):
    return {duplicate_key_mapping.get(key, key): value for key, value in input_dict.items()}

WINE_COUNTRY = {0: 'France',
               1: 'Italy',
               2: 'Australia',
               3: 'Italy',
               4: 'Italy',
               5: 'France',
               6: 'Italy',
               7: 'Italy',
               8: 'Italy',
               9: 'United States',
               10: "Italy",
               11: "Italy",
               12: 'Australia',
               13: 'Spain',
               14: 'Italy',
               15: 'Spain',
               16: 'Italy',
               17: 'Italy',
                18: 'Italy',
                19: 'Spain',
                20: 'Spain',
                21: 'Spain',
                22: 'Portugal',
                23: 'Australia',
                24: 'Spain',
                25: 'France',
                26: 'France',
                27: 'Argentina',
                28: 'Australia',
                29: 'Spain',
                30: 'Argentina',
                31: 'Italy',
                32: 'France',
                33: 'France',
                34: 'Spain',
                35: 'Italy',
                36: 'Italy',
                37: 'Italy',
                38: 'Italy',
                39: 'Italy',
                40: 'Italy',
                41: 'United States',
                42: 'Portugal',
                43: 'Portugal',
                44: 'Italy',
                45: 'Italy',
                46: 'Italy',
                47: 'Italy',
                48: 'Italy',
                49: 'Spain',
                50: 'France',
                51: 'Spain',
                52: 'Italy',
                53: 'France',
                54: 'Italy',
                55: 'Italy',
                56: 'Italy',
                57: 'Italy',
                58: 'Italy',
                59: 'Italy',
                60: 'Spain',
                61: 'Italy',
                62: 'France',
                63: 'Italy',
                64: 'Argentina',
                65: 'Australia',
                66: 'Italy',
                67: 'Ribena',
                68: 'Spain',
                69: 'Spain',
                70: 'Italy',
                71: 'Italy',
                72: 'France',
                73: 'Italy',
                74: 'Italy',
                75: 'Italy',
                76: 'Spain',
                77: 'Italy',
                78: 'Italy',
                79: 'France',
                80: 'France',
                81: 'Italy',
                82: 'Spain',
                83: 'Spain',
                84: 'France',
                85: 'United States',
                86: 'Spain',
                87: 'Portugal',
                88: 'South Africa',
                89: 'France',
                90: 'Spain',
                91: 'Spain',
                92: 'Spain',
                93: 'Italy',
                94: 'United States',
                95: 'Italy',
                96: 'France',
                97: 'France',
                98: 'Italy',
                99: 'France',
                100: 'Spain',
                101: 'France',
                102: 'France',
                103: 'Italy',
                104: 'Italy',
                105: 'France',
                106: 'Spain',
                107: 'France',
                108: 'Italy',
                109: 'Italy',
                110: 'Italy',
                111: 'Italy',
                112: 'Italy',
                113: 'France',
                114: 'Spain'
                }

WINE_LOC_SPEC = {
    0: 'Saint-Émilion Grand Cru',
    1: 'Northern Italy',
    2: 'South Australia',
    3: 'Southern Italy',
    4: 'Central Italy',
    5: 'Bordeaux',
    6: 'Central Italy',
    7: 'Northern Italy',
    8: 'Southern Italy',
    9: 'California',
    10: 'Northern Italy',
    11: 'Central Italy',
    12: 'South Australia',
    13: 'Vino de España',
    14: 'Central Italy',
    15: 'Castilla y León',
    16: 'Southern Italy',
    17: 'Southern Italy',
    18: 'Southern Italy',
    19: 'Vino de España',
    20: 'Castilla y León',
    21: 'Rioja',
    22: 'Alentejano',
    23: 'South Australia',
    24: 'Murcia',
    25: 'Languedoc-Roussillon',
    26: 'Bordeaux',
    27: 'Mendoza',
    28: 'South Australia',
    29: 'Vino de España',
    30: 'Mendoza',
    31: 'Southern Italy',
    32: 'Rhone Valley',
    33: 'Bordeaux',
    34: 'Catalunya',
    35: 'Central Italy',
    36: 'Central Italy',
    37: 'Central Italy',
    38: 'Central Italy',
    39: 'Southern Italy',
    40: 'Central Italy',
    41: 'California',
    42: 'Península de Setúbal',
    43: 'Northern Portugal',
    44: 'Central Italy',
    45: 'Central Italy',
    46: 'Central Italy',
    47: 'Southern Italy',
    48: 'Northern Italy',
    49: 'Islas Baleares',
    50: 'Bordeaux',
    51: 'Castilla y León',
    52: 'Northern Italy',
    53: 'Bourgogne',
    54: 'Northern Italy',
    55: 'Northern Italy',
    56: 'Southern Italy',
    57: 'Central Italy',
    58: 'Southern Italy',
    59: 'Southern Italy',
    60: 'Rioja',
    61: "Vino d'Italia",
    62: 'Bordeaux',
    63: 'Northern Italy',
    64: 'Salta',
    65: 'South Australia',
    66: 'Southern Italy',
    67: 'Ribena',
    68: 'Castilla y León',
    69: 'Castilla y León',
    70: 'Southern Italy',
    71: 'Central Italy',
    72: 'Bordeaux',
    73: 'Central Italy',
    74: 'Northern Italy',
    75: 'Central Italy',
    76: 'Castilla y León',
    77: 'Central Italy',
    78: 'Southern Italy',
    79: 'Vin de France',
    80: 'Bordeaux',
    81: 'Southern Italy',
    82: 'Aragón',
    83: 'Rioja',
    84: 'Rhone Valley',
    85: 'California',
    86: 'Vino de España',
    87: 'Península de Setúbal',
    88: 'Western Cape',
    89: 'Bordeaux',
    90: 'Castilla y León',
    91: 'Castilla y León',
    92: 'Valencia',
    93: 'Northern Italy',
    94: 'California',
    95: 'Southern Italy',
    96: 'Bordeaux',
    97: 'Bordeaux',
    98: 'Southern Italy',
    99: 'Bourgogne',
    100: 'Castilla y León',
    101: 'Bordeaux',
    102: 'Bordeaux',
    103: 'Southern Italy',
    104: 'Central Italy',
    105: 'Bordeaux',
    106: 'Rioja',
    107: 'Bordeaux',
    108: 'Southern Italy',
    109: 'Southern Italy',
    110: 'Southern Italy',
    111: 'Southern Italy',
    112: 'Central Italy',
    113: 'Bordeaux',
    114: 'Catalunya'}

WINE_STYLE = {0: 'Bordeaux Saint-Émilion',
              1: 'Northern Italy Red',
              2: 'Australian Shiraz',
              3: 'Southern Italy Primitivo',
              4: 'Tuscan Red', 
              5: 'Bordeaux Saint-Émilion',
              6: 'Tuscan Red',
              7: 'Italian Barbera',
              8: 'Southern Italy Red',
              9: 'Californian Pinot Noir',
              10: 'Italian Barbera',
              11: 'Tuscan Red',
              12: 'Australian Shiraz',
              13: 'Spanish Red',
              14: 'Italian Chianti',
              15: 'Spanish Tempranillo',
              16: 'Southern Italy Red',
              17: 'Southern Italy Red',
              18: 'Southern Italy Red',
              19: 'Spanish Syrah',
              20: 'Spanish Ribera Del Duero Red',
              21: 'Spanish Rioja Red',
              22: 'Portuguese Alentejo Red',
              23: 'Australian Shiraz',
              24: 'Spanish Red',
              25: 'Languedoc-Roussillon Red',
              26: 'Bordeaux Médoc',
              27: 'Argentinian Malbec',
              28: 'Australian Shiraz',
              29: 'Spanish Red', 
              30: 'Argentinian Malbec Red Blend',
              31: 'Southern Italy Primitivo',
              32: 'Southern Rhône Red',
              33: 'Bordeaux Saint-Émilion',
              34: 'Spanish Priorat Red',
              35: 'Tuscan Red',
              36: "Italian Montepulciano d'Abruzzo",
              37: 'Tuscan Red',
              38: 'Tuscan Red',
              39: 'Southern Italy Red',
              40: 'Central Italy Red',
              41: 'Californian Pinot Noir',
              42: 'Southern Portugal Red',
              43: 'Portuguese Douro Red',
              44: 'Tuscan Red',
              45: 'Tuscan Red',
              46: 'Central Italy Red',
              47: 'Southern Italy Red',
              48: 'Italian Amarone',
              49: 'Spanish Red',
              50: 'Bordeaux Saint-Émilion',
              51: 'Spanish Ribera Del Duero Red',
              52: 'Northern Italy Red',
              53: 'Burgundy Côte de Beaune Red',
              54: 'Northern Italy Red',
              55: 'Italian Amarone',
              56: 'Southern Italy Red',
              57: 'Central Italy Red',
              58: 'Southern Italy Red',
              59: 'Southern Italy Primitivo',
              60: 'Spanish Rioja Red',
              61: 'Italian Red',
              62: 'Bordeaux Saint-Émilion',
              63: 'Italian Amarone',
              64: 'Argentinian Cabernet Sauvignon - Malbec',
              65: 'Australian Shiraz',
              66: 'Southern Italy Primitivo',
              67: 'Ribena',
              68: 'Spanish Tempranillo',
              69: 'Spanish Ribera Del Duero Red',
              70: 'Southern Italy Primitivo',
              71: 'Central Italy Red',
              72: 'Bordeaux Pauillac',
              73: 'Central Italy Red',
              74: 'Italian Nebbiolo',
              75: 'Tuscan Red',
              76: 'Spanish Toro Red',
              77: 'Italian Chianti',
              78: 'Southern Italy Primitivo',
              79: 'Unknown1',
              80: 'Bordeaux Saint-Émilion',
              81: 'Southern Italy Red',
              82: 'Spanish Grenache',
              83: 'Spanish Rioja Red',
              84: 'Southern Rhône Red',
              85: 'Californian Zinfandel',
              86: 'Spanish Monastrell',
              87: 'Southern Portugal Red',
              88: 'South African Bordeaux Blend',
              89: 'Bordeaux Red',
              90: 'Spanish Ribera Del Duero Red',
              91: 'Spanish Ribera Del Duero Red',
              92: 'Spanish Red',
              93: 'Northern Italy Red',
              94: 'Californian Zinfandel',
              95: 'Southern Italy Primitivo',
              96: 'Bordeaux Médoc',
              97: 'Bordeaux Saint-Émilion',
              98: 'Southern Italy Red',
              99: 'Burgundy Red',
              100: 'Spanish Ribera Del Duero Red',
              101: 'Bordeaux Saint-Émilion',
              102: 'Bordeaux Graves Red',
              103: 'Southern Italy Red',
              104: 'Tuscan Red',
              105: 'Bordeaux Saint-Émilion',
              106: 'Spanish Rioja Red',
              107: 'Bordeaux Red',
              108: 'Southern Italy Red',
              109: 'Southern Italy Primitivo',
              110: 'Southern Italy Primitivo',
              111: 'Southern Italy Red',
              112: 'Italian Bolgheri',
              113: 'Bordeaux Red',
              114: 'Spanish Priorat Red'}

WINE_REGION = {0: 'Saint-Émilion Grand Cru',
               1: 'Italy / Northern Italy / Veneto',
               2: 'Australia / South Australia / Barossa / Barossa Valley',
               3: 'Italy / Southern Italy / Puglia',
               4: 'Italy / Central Italy / Toscana',
               5: 'France / Bordeaux / Libournais / Saint-Émilion / Montagne-Saint-Émilion',
               6: 'Italy / Central Italy / Toscana / Brunello di Montalcino',
               7: 'Italy / Northern Italy / Piemonte',
               8: 'Italy / Southern Italy / Molise / Biferno',
               9: 'United States / California / North Coast / Napa County / Napa Valley',
               10: "Italy / Northern Italy / Piemonte / Barbera d'Asti",
               11: "Italy / Central Italy / Toscana",
               12: 'Australia / South Australia / Barossa',
               13: 'Spain / Vino de España',
               14: 'Italy / Central Italy / Toscana / Chianti',
               15: 'Spain / Castilla y León / Cigales',
               16: 'Italy / Southern Italy / Campania / Irpinia / Irpinia Campi Taurasini',
               17: 'Italy / Southern Italy / Puglia / Salento',
               18: 'Italy / Southern Italy / Campania / Taurasi',
               19: 'Spain / Vino de España',
               20: 'Spain / Castilla y León / Ribera del Duero',
               21: 'Spain / Rioja',
               22: 'Portugal / Alentejano / Alentejo',
               23: 'Australia / South Australia / Barossa / Barossa Valley',
               24: 'Spain / Murcia / Jumilla',
               25: 'France / Languedoc-Roussillon / Languedoc / Terrasses du Larzac',
               26: 'France / Bordeaux / Médoc / Haut-Médoc',
               27: 'Argentina / Mendoza / Uco Valley',
               28: 'Australia / South Australia / Barossa / Barossa Valley',
               29: 'Spain / Vino de España',
               30: 'Argentina / Mendoza / Uco Valley',
               31: 'Italy / Southern Italy / Puglia / Primitivo di Manduria',
               32: 'France / Rhone Valley / Southern Rhône / Cairanne',
               33: 'France / Bordeaux / Libournais / Saint-Émilion / Montagne-Saint-Émilion',
               34: 'Spain / Catalunya / Priorat',
               35: 'Italy / Central Italy / Toscana',
               36: "Italy / Central Italy / Abruzzo / Montepulciano d'Abruzzo",
               37: 'Italy / Central Italy / Toscana',
               38: 'Italy / Central Italy / Toscana',
               39: 'Italy / Southern Italy / Basilicata / Aglianico del Vulture',
               40: 'Italy / Central Italy / Abruzzo',
               41: 'United States / California / Central Coast / Monterey County',
               42: 'Portugal / Península de Setúbal',
               43: 'Portugal / Northern Portugal / Duriense / Douro',
               44: 'Italy / Central Italy / Toscana',
               45: 'Italy / Central Italy / Toscana / Costa Toscana',
               46: 'Italy / Central Italy / Abruzzo',
               47: 'Italy / Southern Italy / Puglia',
               48: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella / Amarone della Valpolicella Classico',
               49: 'Spain / Islas Baleares / Mallorca',
               50: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               51: 'Spain / Castilla y León / Ribera del Duero',
               52: 'Italy / Northern Italy / Veneto',
               53: 'France / Bourgogne / Côte de Beaune / Santenay',
               54: 'Italy / Northern Italy / Emilia-Romagna',
               55: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella',
               56: 'Italy / Southern Italy / Puglia',
               57: 'Italy / Central Italy / Umbria',
               58: 'Italy / Southern Italy / Terre Siciliane',
               59: 'Italy / Southern Italy / Puglia / Salento',
               60: 'Spain / Rioja',
               61: "Italy / Vino d'Italia",
               62: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               63: 'Italy / Northern Italy / Veneto / Valpolicella / Amarone della Valpolicella / Amarone della Valpolicella Classico',
               64: 'Argentina / Salta',
               65: 'Australia / South Australia',
               66: 'Italy / Southern Italy / Puglia',
               67: 'Ribena',
               68: 'Spain / Castilla y León / Sardón de Duero',
               69: 'Spain / Castilla y León / Ribera del Duero',
               70: 'Italy / Southern Italy / Puglia / Primitivo di Manduria',
               71: 'Italy / Central Italy / Marche',
               72: 'France / Bordeaux / Médoc / Pauillac',
               73: 'Italy / Central Italy / Lazio',
               74: 'Italy / Northern Italy / Piemonte / Langhe',
               75: 'Italy / Central Italy / Toscana',
               76: 'Spain / Castilla y León / Toro',
               77: 'Italy / Central Italy / Toscana / Chianti / Chianti Classico',
               78: 'Italy / Southern Italy / Puglia',
               79: 'France / Vin de France',
               80: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               81: 'Italy / Southern Italy / Campania / Irpinia / Irpinia Campi Taurasini',
               82: 'Spain / Aragón / Cariñena',
               83: 'Spain / Rioja',
               84: 'France / Rhone Valley / Southern Rhône / Côtes-du-Rhône',
               85: 'United States / California / Central Valley / Lodi',
               86: 'Spain / Vino de España',
               87: 'Portugal / Península de Setúbal',
               88: 'South Africa / Western Cape / Coastal Region / Stellenbosch',
               89: 'France / Bordeaux',
               90: 'Spain / Castilla y León / Ribera del Duero',
               91: 'Spain / Castilla y León / Ribera del Duero',
               92: 'Spain / Valencia',
               93: 'Italy / Northern Italy / Veneto',
               94: 'United States / California',
               95: 'Italy / Southern Italy / Puglia / Salento',
               96: 'France / Bordeaux / Médoc / Haut-Médoc',
               97: 'France / Bordeaux / Libournais / Saint-Émilion / Montagne-Saint-Émilion',
               98: 'Italy / Southern Italy / Puglia / Salento',
               99: 'France / Bourgogne',
               100: 'Spain / Castilla y León / Ribera del Duero',
               101: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               102: 'France / Bordeaux / Graves',
               103: 'Italy / Southern Italy / Puglia / Salento',
               104: 'Italy / Central Italy / Toscana',
               105: 'France / Bordeaux / Libournais / Saint-Émilion / Saint-Émilion Grand Cru',
               106: 'Spain / Rioja / Rioja Alta',
               107: 'France / Bordeaux / Bordeaux Supérieur',
               108: 'Italy / Southern Italy / Terre Siciliane',
               109: 'Italy / Southern Italy / Puglia',
               110: 'Italy / Southern Italy / Puglia',
               111: 'Italy / Southern Italy / Puglia',
               112: 'Italy / Central Italy / Toscana / Bolgheri',
               113: 'France / Bordeaux / Bordeaux Supérieur',
               114: 'Spain / Catalunya / Priorat'}

WINE_YEAR = {0: '2015',
             1: '2015',
             2: '2018',
             3: '2020',
             4: '2017',
             5: '2019',
             6: '2019',
             7: '2019',
             8: '2015',
             9: '2018',
             10: '2018',
             11: '2016',
             12: '2019',
             13: '2019',
             14: '2016',
             15: '2014',
             16: '2016',
             17: '2018',
             18: '2015',
             19: '2019',
             20: '2020',
             21: '2017',
             22: '2020',
             23: '2019',
             24: '2018',
             25: '2018',
             26: '2017',
             27: '2017',
             28: '2016',
             29: '2022',
             30: '2013',
             31: '2019', 
             32: '2019',
             33: '2015',
             34: '2018',
             35: '2018',
             36: '2018',
             37: '2018',
             38: '2016',
             39: '2020',
             40: '2018',
             41: '2019',
             42 :'2012',
             43: '2019',
             44: '2017',
             45: '2015',
             46: '2021',
             47: '2020',
             48: '2015',
             49: '2018',
             50: '2018',
             51: '2016',
             52: '2018',
             53: '2014',
             54: '2020',
             55: '2018',
             56: '2020',
             57: '2012',
             58: '2018',
             59: '2020',
             60: '2018',
             61: '2020',
             62: '2015',
             63: '2015',
             64: '2015',
             65: '2018',
             66: '2017',
             67: 'Ribena',
             68: '2017',
             69: '2018',
             70: '2018',
             71: '2017',
             72: '2016',
             73: '2018',
             74: '2019',
             75: '2016',
             76: '2017',
             77: '2018',
             78: '2019',
             79: '2016',
             80: '2016',
             81: '2017',
             82: '2018',
             83: '2016',
             84: '2019',
             85: '2019',
             86: '2020',
             87: '2020',
             88: '2018',
             89: '2015',
             90: '2020',
             91: '2017',
             92: '2016',
             93: '2019',
             94: '2019',
             95: '2018',
             96: '2017',
             97: '2019',
             98: '2020',
             99: '2016',
             100: '2018',
             101: '2018',
             102: '2010',
             103: '2019',
             104: '2019',
             105: '2018',
             106: '2010',
             107: '2018',
             108: '2020',
             109: '2019',
             110: '2020',
             111: '2020',
             112: '2020',
             113: '2018',
             114: '2015'}

WINE_ALCOHOL_PERCENTAGE = {0: '14',
                           1: '13',
                           2: '14',
                           3: '14',
                           4: '14',
                           5: '12.5',
                           6: '13.5',
                           7: '13.5',
                           8: '14',
                           9: '13.5',
                           10: '14.5',
                           11: '13.5',
                           12: '14',
                           13: '14',
                           14: '12.5',
                           15: '14',
                           16: '15',
                           17: '13.5',
                           18: '14',
                           19: '13',
                           20: '14',
                           21: '12',
                           22: '14',
                           23: '15',
                           24: '14',
                           25: '13.5',
                           26: '13',
                           27: '14.8',
                           28: '14.5',
                           29: '14',
                           30: '14.5',
                           31: '14',
                           32: '14',
                           33: '14',
                           34: '14.5',
                           35: '14',
                           36: '13.5',
                           37: '13.5',
                           38: '14',
                           39: '13',
                           40: '14.5',
                           41: '13.5',
                           42: '14.5',
                           43: '13.5',
                           44: '13.5',
                           45: '14',
                           46: '14.5',
                           47: '14.5',
                           48: '16.5',
                           49: '15',
                           50: '13',
                           51: '14',
                           52: '13.5',
                           53: '13',
                           54: '18',
                           55: '15',
                           56: '14',
                           57: '13',
                           58: '14.5',
                           59: '14',
                           60: '14',
                           61: '14',
                           62: '14.5',
                           63: '15.5',
                           64: '16',
                           65: '14.5',
                           66: '14',
                           67: 'Ribena',
                           68: '14',
                           69: '14',
                           70: '14.5',
                           71: '14',
                           72: '14',
                           73: '15',
                           74: '14',
                           75: '14.5',
                           76: '15',
                           77: '14',
                           78: '15',
                           79: '14.5',
                           80: '14',
                           81: '15',
                           82: '14.5',
                           83: '13.5',
                           84: '13.5',
                           85: '14.5',
                           86: '15',
                           87: '14.5',
                           88: '13.5',
                           89: '13.5',
                           90: '14',
                           91: '13.5',
                           92: '13.5',
                           93: '14',
                           94: '14.5',
                           95: '14',
                           96: '13',
                           97: '14',
                           98: '14.5',
                           99: '13',
                           100: '14',
                           101: '14',
                           102: '13',
                           103: '14',
                           104: '14',
                           105: '14.5',
                           106: '14',
                           107: '12.5',
                           108: '13.5',
                           109: '13.7',
                           110: '14',
                           111: '14', 
                           112: '14',
                           113: '13.5',
                           114: '15'}

WINE_GRAPE = {0: ['Merlot', 'Cabernet Sauvignon'],
              1: ['Montepulciano'],
              2: ['Shiraz/Syrah'],
              3: ['Primitivo'],
              4: ['Sangiovese'],
              5: ['Cabernet Franc', 'Merlot'],
              6: ['Sangiovese'],
              7: ['Barbera'],
              8: ['Montepulciano', 'Aglianico'],
              9: ['Pinot Noir'],
              10: ['Barbera'],
              11: ['Merlot', 'Cabernet Sauvignon', 'Petit Verdot'],
              12: ['Shiraz/Syrah'],
              13: ['Shiraz/Syrah', 'Tempranillo'],
              14: ['Sangiovese'],
              15: ['Tempranillo'],
              16: ['Aglianico'],
              17: ['Negroamaro'],
              18: ['Aglianico'],
              19: ['Syrah/Shiraz'],
              20: ['Tempranillo'],
              21: ['Tempranillo', 'Mazuelo', 'Garnacha'],
              22: ['Alicante Bouschet'],
              23: ['Shiraz/Syrah'],
              24: ['Shiraz/Syrah', 'Petite Sirah', 'Monastrell'],
              25: ['Grenache'],
              26: ['Cabernet Sauvignon', 'Merlot'],
              27: ['Malbec'],
              28: ['Shiraz/Syrah', 'Tempranillo', 'Mourvedre', 'Grenache'],
              29: ['Shiraz/Syrah', 'Tempranillo'],
              30: ['Cabernet Sauvignon', 'Malbec', 'Shiraz/Syrah'],
              31: ['Primitivo'],
              32: ['Grenache', 'Mourvedre', 'Shiraz/Syrah'],
              33: ['Cabernet Sauvignon', 'Cabernet Franc', 'Merlot'],
              34: ['Garnacha', 'Cabernet Sauvignon', 'Merlot', 'Cariñena'],
              35: ['Merlot', 'Sangiovese'],
              36: ['Montepulciano'],
              37: ['Sangiovese'],
              38: ['Merlot', 'Sangiovese', 'Cabernet Sauvignon'],
              39: ['Aglianico'],
              40: ['Sangiovese'],
              41: ['Pinot Noir'],
              42: ['Shiraz/Syrah'],
              43: ['Tinta Roriz', 'Tinta Barroca', 'Touriga Franca', 'Touriga Nacional'],
              44: ['Sangiovese', 'Cabernet Sauvignon'],
              45: ['Merlot'],
              46: ['Montepulciano'],
              47: ['Negromaro'],
              48: ['Corvina', 'Rondinella', 'Corvinone'],
              49: ['Merlot', 'Cabernet Sauvignon', 'Callet', 'Manto Negro'],
              50: ['Merlot', 'Cabernet Sauvignon'],
              51: ['Tempranillo'],
              52: ['Corvina'],
              53: ['Pinot Noir'],
              54: ['Bonarda'],
              55: ['Corvina', 'Rondinella'],
              56: ['Merlot'],
              57: ['Shiraz/Syrah', 'Merlot', 'Sangiovese'],
              58: ['Cabernet Sauvignon', "Nero d'Avola"],
              59: ['Primitivo'],
              60: ['Tempranillo'],
              61: ['Merlot', "Nero d'Avola", 'MontePulciano', 'Primitivo'],
              62: ['Merlot', 'Cabernet Franc', 'Cabernet Sauvignon'],
              63: ['Corvinone', 'Corvina', 'Rondinella', 'Croatina'],
              64: ['Malbec'],
              65: ['Shiraz/Syrah'],
              66: ['Primitivo'],
              67: ['Ribena'],
              68: ['Tempranillo'],
              69: ['Tempranillo'],
              70: ['Primitivo'],
              71: ['Sangiovese', 'Montepulciano', 'Cabernet Sauvignon'],
              72: ['Cabernet Sauvignon', 'Merlot', 'Cabernet Franc', 'Petit Verdot'],
              73: ['Shiraz/Syrah'],
              74: ['Nebbiolo'],
              75: ['Merlot'],
              76: ['Tinta de toro'],
              77: ['Sangiovese'],
              78: ['Primitivo'],
              79: ['Merlot'],
              80: ['Merlot', 'Cabernet Franc', 'Cabernet Sauvignon'],
              81: ['Aglianico'],
              82: ['Garnacha'],
              83: ['Tempranillo', 'Mazuelo', 'Graciano'],
              84: ['Shiraz/Syrah', 'Grenache'],
              85: ['Zinfandel'],
              86: ['Monastrell'],
              87: ['Touriga Nacional', 'Castelao', 'Alicante Bouschet'],
              88: ['Cabernet Franc', 'Merlot', 'Cabernet Sauvignon'],
              89: ['Cabernet Sauvignon', 'Merlot'],
              90: ['Tempranillo'],
              91: ['Tempranillo'],
              92: ['Tempranillo', 'Monastrell'],
              93: ['Cabernet Sauvignon', 'Merlot'],
              94: ['Zinfandel'],
              95: ['Primitivo'],
              96: ['Cabernet Sauvignon', 'Merlot'],
              97: ['Merlot', 'Cabernet Franc'],
              98: ['Montemajor'],
              99: ['Gamay'],
              100: ['Tempranillo', 'Cabernet Sauvignon'],
              101: ['Merlot', 'Cabernet Sauvignon'],
              102: ['Cabernet Sauvignon', 'Merlot', 'Petit Verdot'],
              103: ['Merlot', 'Malvasia Nera'],
              104: ['Sangiovese'],
              105: ['Merlot', 'Cabernet Franc', 'Cabernet Sauvignon'],
              106: ['Tempranillo', 'Mazuelo'],
              107: ['Merlot'],
              108: ['Cabernet Sauvignon', 'Merlot', "Nero d'Avola"],
              109: ['Zinfandel'],
              110: ['Negromaro'],
              111: ['Merlot'],
              112: ['Cabernet Sauvignon'],
              113: ['Merlot'],
              114: ['Shiraz/Syrah', 'Merlot', 'Cabernet Sauvignon', 'Carignan', 'Grenache']}

WINE_GRAPE_SIMPLIFIED = {
        0: 'Blend',
        1: 'Montepulciano',
        2: 'Shiraz/Syrah',
        3: 'Primitivo',
        4: 'Sangiovese',
        5: 'Blend',
        6: 'Sangiovese',
        7: 'Barbera',
        8: 'Blend',
        9: 'Pinot Noir',
        10: 'Barbera',
        11: 'Blend',
        12: 'Shiraz/Syrah',
        13: 'Blend',
        14: 'Sangiovese',
        15: 'Tempranillo',
        16: 'Aglianico',
        17: 'Negroamaro',
        18: 'Aglianico',
        19: 'Blend',
        20: 'Tempranillo',
        21: 'Blend',
        22: 'Alicante Bouschet',
        23: 'Shiraz/Syrah',
        24: 'Blend',
        25: 'Grenache',
        26: 'Blend',
        27: 'Malbec',
        28: 'Blend',
        29: 'Blend',
        30: 'Blend',
        31: 'Primitivo',
        32: 'Blend',
        33: 'Blend',
        34: 'Blend',
        35: 'Blend',
        36: 'Montepulciano',
        37: 'Sangiovese',
        38: 'Blend',
        39: 'Aglianico',
        40: 'Sangiovese',
        41: 'Pinot Noir',
        42: 'Shiraz/Syrah',
        43: 'Blend',
        44: 'Blend',
        45: 'Merlot',
        46: 'Montepulciano',
        47: 'Negromaro',
        48: 'Blend',
        49: 'Blend',
        50: 'Blend',
        51: 'Tempranillo',
        52: 'Corvina',
        53: 'Pinot Noir',
        54: 'Bonarda',
        55: 'Blend',
        56: 'Merlot',
        57: 'Blend',
        58: 'Blend',
        59: 'Primitivo',
        60: 'Tempranillo',
        61: 'Blend',
        62: 'Blend',
        63: 'Blend',
        64: 'Malbec',
        65: 'Shiraz/Syrah',
        66: 'Primitivo',
        67: 'Ribena',
        68: 'Tempranillo',
        69: 'Tempranillo',
        70: 'Primitivo',
        71: 'Blend',
        72: 'Blend',
        73: 'Shiraz/Syrah',
        74: 'Nebbiolo',
        75: 'Merlot',
        76: 'Tinta de toro',
        77: 'Sangiovese',
        78: 'Primitivo',
        79: 'Merlot',
        80: 'Blend',
        81: 'Aglianico',
        82: 'Garnacha',
        83: 'Blend',
        84: 'Blend',
        85: 'Zinfandel',
        86: 'Monastrell',
        87: 'Blend',
        88: 'Blend',
        89: 'Blend',
        90: 'Tempranillo',
        91: 'Tempranillo',
        92: 'Blend',
        93: 'Blend',
        94: 'Zinfandel',
        95: 'Primitivo',
        96: 'Blend',
        97: 'Blend',
        98: 'Montemajor',
        99: 'Gamay',
        100: 'Blend',
        101: 'Blend',
        102: 'Blend',
        103: 'Blend',
        104: 'Sangiovese',
        105: 'Blend',
        106: 'Blend',
        107: 'Merlot',
        108: 'Blend',
        109: 'Zinfandel',
        110: 'Negromaro',
        111: 'Merlot',
        112: 'Cabernet Sauvignon',
        113: 'Merlot',
        114: 'Blend'
    }

# Collected on
# May 2nd 2023

WINE_PRICE = {
    0: 239.95,
    1: 35.09,
    2: 189,
    3: 75.95,
    4: 266.80,
    5: 119,
    6: 595.80,
    7: 59.98,
    8: 69,
    9: 99,
    10: 475,
    11: 115,
    12: 99,
    13: 98,
    14: 78.09,
    15: 80.02,
    16: 139.95,
    17: 85,
    18: 259,
    19: 69,
    20: 89,
    21: 99.95,
    22: 52.08,
    23: 199,
    24: 76.07,
    25: 96.12,
    26: 91.72,
    27: 97.61,
    28: 77.19,
    29: 59,
    30: 210.34,
    31: 123.31,
    32: 99,
    33: 109.30,
    34: 174.35,
    35: 69,
    36: 119.21,
    37: 110.18,
    38: 79,
    39: 59.53,
    40: 159,
    41: 149.95,
    42: 79,
    43: 139,
    44: 99,
    45: 149,
    46: 169,
    47: 46.94,
    48: 491.65,
    49: 147.53,
    50: 299,
    51: 87.40,
    52: 62.59,
    53: 144.40,
    54: 73,
    55: 215,
    56: 97.90,
    57: 60.13,
    58: 168.45,
    59: 59.01,
    60: 89,
    61: 69,
    62: 169,
    63: 409,
    64: 228.29,
    65: 79,
    66: 66.69,
    67: 'Ribena',
    68: 329.95,
    69: 169,
    70: 99,
    71: 149,
    72: 159,
    73: 99.95,
    74: 139.95,
    75: 120.18,
    76: 99.10,
    77: 219,
    78: 69,
    79: 81.14,
    80: 191.41,
    81: 139.95,
    82: 94.55,
    83: 99.99,
    84: 119.21,
    85: 63.11,
    86: 59,
    87: 66.61,
    88: 202.02,
    89: 86.80,
    90: 190,
    91: 254.95,
    92: 44.71,
    93: 132.70,
    94: 75,
    95: 139,
    96: 96.86,
    97: 119,
    98: 129,
    99: 161.98,
    100: 89,
    101: 299,
    102: 153.94,
    103: 114.80,
    104: 69,
    105: 149.84,
    106: 159,
    107: 118.47,
    108: 73,
    109: 169,
    110: 69.44,
    111: 97.90,
    112: 239,
    113: 66.31,
    114: 99.99
}

WINE_RATING = {
    0: 4.1,
    1: 3.4,
    2: 3.8,
    3: 4.1,
    4: 3.9,
    5: 4.0,
    6: 4.5,
    7: 3.9,
    8: 4.0,
    9: 4.0,
    10: 3.9,
    11: 3.7,
    12: 4.2,
    13: 4.2,
    14: 3.2,
    15: 4.1,
    16: 4.2,
    17: 4.2,
    18: 3.9,
    19: 4.1,
    20: 4.0,
    21: 3.9,
    22: 4.0,
    23: 4.0,
    24: 3.9,
    25: 4.0,
    26: 3.8,
    27: 4.0,
    28: 3.4,
    29: 4.1,
    30: 4.2,
    31: 4.3,
    32: 4.0,
    33: 4.0,
    34: 4.3,
    35: 3.9,
    36: 3.8,
    37: 3.9,
    38: 4.0,
    39: 3.8,
    40: 4.3,
    41: 3.8,
    42: 4.1,
    43: 4.0,
    44: 4.2,
    45: 4.2,
    46: 4.3,
    47: 4.1,
    48: 4.3,
    49: 4.1,
    50: 4.1,
    51: 3.6,
    52: 4.1,
    53: 3.9,
    54: 4.0,
    55: 4.3,
    56: 4.0,
    57: 3.1,
    58: 4.0,
    59: 4.1,
    60: 3.9,
    61: 4.1,
    62: 4.1,
    63: 4.1,
    64: 4.3,
    65: 4.0,
    66: 4.0,
    67: 'Ribena',
    68: 4.4,
    69: 4.0,
    70: 4.3,
    71: 4.2,
    72: 4.1,
    73: 3.8,
    74: 3.9,
    75: 3.8,
    76: 3.7,
    77: 4.1,
    78: 3.9,
    79: 3.6,
    80: 4.1,
    81: 4.2,
    82: 4.0,
    83: 4.0,
    84: 4.2,
    85: 3.8,
    86: 4.1,
    87: 4.2,
    88: 4.3,
    89: 4.2,
    90: 4.1,
    91: 4.1,
    92: 3.4,
    93: 4.1,
    94: 3.9,
    95: 4.0,
    96: 3.8,
    97: 4.0,
    98: 4.2,
    99: 3.8,
    100: 3.7,
    101: 4.1,
    102: 4.2,
    103: 4.0,
    104: 4.0,
    105: 4.2,
    106: 4.3,
    107: 4.0,
    108: 4.2,
    109: 4.1,
    110: 4.1,
    111: 4.0,
    112: 4.2,
    113: 4.1,
    114: 3.9
}

for key in WINE_GRAPE_SIMPLIFIED:
    if WINE_GRAPE_SIMPLIFIED[key] == 'Blend':
        WINE_GRAPE_SIMPLIFIED[key] = WINE_GRAPE[key][0]

for key in WINE_RATING:
    if WINE_RATING[key] != 'Ribena':
        value = float(WINE_RATING[key])
        rounded_value = round(value * 2) / 2
        WINE_RATING[key] = str(rounded_value)

for key in WINE_ALCOHOL_PERCENTAGE:
    if WINE_ALCOHOL_PERCENTAGE[key] != 'Ribena':
        WINE_ALCOHOL_PERCENTAGE[key] = str(round(float(WINE_ALCOHOL_PERCENTAGE[key])))

for key in WINE_PRICE:
    if WINE_PRICE[key] != 'Ribena':
        WINE_PRICE[key] = str(round(float(WINE_PRICE[key]) * 0.02) * 50)


def remove_ribena(input_dict):
    return {key: value for key, value in input_dict.items() if value != 'Ribena'}

DICTIONARIES = {
    "Country" : WINE_COUNTRY,
    "Region": WINE_REGION,
    "Grape": WINE_GRAPE_SIMPLIFIED,
    "Alcohol %": WINE_ALCOHOL_PERCENTAGE,
    "Year": WINE_YEAR,
    "Price": WINE_PRICE,
    "Rating": WINE_RATING
}

DICTIONARIES = {key: replace_ids_in_dict(value, duplicate_key_mapping) for key, value in DICTIONARIES.items()}

if not RIBENA:
    DICTIONARIES = {
        key: remove_ribena(replace_ids_in_dict(value, duplicate_key_mapping))
        for key, value in DICTIONARIES.items()
    }


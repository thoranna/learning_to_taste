import json

def load_data():
    with open('data/wine_coordinates.json', 'r') as json_file:
        wine_coordinates_data = json.load(json_file)
    
    with open('data/experiment_rounds.json', 'r') as json_file:
        experiment_rounds_data = json.load(json_file)
    
    return wine_coordinates_data, experiment_rounds_data
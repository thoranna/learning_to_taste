import pandas as pd


def load_data():
    # Read the CSV file as a pandas DataFrame
    # data = pd.read_csv('data/user_vintage_review.csv')
    data = pd.read_csv('data/vintage_review.csv_75K.part_00000')
    return data
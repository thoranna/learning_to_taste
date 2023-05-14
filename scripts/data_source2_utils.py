import pandas as pd


def load_data():
    # Read the CSV file as a pandas DataFrame
    data = pd.read_csv('data/user_vintage_review.csv')
    return data
import os
import logging
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load parameters from params.yaml
def load_params():
    try:
        params = yaml.safe_load(open('params.yaml', 'r'))
        max_features = params['feature_engineering']['max_features']
        logging.info(f"Max features loaded from params.yaml: {max_features}")
        return max_features
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        return None

# Function to load data
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")
        return data
    else:
        logging.error(f"File {file_path} does not exist.")
        return None

# Handle missing values in data
def handle_missing_values(df):
    df['content'] = df['content'].fillna("")
    logging.info("Missing values in 'content' column handled.")
    return df

# Function to create directory if it doesn't exist
def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory at {path}")
    else:
        logging.info(f"Directory {path} already exists")

# Function to vectorize text data
def vectorize_text(X_train, X_test, max_features):
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    logging.info("Text data vectorized using CountVectorizer.")
    return X_train_bow, X_test_bow, vectorizer

# Function to save data to CSV
def save_to_csv(X_train_bow, X_test_bow, y_train, y_test, vectorizer):
    train_df = pd.DataFrame(X_train_bow.toarray(), columns=vectorizer.get_feature_names_out())
    train_df['labels'] = y_train

    test_df = pd.DataFrame(X_test_bow.toarray(), columns=vectorizer.get_feature_names_out())
    test_df['labels'] = y_test

    data_path = os.path.join('data', 'processed')
    create_data_directory(data_path)

    train_df.to_csv(os.path.join(data_path, 'train_bow.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'test_bow.csv'), index=False)
    logging.info("Processed data saved to 'data/processed'.")

# Main function to process the data
def process_data():
    # Load parameters
    max_features = load_params()

    if max_features is None:
        logging.error("Max features parameter not found. Exiting.")
        return

    # Load train and test data
    train_data = load_data('./data/interim/train_processed_data.csv')
    test_data = load_data('./data/interim/test_processed_data.csv')

    if train_data is None or test_data is None:
        logging.error("Error loading data. Exiting.")
        return

    # Handle missing values
    train_data = handle_missing_values(train_data)
    test_data = handle_missing_values(test_data)

    # Prepare feature and target variables
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values
    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # Vectorize the data
    X_train_bow, X_test_bow, vectorizer = vectorize_text(X_train, X_test, max_features)

    # Save processed data
    save_to_csv(X_train_bow, X_test_bow, y_train, y_test, vectorizer)

if __name__ == "__main__":
    process_data()

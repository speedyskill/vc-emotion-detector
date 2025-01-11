import os
import logging
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load parameters from params.yaml
def load_params():
    try:
        params = yaml.safe_load(open('params.yaml', 'r'))
        n_estimators = params['model_building']['n_estimators']
        learning_rate = params['model_building']['learning_rate']
        logging.info(f"Parameters loaded from params.yaml: n_estimators={n_estimators}, learning_rate={learning_rate}")
        return n_estimators, learning_rate
    except Exception as e:
        logging.error(f"Error loading params.yaml: {e}")
        return None, None

# Function to load training data
def load_training_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"Loaded training data from {file_path}")
        return data
    else:
        logging.error(f"File {file_path} does not exist.")
        return None

# Function to train the model
def train_model(X_train, y_train, n_estimators, learning_rate):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        logging.info("Model trained successfully.")
        return clf
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

# Function to save the model
def save_model(model, model_filename):

    models_path = os.path.join('src', 'models')  # Path to 'src/models' directory
    
    # Create the folder if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    # Full path to save the model
    model_filepath = os.path.join(models_path, model_filename)

    try:
        with open(model_filepath, 'wb') as model_file:
            pickle.dump(model, model_file)
        logging.info(f"Model saved to {model_filepath}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

# Main function to build and save the model
def build_and_save_model():
    # Load parameters
    n_estimators, learning_rate = load_params()

    if n_estimators is None or learning_rate is None:
        logging.error("Parameters not found. Exiting.")
        return

    # Load training data
    train_df = load_training_data('./data/processed/train_bow.csv')

    if train_df is None:
        logging.error("Training data not found. Exiting.")
        return

    # Prepare feature and target variables
    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values

    # Train the model
    clf = train_model(X_train, y_train, n_estimators, learning_rate)

    if clf is None:
        logging.error("Model training failed. Exiting.")
        return

    # Save the model
    save_model(clf, 'model.pkl')

if __name__ == "__main__":
    build_and_save_model()

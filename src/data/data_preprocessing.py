import os
import re
import logging
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Create a directory for processed data
def create_data_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory at {path}")
    else:
        logging.info(f"Directory {path} already exists")

# Function to load the dataset
def load_data(file_path):
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        logging.info(f"Loaded data from {file_path}")
        return data
    else:
        logging.error(f"File {file_path} does not exist.")
        return None

# Text preprocessing functions
def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )
    text = re.sub('\s+', ' ', text)  # Remove extra whitespaces
    return " ".join(text.split())

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

# Remove small sentences (if less than 3 words)
def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan
    logging.info("Removed small sentences from the dataset.")
    return df

# Normalize text by applying all preprocessing functions
def normalize_text(df):
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    logging.info("Text normalization completed.")
    return df

# Main function to preprocess the dataset
def preprocess_data():
    # Load train and test data
    train_data = load_data('./data/raw/train_data.csv')
    test_data = load_data('./data/raw/test_data.csv')

    # Normalize the data
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    # Create processed data directory
    processed_data_path = os.path.join('data', 'interim')
    create_data_directory(processed_data_path)

    # Store the processed data
    train_processed_data.to_csv(os.path.join(processed_data_path, 'train_processed_data.csv'))
    test_processed_data.to_csv(os.path.join(processed_data_path, 'test_processed_data.csv'))
    logging.info("Processed data saved in 'data/processed'.")

if __name__ == "__main__":
    preprocess_data()

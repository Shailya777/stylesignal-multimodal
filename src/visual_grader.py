# Imports:
import os
import logging
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Setting up logging:
logging.basicConfig(level= logging.INFO,
                    format= '%(asctime)s - %(levelname)s - %(message)s')

# Setting up Data Directory Paths:
IMAGE_DIR= 'data/images_10k'
TARGET_CSV= 'data/processed/cnn_proxy_targets_10k.csv'
OUTPUT_CSV= 'data/processed/cnn_visual_scores.csv'
MODEL_SAVE_PATH= 'data/processed/mobilenetv2_visual_grader.h5'

# Variables:
BATCH_SIZE= 32
IMG_SIZE= (224, 224)
EPOCHS= 5

# Function to load and preprocess the dataset:
def load_and_preprocess_data():
    """
    Loads the target CSV and constructs file paths for the images.

    Reads the pre-processed CNN proxy targets, verifies that the corresponding 
    images actually exist in the image directory, and constructs absolute 
    file paths to be used by the TensorFlow data pipeline.

    Returns:
        pd.DataFrame: A dataframe containing 'article_id', 'file_path', 
        and the continuous target 'cnn_target_scaled'.
    """

    logging.info('Loading target data and mapping image paths...')
    df= pd.read_csv(TARGET_CSV, dtype= {'article_id': str})

    # Mapping the Expected File Path:
    df['file_path']= df['article_id'].apply(lambda x: os.path.join(IMAGE_DIR, f"{x}.jpg"))

    # Filtering Out any Missing Images:
    df= df[df['file_path'].apply(os.path.exists)]

    logging.info(f"Successfully Mapped {len(df)} images to their corresponding article IDs.")

    return df

if __name__== "__main__":
    # Load and preprocess the data:
    df= load_and_preprocess_data()

    # Display the first few rows of the dataframe:
    logging.info(f"First few rows of the dataframe:\n{df.head()}")
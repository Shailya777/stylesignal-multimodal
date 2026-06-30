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

# Building TensorFlow Data Pipeline:
def build_tf_dataset(df, is_training= True):
    """
    Builds an optimized tf.data.Dataset for training or inference.

    Args:
        df (pd.DataFrame): Dataframe containing 'file_path' and 'cnn_target_scaled'.
        is_training (bool): If True, shuffles the dataset for training. 
                            If False, keeps order for inference mapping.

    Returns:
        tf.data.Dataset: A batched and prefeched TensorFlow dataset ready for model input.
    """

    paths= df['file_path'].values
    targets= df['cnn_target_scaled'].values

    # parsing and Decoding the images:
    def process_image(file_path, label):
        img= tf.io.read_file(file_path)
        img= tf.image.decode_jpeg(img, channels= 3)
        img= tf.image.resize(img, IMG_SIZE)
        img= tf.keras.applications.mobilenet_v2.preprocess_input(img) # MobileNetV2 expects inputs scaled between -1 and 1
        return img, label
    
    ds= tf.data.Dataset.from_tensor_slices((paths, targets))
    ds= ds.map(process_image, num_parallel_calls= tf.data.AUTOTUNE)

    if is_training:
        ds= ds.shuffle(buffer_size= 1000)
    
    ds= ds.batch(BATCH_SIZE).prefetch(buffer_size= tf.data.AUTOTUNE)
    return ds



if __name__== "__main__":
    # Load and preprocess the data:
    df= load_and_preprocess_data()

    # Display the first few rows of the dataframe:
    logging.info(f"First few rows of the dataframe:\n{df.head()}")
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
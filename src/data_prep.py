# Imports:
import os
import zipfile
import pandas as pd
import numpy as np
from skleatn.preprocessing import MinMaxScaler
import random
import logging

# Setting up logging:
logging.basicConfig(level= logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Setting up Data Directory Paths:
ZIP_PATH= '..data/raw/h-and-m-personalized-fashion-recommendations.zip'
PROCESSED_DIR= '../data/processed/'
IMAGE_OUT_DIR= '../data/images_10k/'
SAMPLE_SIZE= 10000


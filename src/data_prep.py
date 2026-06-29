# Imports:
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
import logging

# Setting up logging:
logging.basicConfig(level= logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Setting up Data Directory Paths:
ZIP_PATH= os.path.abspath('data/raw/h-and-m-personalized-fashion-recommendations.zip')
PROCESSED_DIR= os.path.abspath('data/processed/')
IMAGE_OUT_DIR= os.path.abspath('data/images_10k/')
SAMPLE_SIZE= 10000

def run_data_prep():
    """

    """
    
    # Ensure the data directories exists:
    os.makedirs(PROCESSED_DIR, exist_ok= True)
    os.makedirs(IMAGE_OUT_DIR, exist_ok= True)

    logging.info("Extracting the zip file...")
    with zipfile.ZipFile(file= ZIP_PATH, mode= 'r') as archive:

        # Reading Data Directly from the Zip File:
        logging.info("Reading transactions_train.csv into memory...")
        with archive.open('transactions_train.csv') as f:
            df= pd.read_csv(f, dtype= {'article_id': str})

        
        logging.info("Parsing dates and extracting month...")
        df['t_dat']= pd.to_datetime(df['t_dat'])
        df['month']= df['t_dat'].dt.to_period('M').astype(str)

        # Creating Time-Series Target Dataframe:
        logging.info("Building time-series demand targets...")
        ts_df= df.groupby(['article_id', 'month']).size().reset_index(name= 'monthly_sales')
        ts_df.to_csv(os.path.join(PROCESSED_DIR, 'time_series_targets.csv'), index= False)

        # Creating CNN Proxt Label Dataframe:
        logging.info("Building CNN proxy targets (total volume)...")
        total_df= df.groupby('article_id').size().reset_index(name= 'total_sales')
        
        # Scaling Total Scales:
        logging.info("Applying np.log1p and MinMax scaling...")
        total_df['total_sales_log']= np.log1p(total_df['total_sales'])
        
        scaler= MinMaxScaler()
        total_df['cnn_target_scaled']= scaler.fit_transform(total_df[['total_sales_log']])

        # Sampling 10K Unique Articles for CNN Training:
        logging.info(f"Sampling {SAMPLE_SIZE} unique articles for CNN training...")
        unique_articles= total_df['article_id'].unique().tolist()
        sampled_articles= random.sample(unique_articles, min(SAMPLE_SIZE, len(unique_articles)))

        # Filtering Dataframes to Only Include Sampled Articles:
        logging.info("Filtering dataframes to only include sampled articles...")
        sampled_total_df= total_df[total_df['article_id'].isin(sampled_articles)]
        sampled_total_df.to_csv(os.path.join(PROCESSED_DIR, 'cnn_proxy_targets_10k.csv'), index= False)

        # Extracting Images for Sampled Articles:
        logging.info(f"Extracting images for sampled {len(sampled_articles)} articles...")

        all_zip_files= archive.namelist()
        extracted_count= 0
        missing_count= 0

        for article_id in sampled_articles:

            # Constructing the expected image file path (The H&M dataset structures images in folders by the first 3 digits of the article_id
            # e.g., images/010/0108775015.jpg):
            folder_prefix= article_id[:3]
            expected_image_path= f"images/{folder_prefix}/{article_id}.jpg"

            if expected_image_path in all_zip_files:

                # Extracting File to Memory and writing it straight to the output (images_1ok) directory:
                image_data= archive.read(expected_image_path)
                out_path= os.path.join(IMAGE_OUT_DIR, f"{article_id}.jpg")

                with open(out_path, 'wb') as img_file:
                    img_file.write(image_data)
                
                extracted_count += 1
            
            else:
                missing_count += 1
            
        logging.info(f"Extraction complete. Extracted {extracted_count} images, {missing_count} images were missing.")


if __name__ == "__main__":
    run_data_prep()
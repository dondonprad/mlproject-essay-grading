import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass 

from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'dataset_essay_grading_training.csv')
    test_data_path:str = os.path.join('artifacts', 'dataset_essay_grading_test.csv')
    raw_data_path:str = os.path.join('artifacts', 'dataset_essay_grading.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def InitiateDataIngestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df_essay = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Dataset ready to use')

            #create label 
            x, y = np.asarray(df_essay["jawaban_mahasiswa"]), np.asarray(df_essay["jawaban_dosen"])
            label_map = {cat:index for index,cat in enumerate(np.unique(y))}
            y_prep = np.asarray([label_map[l] for l in y])
            df_essay['label'] = y_prep
            logging.info('label initiated')

            #feature = df_essay[["jawaban_mahasiswa","jawaban_dosen"]]
            label = df_essay['label']

            #train test split data
            train_set, test_set = train_test_split(df_essay, test_size=0.2, stratify=label, random_state=42 )
            
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info('train and test data initiated')

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    x, y = obj.InitiateDataIngestion()

    

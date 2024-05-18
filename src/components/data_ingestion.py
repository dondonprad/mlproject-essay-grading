import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass 


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'dataset_essay_grading_training.csv')
    test_data_path:str = os.path.join('artifacts', 'dataset_essay_grading_test.csv')
    raw_data_path:str = os.path.join('artifacts', 'dataset_esay_grading.csv')

class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def InitiateDataIngestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df_essay = pd.read_csv(self.ingestion_config.raw_data_path)
            df_essay_train = pd.read_csv(self.ingestion_config.train_data_path)
            df_essay_test = pd.read_csv(self.ingestion_config.test_data_path)
            print(df_essay)
            return (self.ingestion_config.raw_data_path,
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    x, y, z = obj.InitiateDataIngestion()
    print(x)
    print(y)
    print(z)

    

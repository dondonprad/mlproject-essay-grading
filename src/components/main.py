import os
import sys
import pandas as pd
from src.components.data_ingestion import DataIngestionConfig
from src.components.data_processing import PreprocessingText, CreateFeature, WordEmbedding

if __name__=='__main__':
    df = pd.read_csv(DataIngestionConfig.train_data_path)
    obj = PreprocessingText()
    data = obj.processing_text(df)
    token, y = CreateFeature().label_tokenizer(data)
    x_vec = WordEmbedding().text2vec(token)
    print(x_vec)
    print(x_vec.shape)
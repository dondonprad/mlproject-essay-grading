import os
import sys
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestionConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import re, collections
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
import nltk 
import gensim

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

@dataclass
class ProcessingTextConfig():
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class PreprocessingText():
    def __init__ (self, df):
        self.df = df

    def preprocessing_text(self):
        try: 
            logging.info('Preprocessing Text will process')
            factory = StopWordRemoverFactory()
            stopword = factory.create_stop_word_remover()

            factory = StemmerFactory()
            stemmer = factory.create_stemmer()

            new_data = []
            exclude = ['', '"', '”']

            self.df['jawaban_mahasiswa'] = self.df['jawaban_mahasiswa'].str.strip().str.lower()
            self.df['jawaban_dosen'] = self.df['jawaban_dosen'].str.strip().str.lower()
            
            for index, row in self.df.iterrows():
                label = row['human_rater']
                sentence_mahasiswa = row['jawaban_mahasiswa']
                # sentence_dosen = row['jawaban_dosen']
                ## === menghilangkan kata yang dianggap tidak mengacu pada inti kalimatnya, seperti kata sang, si, dan, itu, dan lain sebagainya
                sentence_mahasiswa = stopword.remove(sentence_mahasiswa)
                # sentence_dosen = stopword.remove(sentence_dosen)

                ## === mentransformasi kata-kata pada text menjadi kata dasarnya
                sentence_mahasiswa = stemmer.stem(sentence_mahasiswa)
                # sentence_dosen = stemmer.stem(sentence_dosen)
                # sentence = sentence_stemming if sentence_stopwords == "" or sentence_stopwords == " " else sentence_stopwords

                self.df.at[index, 'jawaban_mahasiswa']= sentence_mahasiswa
                # df.at[index, 'jawaban_dosen']= sentence_dosen

                # sentence = find_words = re.compile(r'(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))').findall
                # sentence = re.match("^[A-Za-z]*$", sentence):
                tokens_mahasiswa = nltk.tokenize.sent_tokenize(sentence_mahasiswa)
                # tokens_dosen = nltk.tokenize.sent_tokenize(sentence_dosen)

                if len(tokens_mahasiswa) > 1:
                    print("here")
                    test = []
                    for i, token in enumerate(tokens):
                        ## === splitting text menjadi beberapa kalimat berdasarkan tanda ? ! "" dengan split regex
                        test += re.split(r'[!?"(.*?)"]+|(?<!\.)\.(?!\.)', token)

                    ## === menghapus tanda baca dan number pada kalimat yang dihasilkan dengan substract regex
                    tokens = [re.sub(r'[^\w]', ' ', i) for i in test if i not in exclude]

                    ## === menghilangkan angka dengan substract regex
                    tokens = [re.sub("(\s\d+)","",i)  for i in tokens]

                    for i, token in enumerate(tokens):
                        if token != '' or token != ' ' or token != "\"":

                            ## === menghilangkan duplicate words
                            token = ' '.join(dict.fromkeys(token.split()))

                            new_data.append([token, label])
            return self.df.copy()
            logging.info('Preprocessing Text is completed')
        
        except Exception as e:
            raise CustomException(e, sys)


class CreateFeature():
    def __init__(self, df):
        self.df = df

    def cosine_similarity(self):
        try:
            df_cos = self.preprocessing_text(self.df)
            vec = TfidfVectorizer()
            similarity = []
            for index, row in df_cos.iterrows():
                corpus = [row['jawaban_mahasiswa'], row['jawaban_dosen']]
                sparse_matrix = vec.fit_transform(corpus)
                doc_term_matrix = sparse_matrix.todense()
                sim = cosine_similarity(sparse_matrix[0], sparse_matrix[1])
                similarity.append(sim[0][0])
            df_cos['similarity'] = similarity
            return df_cos
          
        except Exception as e:
            raise CustomException(e, sys)


    def label_encoding(self):
        try:
            df_le = self.cosine_similarity()
            x, y = np.asarray(df_le["jawaban_mahasiswa"]), np.asarray(df_le["jawaban_dosen"])
            label_map = {cat:index for index,cat in enumerate(np.unique(y))}
            y_prep = np.asarray([label_map[l] for l in y])
            x_tokenized = [[w for w in sentence.split(" ") if w != ""] for sentence in x] #split word into token
            
class WordEmbedding():
    def __init__(self, )
            


if __name__=='__main__':
    df = pd.read_csv(DataIngestionConfig.train_data_path)
    obj = PreprocessingText()
    data = obj.preprocessing_text(df)
    print(data)
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

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

@dataclass
class WordEmbeddingConfig():
    WordEmbedding_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

@dataclass
class TfidfConfig():
    tfidf_obj_file_path = os.path.join('artifacts', 'tfidf.pkl')


class DataPreprocessing():
    def initiate_data_preprocessing(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('read train and test data are completed')
            logging.info('Obtaining preprocessing object')

            data_set = {'train_data':train_df,
                        'test_data':test_df}

            return data_set

        except Exception as e:
            raise CustomException(e, sys)

class PreprocessingText():
    def processing_text(self, data_set):
        try: 
            logging.info('Preprocessing Text will process')
            factory = StopWordRemoverFactory()
            stopword = factory.create_stop_word_remover()

            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            new_data = []
            df_set = []
            exclude = ['', '"', '”']

            for i in data_set.keys():
                df = data_set[i]
                df['jawaban_mahasiswa'] = df['jawaban_mahasiswa'].str.strip().str.lower()
                df['jawaban_dosen'] = df['jawaban_dosen'].str.strip().str.lower()
                
                for index, row in df.iterrows():
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

                    df.at[index, 'jawaban_mahasiswa']= sentence_mahasiswa
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

                df_set.append(df)

            data_set_new = {'train_set':df_set[0],
                            'test_set':df_set[1]}

            return data_set_new
            logging.info('Preprocessing Text is completed')
        
        except Exception as e:
            raise CustomException(e, sys)


class CreateFeature():
    def __init__(self):
        self.tfidf_config = TfidfConfig()

    def similarity_tokenizer(self,data_set):
        try:
            logging.info("calculate cosine similarity")
            data_cos = PreprocessingText().processing_text(data_set)
            vec = TfidfVectorizer()
            x_tokenized = []
            similarity = [[],[]]
            for i in data_cos.keys():
                if str(i) == 'train_set':
                    for _, row in data_cos['train_set'].iterrows():
                        corpus = [row['jawaban_mahasiswa'], row['jawaban_dosen']]
                        sparse_matrix = vec.fit_transform(corpus)
                        #doc_term_matrix = sparse_matrix.todense()
                        sim = cosine_similarity(sparse_matrix[0], sparse_matrix[1])
                        similarity[0].append(sim[0][0])
    
                else:
                    for _, row in data_cos['test_set'].iterrows():
                        corpus = [row['jawaban_mahasiswa'], row['jawaban_dosen']]
                        sparse_matrix = vec.transform(corpus)
                        #doc_term_matrix = sparse_matrix.todense()
                        sim = cosine_similarity(sparse_matrix[0], sparse_matrix[1])
                        similarity[1].append(sim[0][0])
                
                x = np.asarray(data_cos[i]["jawaban_mahasiswa"])
                x_token = [[w for w in sentence.split(" ") if w != ""] for sentence in x] #split word into token
                x_tokenized.append(x_token)
                    

            save_object(self.tfidf_config.tfidf_obj_file_path, vec) #save object tfidf

            logging.info("object vec has been saved")
            data_set['train_set']['similarity'] = similarity[0]
            data_set['test_set']['similarity'] = similarity[1]
            return data_set, x_tokenized
          
        except Exception as e:
            raise CustomException(e, sys)

'''
    def label_tokenizer(self, data_set):
        try:
            logging.info("tokenize sentences")
            data_set_n = self.cosine_similarity(data_set)
            x_tokenized = []
            for i in data_set_n.keys():
                x = np.asarray(data_set_n[i]["jawaban_mahasiswa"])
                x_token = [[w for w in sentence.split(" ") if w != ""] for sentence in x] #split word into token
                x_tokenized.append(x_token)
            
            return x_tokenized
        
        except Exception as e:
            raise CustomException(e, sys)
'''
            
class Sequencer():

    def __init__(self,
                 all_words,
                 max_words,
                 seq_len,
                 embedding_matrix
                ):

        self.seq_len = seq_len
        self.embed_matrix = embedding_matrix
        """
        temp_vocab = Vocab which has all the unique words
        self.vocab = Our last vocab which has only most used N words.

        """
        temp_vocab = list(set(all_words))
        self.vocab = []
        self.word_cnts = {}
        """
        Now we'll create a hash map (dict) which includes words and their occurencies
        """
        for word in temp_vocab:
            # 0 does not have a meaning, you can add the word to the list
            # or something different.
            count = len([0 for w in all_words if w == word])
            self.word_cnts[word] = count
            counts = list(self.word_cnts.values())
            indexes = list(range(len(counts)))

        # Now we'll sort counts and while sorting them also will sort indexes.
        # We'll use those indexes to find most used N word.
        cnt = 0
        while cnt + 1 != len(counts):
            cnt = 0
            for i in range(len(counts)-1):
                if counts[i] < counts[i+1]:
                    counts[i+1],counts[i] = counts[i],counts[i+1]
                    indexes[i],indexes[i+1] = indexes[i+1],indexes[i]
                else:
                    cnt += 1

        for ind in indexes[:max_words]:
            self.vocab.append(temp_vocab[ind])

    def textToVector(self,text):
        # First we need to split the text into its tokens and learn the length
        # If length is shorter than the max len we'll add some spaces (100D vectors which has only zero values)
        # If it's longer than the max len we'll trim from the end.
        tokens = text.split()
        len_v = len(tokens)-1 if len(tokens) < self.seq_len else self.seq_len-1
        vec = []
        for tok in tokens[:len_v]:
            try:
                vec.append(self.embed_matrix[tok])
            except Exception as e:
                pass

        last_pieces = self.seq_len - len(vec)
        for i in range(last_pieces):
            vec.append(np.zeros(100,))

        return np.asarray(vec).flatten()

class WordEmbedding():
    def text2vec(self, token):
        try:
            model =  gensim.models.Word2Vec(token,
                    vector_size=100
                    # Size is the length of our vector.
                    )
            #model.save(os.path.join('artifacts','word2vec.model'))
            logging.info('model word2vec has been define')
            sequencer = Sequencer(all_words = [tk for seq in token for tk in seq],
              max_words = 1200,
              seq_len = 15,
              embedding_matrix = model.wv)
            
            x_vec = np.asarray([sequencer.textToVector(" ".join(seq)) for seq in token])
            return x_vec  

        except Exception as e:
            raise CustomException(e, sys)


class PCA_model():
    def __init__(self, x, component):
        self.x = x
        self.component = component

    def fit(self):
        pca_model = PCA(n_components= self.component)
        pca_model.fit(self.x)
        return 


if __name__=='__main__':
    #df = pd.read_csv(DataIngestionConfig.train_data_path)
    train_data = DataIngestionConfig().train_data_path
    test_data = DataIngestionConfig().test_data_path

    data = DataPreprocessing().initiate_data_preprocessing(train_data, test_data)
    data_proc = PreprocessingText().processing_text(data)
    data_sim, token = CreateFeature().similarity_tokenizer(data_proc)
    print(data_sim)
    print(token)
    """"
    x_vec = WordEmbedding().text2vec(token)
    print(x_vec)
    print(x_vec.shape)
    """
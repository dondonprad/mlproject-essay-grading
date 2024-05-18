import pandas as pd
import numpy as np

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

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class Preprocessing():
    def preprocessing_text(self, df):
        


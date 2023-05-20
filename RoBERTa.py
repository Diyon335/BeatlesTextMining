import numpy as np
import pandas as pd
import nltk
from transformers import RobertaTokenizerFast
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
import re
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
def prova():
    print("ciao")


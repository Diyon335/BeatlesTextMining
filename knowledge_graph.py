import nltk
import os
from nltk.sem import relextract
import networkx as nx
import matplotlib.pyplot as plt


data_folder = "data/resolved_corefs/Get_Back.txt"


def run_relation_extraction():

    with open(data_folder) as f:

        sentences = nltk.sent_tokenize(f.read())

        tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]

        pattern = r'(<PERSON>)+ was (<PERSON>)+'

        reldict = relextract.extract_rels(tagged_sentences)

        reldicts = []
        for sent in tagged_sentences:
            reldicts += relextract.extract_rels('PER', 'ORG', sent, pattern=pattern)

        for reldict in reldicts:
            print(reldict['subjtext'], reldict['filler'], reldict['objtext'])




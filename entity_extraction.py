from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import nltk
import os

data_folder = "data/pre_processed_data/sentences/all/"


def run_entity_extraction():

    entities = []

    for album in os.listdir(data_folder):
        for song in os.listdir(data_folder + album):

            with open(data_folder + album + "/" + song) as f:

                sentences = sent_tokenize(f.read())
                words = [word_tokenize(sentence) for sentence in sentences]

                tagged_words = [pos_tag(word) for word in words]

                for word in tagged_words:
                    named_entities = nltk.ne_chunk(word)

                    for entity in named_entities:

                        if entity[0] == "PERSON":
                            entities.append(entity)
                # for tagged_word in tagged_words:
                #     for word, tag in tagged_word:
                #         if tag == "NNP":
                #             entities.append(word)

    print(entities)

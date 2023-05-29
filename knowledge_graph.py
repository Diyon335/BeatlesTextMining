import stanza
from nltk.parse.stanford import StanfordParser
import stanza
import os
import networkx as nx
import matplotlib.pyplot as plt
from textblob import TextBlob
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

def prova1():
    stanza.download('en')
    nlp = stanza.Pipeline('en')

    # Input sentence
    sentence = "Luca eats fruit, Diyon eats meal, Francesca eats chocolate, Luca kills cats."

    # Parse the sentence
    doc = nlp(sentence)

    # Create a directed graph
    graph = nx.DiGraph()

    # Iterate through the words in the sentence
    for word in doc.sentences[0].words:
        # Check if the word is a verb
        if word.upos == "VERB":
            verb = word.lemma

            # Find the nominal subjects and objects related to the verb
            subjects = []
            objects = []
            for dep_word in doc.sentences[0].words:
                if dep_word.head == word.id:
                    if dep_word.deprel == "nsubj":
                        subjects.append(dep_word.text)
                    elif dep_word.deprel == "obj":
                        objects.append(dep_word.text)

            # Add subjects and objects as nodes and create edges between them if they are directly related
            for subj in subjects:
                graph.add_node(subj)
                if len(objects) == 1:
                    graph.add_node(objects[0])
                    graph.add_edge(subj, objects[0], label=verb)  # Add the verb as an attribute to the edge

            for obj in objects:
                graph.add_node(obj)

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=12,
                     font_weight='bold', edge_color='gray', arrows=True)

    # Add labels to the edges
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()
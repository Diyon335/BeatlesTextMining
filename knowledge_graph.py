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

def prova1():
    stanza.download('en')
    nlp = stanza.Pipeline('en')

    # Input sentence
    sentence = "He's a real nowhere man sitting in his nowhere land making all his nowhere plans for nobody."
    # Parse the sentence
    doc = nlp(sentence)

    # Create a directed graph
    graph = nx.DiGraph()

    # Iterate through the words in the sentence
    for word in doc.sentences[0].words:
        print(word)
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

def prova(dictionary):
    stanza.download('en')
    nlp = stanza.Pipeline('en')

    graph = nx.DiGraph()

    for key in dictionary:
        for sentence in dictionary[key]:
            doc = nlp(sentence)
            for word in doc.sentences[0].words:
                if word.upos == "VERB":
                    verb = word.lemma
                    subjects = []
                    objects = []
                    for dep_word in doc.sentences[0].words:
                        if dep_word.head == word.id:
                            if dep_word.deprel == "nsubj":
                                subjects.append(dep_word.text)
                            elif dep_word.deprel == "obj":
                                objects.append(dep_word.text)
                    for subj in subjects:
                        graph.add_node(subj)
                    for obj in objects:
                        graph.add_node(obj)
                    for element1, element2 in zip(subjects, objects):
                        graph.add_node(element1)
                        graph.add_node(element2)
                        graph.add_edge(element1, element2, label=verb)

                    for obj in objects:
                        graph.add_node(obj)

    # Draw the graph
    pos = nx.spring_layout(graph)
    nx.draw_networkx(graph, pos, with_labels=True, node_color='lightblue', node_size=100, font_size=6,
                     font_weight='bold', edge_color='gray', arrows=True)

    # Add labels to the edges
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()
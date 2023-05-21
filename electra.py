from transformers import pipeline
import os
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import networkx as nx


data_folder = "data/pre_processed_data/sentences/all/"
model_emotions = ['joy', 'love', 'anger', 'fear', 'surprise', 'sadness']
output_folder = "data/sentiment_cluster_diagrams/"


def run_classifier():

    classifier = pipeline("text-classification", model='bhadresh-savani/electra-base-emotion', return_all_scores=True)

    total_dict = {}

    for album in os.listdir(data_folder):

        album_dict = {}

        for song in os.listdir(data_folder + album):

            # Highest scoring sentiments for all sentences in the song
            song_sentiments = {}

            with open(data_folder + album + f"/{song}", "r") as f:

                sentence_tokens = sent_tokenize(f.read())

                for sentence in sentence_tokens:

                    prediction = classifier(sentence)

                    sentiments = prediction[0]

                    # Get the highest scoring sentiment for the sentence
                    max_score_sentiment = max(sentiments, key=lambda x: x['score'])
                    max_label = max_score_sentiment['label']
                    # max_score = max_score_sentiment['score']

                    if max_label not in song_sentiments:
                        song_sentiments[max_label] = 0

                    song_sentiments[max_label] += 1

            album_dict[song] = song_sentiments
            total_dict[song] = song_sentiments

        plot_custer_diagram(album, album_dict, output_folder + album + ".png")

    plot_custer_diagram("All songs", total_dict, output_folder + "all.png")


def plot_custer_diagram(title, song_dict, output_path):

    # Create an empty graph
    G = nx.Graph()

    final_emotions = []
    # Add nodes for each emotion
    for emotion in model_emotions:

        for song in song_dict.keys():

            found_emotion = False

            for emos in song_dict[song].keys():

                if emotion in emos:
                    final_emotions.append(emotion)
                    G.add_node(emotion)
                    break

            if found_emotion:
                break

    # Iterate over songs and emotions
    for song, emotions in song_dict.items():
        # Calculate the total count of emotions in the song
        total_emotions = sum(emotions.values())

        # Add edges from emotions to songs
        for emotion, count in emotions.items():
            # Calculate the normalized weight as the fraction of the emotion count in the song
            weight = count / total_emotions

            # Add edge with the normalized weight
            G.add_edge(emotion, song, weight=weight)

            # Set the 'title' attribute for song nodes
            G.nodes[song]['title'] = song.replace(".txt", "").replace("_", " ")

    # Set positions for the nodes in the clu-ster diagram
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=final_emotions, node_color='lightgray', node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=song_dict.keys(), node_color='skyblue', node_size=800)

    # Set node labels
    labels = {node: node if node in final_emotions else song for node, song in G.nodes(data='title')}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Remove edges
    plt.axis('off')
    plt.title(title)
    # Display the plot
    plt.savefig(output_path)
    plt.clf()

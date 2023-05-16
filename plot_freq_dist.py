from nltk import FreqDist
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

data_folder = "data/pre_processed_data/words/"
output_directory = "data/freq_dist_graphs/"
num_words = 20


def plot():
    """
    Plots the frequency distribution graphs for all songs

    :return: None
    """

    i = 0
    all_tokens = []
    for album in os.listdir(data_folder):

        album_tokens = []

        for song in os.listdir(data_folder+album):

            with open(data_folder+album+"/"+song, "r") as f:

                words_and_lem = f.readlines()

                words = [lem.split(":")[1].replace("\n", "") for lem in words_and_lem]

                all_tokens.extend(words)
                album_tokens.extend(words)

        album_freq_dist = FreqDist()

        for token in all_tokens:
            album_freq_dist[token.lower()] += 1

        save_plot(f"Word Frequency Distribution for {get_album_name(album)}",
                  output_directory+album+".png",
                  album_freq_dist,
                  num_words)

        # TODO REMOVE WHEN DOING ALL DATA
        if i == 1:
            break
        i += 1

    all_freq_dist = FreqDist()

    for token in all_tokens:
        all_freq_dist[token.lower()] += 1

    save_plot("Word Frequency Distribution for all Beatles' songs",
              output_directory+"all_songs.png",
              all_freq_dist,
              num_words)


def get_album_name(title):

    strings = re.findall('[A-Z][^A-Z]*', title)
    name = " ".join(strings)

    return name


def save_plot(title, file_path, freq_dist_dict, num_words):

    # Conversion to Pandas series via Python Dictionary for easier plotting
    all_fdist = pd.Series(dict(freq_dist_dict.most_common(num_words)))

    # Setting figure, ax into variables
    fig, ax = plt.subplots(figsize=(10, 10))

    # Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
    all_plot = sns.barplot(x=all_fdist.values, y=all_fdist.index, ax=ax, orient='horizontal')
    plt.xticks(rotation=30)
    plt.title(title)
    plt.savefig(file_path)



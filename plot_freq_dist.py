from nltk import FreqDist
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

data_folder = "data/pre_processed_data/words/"
output_directory = "data/freq_dist_graphs/"
num_words = 20

beatles = ["John", "Paul", "George", "Ringo"]


def plot():
    """
    Plots the frequency distribution graphs for all songs

    :return: None
    """

    all_tokens = []

    for album in os.listdir(data_folder + "all/"):

        album_tokens = []

        for song in os.listdir(data_folder + "all/" + album):

            with open(data_folder + "all/" + album + "/" + song, "r") as f:

                words_and_lem = f.readlines()

                # Get the lemmatized word
                words = [lem.split(":")[1].replace("\n", "") for lem in words_and_lem]

                all_tokens.extend(words)
                album_tokens.extend(words)

        album_freq_dist = FreqDist()

        for token in all_tokens:
            album_freq_dist[token.lower()] += 1

        album_string = album.split("_")[1]

        save_plot(f"Word Frequency Distribution for {get_album_name(album_string)}",
                  output_directory+album+".png",
                  album_freq_dist,
                  num_words)

    all_freq_dist = FreqDist()

    for token in all_tokens:
        all_freq_dist[token.lower()] += 1

    save_plot("Word Frequency Distribution for all Beatles' songs",
              output_directory+"all_songs.png",
              all_freq_dist,
              num_words)

    for beatle in beatles:

        for album in os.listdir(data_folder+f"{beatle}"):

            if not os.path.exists(output_directory+f"{beatle}"):
                os.makedirs(output_directory+f"{beatle}")

            freq_dist = FreqDist()
            album_tokens_beatle = []

            for song in os.listdir(data_folder+f"{beatle}/{album}"):

                with open(data_folder + f"{beatle}/{album}/" + song, "r") as f:

                    words_and_lem_beatle = f.readlines()

                    # Get the lemmatized word
                    words_beatle = [lem.split(":")[1].replace("\n", "") for lem in words_and_lem_beatle]

                    album_tokens_beatle.extend(words_beatle)

            for token in album_tokens_beatle:
                freq_dist[token.lower()] += 1

            album_string = album.split("_")

            save_plot(f"Word Frequency Distribution of {beatle} in {album_string[1]}",
                      output_directory+beatle+f"/{album}.png",
                      freq_dist,
                      20)


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



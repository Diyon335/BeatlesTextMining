import os
import shutil
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

data_folder = "data/raw_data/"
output_folder = "data/pre_processed_data/"

extra_stop_words = ["oh", "ah", "yeah", "heh", "whoa"]

def get_wordnet_pos(treebank_tag):
    """
    This function identifies if a word is a noun, adjective, verb or adverb

    :param treebank_tag: String indicating a word's treebank tag
    :return: Returns a wordnet object indicating if a word is a noun, adjective, verb or adverb
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def run_word_tokenize():
    """
    Tokenizes all the raw data (words and sentences) and saves them to text files

    :return: None
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Tokenizer type
    tokenizer = RegexpTokenizer(r"\b\w+['']\w+\b|\w+")
    lemmatiser = WordNetLemmatizer()

    # All stop words
    nltk_stop_words = set(stopwords.words("english"))

    for word in extra_stop_words:
        nltk_stop_words.add(word)

    # For each album
    for folder in os.listdir(data_folder):

        # For each song
        for text_file in os.listdir(data_folder+folder):

            # Read lyrics, tokenize words and sentences after removing stop words
            # Also applies lemmatisation
            with open(data_folder+folder+"/"+text_file) as f:

                beatle_folder = "_".join(f.readline().split(":")).replace("\n", "")
                lyrics = f.read()

                tokens = tokenizer.tokenize(lyrics)
                sentences = sent_tokenize(lyrics)

                filtered_tokens = [token.lower() for token in tokens if not token.lower() in nltk_stop_words]

                words = pos_tag(filtered_tokens)
                lemmatised_words = []

                for w in words:

                    wnet = get_wordnet_pos(w[1])

                    if wnet == '':
                        lemmatised_word = lemmatiser.lemmatize(w[0])
                    else:
                        lemmatised_word = lemmatiser.lemmatize(w[0], wnet)

                    lemmatised_words.append(lemmatised_word)

                # Saves into general folders and beatle-specific folders
                if not os.path.exists(output_folder+"words/all/"+folder):
                    os.makedirs(output_folder+"words/all/"+folder)

                if not os.path.exists(output_folder+"sentences/all/"+folder):
                    os.makedirs(output_folder+"sentences/all/"+folder)

                if not os.path.exists(output_folder+f"words/{beatle_folder}/{folder}"):
                    os.makedirs(output_folder+f"words/{beatle_folder}/{folder}")

                if not os.path.exists(output_folder+f"sentences/{beatle_folder}/{folder}"):
                    os.makedirs(output_folder+f"sentences/{beatle_folder}/{folder}")

                with open(output_folder+"sentences/all/"+folder+"/"+text_file, "w") as output_file:

                    for sentence in sentences:

                        output_file.write(f"{sentence}\n")

                shutil.copyfile(output_folder+"sentences/all/"+folder+"/"+text_file,
                                output_folder+f"sentences/{beatle_folder}/{folder}/{text_file}")

                with open(output_folder+"words/all/"+folder+"/"+text_file, "w") as output_file:

                    for word, lword in zip(words, lemmatised_words):

                        output_file.write(f"{word[0]}:{lword}\n")

                shutil.copyfile(output_folder + "words/all/" + folder + "/" + text_file,
                                output_folder + f"words/{beatle_folder}/{folder}/{text_file}")


def dict_creation():
    path = "data/pre_processed_data/sentences/all"
    song_dict = {}
    labels_dict = {}
    sentences = []
    labels = []
    for folder in os.listdir(path):
        for file in os.listdir(path + "/" + folder):
            if file == "Anna_(Go_To_Him).txt":
                with open(path + "/" + folder + "/" + file) as f:
                    lyrics = f.readlines()
                    lyrics = [lyric.replace("\n", "") for lyric in lyrics]
                    sentences = []
                    labels = []
                    for line in lyrics:
                        sentence = line.split("%")[0]
                        label = line.split("%")[1]
                        sentences.append(sentence)
                        labels.append(label)
                    song_dict[file.replace(".txt", "")] = sentences
                    labels_dict[file.replace(".txt", "")] = labels

    return song_dict, labels_dict
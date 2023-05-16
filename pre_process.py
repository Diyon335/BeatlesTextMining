import os
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
        os.mkdir(output_folder)

    # Tokenizer type
    tokenizer = RegexpTokenizer(r"\b\w+['']\w+\b|\w+")
    lemmatiser = WordNetLemmatizer()

    # All stop words
    nltk_stop_words = set(stopwords.words("english"))

    for word in extra_stop_words:
        nltk_stop_words.add(word)

    i = 0
    # For each album
    for folder in os.listdir(data_folder):

        # For each song
        for text_file in os.listdir(data_folder+folder):

            # Read lyrics, tokenize words and sentences after removing stop words
            # Also applies lemmatisation
            with open(data_folder+folder+"/"+text_file) as f:

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

                if not os.path.exists(output_folder+"words/"+folder):
                    os.mkdir(output_folder+"words/"+folder)

                if not os.path.exists(output_folder+"sentences/"+folder):
                    os.mkdir(output_folder+"sentences/"+folder)

                with open(output_folder+"sentences/"+folder+"/"+text_file, "w") as output_file:

                    for sentence in sentences:

                        output_file.write(f"{sentence}\n")

                with open(output_folder+"words/"+folder+"/"+text_file, "w") as output_file:

                    for word, lword in zip(words, lemmatised_words):

                        output_file.write(f"{word[0]}:{lword}\n")

        # TODO REMOVE WHEN WE WANT TO DO ALL DATA
        if i == 1:
            break

        i += 1

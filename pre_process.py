import os
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

data_folder = "data/raw_data/"
output_folder = "data/pre_processed_data/"


def get_wordnet_pos(treebank_tag):
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

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tokenizer = RegexpTokenizer(r"\b\w+['']\w+\b|\w+")

    nltk_stop_words = set(stopwords.words("english"))

    for folder in os.listdir(data_folder):

        for text_file in os.listdir(data_folder+folder):

            with open(data_folder+folder+"/"+text_file) as f:

                lyrics = f.read()

                tokens = tokenizer.tokenize(lyrics)
                sentences = sent_tokenize(lyrics)

                filtered_tokens = [token.lower() for token in tokens if not token.lower() in nltk_stop_words]

                words = pos_tag(filtered_tokens)
                lemmatised_words = []

                lemmatiser = WordNetLemmatizer()

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

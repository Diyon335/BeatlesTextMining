from fastcoref import FCoref
import nltk
from nltk.tag import pos_tag
import os
import re


data_folder = [
    "data/pre_processed_data/sentences/all/1970_LetItBe/Get_Back.txt",
    "data/pre_processed_data/sentences/all/1969_YellowSubmarine/Yellow_Submarine.txt",
    "data/pre_processed_data/sentences/all/1969_AbbeyRoad/Polythene_Pam.txt",
    "data/pre_processed_data/sentences/all/1969_AbbeyRoad/Mean_Mr_Mustard.txt",
    "data/pre_processed_data/sentences/all/1969_AbbeyRoad/Maxwells_Silver_Hammer.txt",
    "data/pre_processed_data/sentences/all/1969_AbbeyRoad/Her_Majesty.txt",
    "data/pre_processed_data/sentences/all/1968_TheBeatles/Sexy_Sadie.txt",
    "data/pre_processed_data/sentences/all/1968_TheBeatles/Martha_My_Dear.txt",
    "data/pre_processed_data/sentences/all/1967_SgtPeppers/Lucy_In_The_Sky_With_Diamonds.txt",
    "data/pre_processed_data/sentences/all/1967_SgtPeppers/Lovely_Rita.txt",
    "data/pre_processed_data/sentences/all/1967_SgtPeppers/Being_For_The_Benefit_Of_Mr_Kite.txt",
    "data/pre_processed_data/sentences/all/1967_MagicalMysteryTour/Penny_Lane.txt",
    "data/pre_processed_data/sentences/all/1965_RubberSoul/Nowhere_Man.txt"
]

output_dir = "data/resolved_corefs/"


def run_coref():

    model = FCoref()

    for song in data_folder:

        with open(song) as f:

            text = " ". join(f.readlines())

            preds = model.predict(
                texts=text
            )

            clusters = preds.get_clusters()
            tags = []
            for cluster in clusters:
                tags.append(pos_tag(cluster))

            print(text)
            print(preds.clusters)
            print(clusters)

            new_text = replace_pronouns_with_proper_nouns(text, clusters, tags)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            with open(output_dir + song.split("/")[-1], "w") as another_f:
                another_f.write(new_text)


def replace_pronouns_with_proper_nouns(text, clusters, pos_tags):
    # Convert the text to a list of words
    words = nltk.word_tokenize(text)

    # Iterate through the clusters and POS tags
    for cluster, pos_tag_list in zip(clusters, pos_tags):
        # print(f"cluster: {cluster}")
        # print(f"pos tag list: {pos_tag_list}")
        proper_noun = None
        pronouns = []

        # Iterate through the words in the cluster
        for word in cluster:
            # Find the corresponding POS tag for the current word in the cluster
            for token, pos in pos_tag_list:
                if token == word:
                    # Check if the word is a proper noun
                    if pos == 'NNP' or pos == 'NN':
                        proper_noun = word
                    # Check if the word is a pronoun
                    elif pos == 'PRP' or pos == 'PRP$':
                        pronouns.append(word)
                    break
        # print(f"Proper noun: {proper_noun}")
        # print(f"pronouns: {pronouns}")

        # Replace pronouns with the proper noun in the original text
        if proper_noun is not None:
            for pronoun in pronouns:
                # Use regular expression with word boundaries to match whole words
                pattern = r'\b' + re.escape(pronoun) + r'\b'
                text = re.sub(pattern, proper_noun, text)

    return text




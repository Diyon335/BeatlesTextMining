import t5_model
import knowledge_graph
import BERT
import voting_scheme
import nltk
import plot_freq_dist
import pre_process
import coref
import electra

"""
You can change what you want to run over here
"""
run_pre_processing = False
freq_dist = False
t5 = True
bert = False
run_coref = False
run_electra = False


def main():

    # Download necessary NLTK libraries
    nltk.download(["names",
                   "stopwords",
                   "state_union",
                   "twitter_samples",
                   "movie_reviews",
                   "averaged_perceptron_tagger",
                   "vader_lexicon",
                   "punkt"])

    if run_pre_processing:
        pre_process.run_word_tokenize()

    if freq_dist:
        plot_freq_dist.plot()

    if t5:
        sentences_dict, labels_dict = t5_model.dict_creation()
        print(sentences_dict)
        #dict = BERT.emotion_classification(sentences_dict)
        entities_list = BERT.pos_tagging(sentences_dict)
        print(entities_list)
        names_dict = BERT.extract_most_common_names(entities_list)
        print(names_dict)
        #dict = t5_model.sentences_emotion_classification(sentences_dict)
        #voting_scheme.vote(dict)
        #knowledge_graph.produce_graph(dict)

    if bert:
        BERT.emotion_classification()

    if run_coref:
        coref.run_coref()

    if run_electra:
        electra.run_classifier()


if __name__ == '__main__':
    main()

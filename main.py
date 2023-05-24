import t5_model
import knowledge_graph
import BERT
import voting_scheme
import nltk
import plot_freq_dist
import pre_process
import coref
import electra
import entity_extraction

"""
You can change what you want to run over here
"""
run_pre_processing = False
freq_dist = False
t5 = False
bert = False
run_coref = True
run_electra = False
run_ee = False


def main():

    # Download necessary NLTK libraries
    nltk.download(["names",
                   "stopwords",
                   "state_union",
                   "twitter_samples",
                   "movie_reviews",
                   "averaged_perceptron_tagger",
                   "vader_lexicon",
                   "punkt",
                   'maxent_ne_chunker',
                   "words"])

    if run_pre_processing:
        pre_process.run_word_tokenize()

    if freq_dist:
        plot_freq_dist.plot()

    if t5:
        sentences_dict, labels_dict = pre_process.dict_creation()
        #sentences_dict = coref.coreference(sentences_dict)
        BERT.fine_tune(sentences_dict, labels_dict)
        #dict = BERT.emotion_classification(sentences_dict)
        #print(dict)
        #entities_list = BERT.pos_tagging(sentences_dict)
        #names_dict = BERT.extract_most_common_names(entities_list)
        #dict = t5_model.sentences_emotion_classification(sentences_dict)
        #voting_scheme.vote(dict)
        #knowledge_graph.produce_graph(dict)

    if bert:
        BERT.emotion_classification()

    if run_coref:
        coref.run_coref()

    if run_electra:
        #electra.run_classifier()
        electra.fine_tune()

    if run_ee:
        entity_extraction.run_entity_extraction()


if __name__ == '__main__':
    main()

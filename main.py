import BERT_model
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
bert = False
run_coref = False
run_electra = False
run_ee = True


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

    if bert:
        dict = BERT_model.dict_creation()
        dict = BERT_model.sentences_emotion_classification(dict)
        voting_scheme.vote(dict)
        
    if run_coref:
        coref.run_coref()

    if run_electra:
        electra.run_classifier()

    if run_ee:
        entity_extraction.run_entity_extraction()


if __name__ == '__main__':
    main()

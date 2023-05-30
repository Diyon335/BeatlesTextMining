import BERT
import nltk
import plot_freq_dist
import pre_process
import coref
import electra
import knowledge_graph

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
                   "punkt",
                   'maxent_ne_chunker',
                   "words"])

    if run_pre_processing:
        pre_process.run_word_tokenize()

    if freq_dist:
        plot_freq_dist.plot()

    if bert:
        BERT.fine_tune()

    if run_coref:
        coref.run_coref()

    if run_electra:
        # electra.run_classifier()
        print("Electra before fine tuning")
        electra.test_fine_tuned(False)

        electra.fine_tune()

        print("Electra after fine tuning")
        electra.test_fine_tuned()

    if run_ee:
        sentences = pre_process.dict_creation()
        knowledge_graph.produce_knowledge_graph1(sentences)


if __name__ == '__main__':
    main()

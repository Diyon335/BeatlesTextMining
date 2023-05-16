import pre_process, plot_freq_dist
import nltk

"""
You can change what you want to run over here
"""
run_pre_processing = False
freq_dist = True


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


if __name__ == '__main__':
    main()

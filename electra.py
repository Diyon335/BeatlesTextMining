from transformers import pipeline


def run_classifier():
    classifier = pipeline("text-classification", model='bhadresh-savani/electra-base-emotion', return_all_scores=True)
    prediction = classifier("I really love bears!!")

    print(prediction[0])
    sentiments = prediction[0]

    max_score_sentiment = max(sentiments, key=lambda x: x['score'])
    max_label = max_score_sentiment['label']
    max_score = max_score_sentiment['score']

    print("Label with the maximum score:", max_label)
    print("Maximum score:", max_score)

    sorted_sentiments = sorted(sentiments, key=lambda x: x['score'], reverse=True)
    sorted_labels = [sentiment['label'] for sentiment in sorted_sentiments]

    total_list = [sentiment['score'] for sentiment in sorted_sentiments]

    print("sum: ", sum(total_list))

    print("Labels sorted by highest score:", sorted_labels)

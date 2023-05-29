from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForTokenClassification, \
    TokenClassificationPipeline,pipeline, RobertaTokenizerFast, TrainingArguments,\
    DataCollatorWithPadding, Trainer, AutoModelForSequenceClassification
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
import wandb
from transformers import pipeline, DataCollatorWithPadding
import os
from datasets import Dataset
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import networkx as nx
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, ElectraForSequenceClassification, ElectraTokenizer
import random
import numpy as np
import evaluate
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

classifier = pipeline("text-classification", model='tae898/emoberta-large', return_all_scores=True)

data_folder = "data/pre_processed_data/sentences/all/"
labelled_data = "data/BERT_labelled_data/"
model_emotions = ['joy', 'neutral', 'anger', 'fear', 'surprise', 'sadness', "disgust"]
output_folder = "data/BERT_sentiment_cluster_diagrams/"

def run_classifier():

    classifier = pipeline("text-classification", model='tae898/emoberta-large', return_all_scores=True)

    total_dict = {}

    for album in os.listdir(data_folder):

        album_dict = {}

        for song in os.listdir(data_folder + album):

            # Highest scoring sentiments for all sentences in the song
            song_sentiments = {}

            with open(data_folder + album + f"/{song}", "r") as f:

                sentence_tokens = sent_tokenize(f.read())

                for sentence in sentence_tokens:

                    prediction = classifier(sentence)

                    sentiments = prediction[0]

                    # Get the highest scoring sentiment for the sentence
                    max_score_sentiment = max(sentiments, key=lambda x: x['score'])
                    max_label = max_score_sentiment['label']
                    # max_score = max_score_sentiment['score']

                    if max_label not in song_sentiments:
                        song_sentiments[max_label] = 0

                    song_sentiments[max_label] += 1

            album_dict[song] = song_sentiments
            total_dict[song] = song_sentiments

        plot_custer_diagram(album, album_dict, output_folder + album + ".png")

    plot_custer_diagram("All songs", total_dict, output_folder + "all.png")

def extract_most_common_names(entities_list):
    final_list = []
    final_dict = {}
    for element in entities_list:
        if element[0] == "NNP":
            print(element[1])
            print(final_list)
            if element[1] not in final_list:
                final_dict[element[1]] = 0
                final_list.append(element[1])
            final_dict[element[1]] += 1
    return final_dict

def compute_metrics(eval_pred):

    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def fine_tune():
    dataset = {"train": [], "test": [], "validation": []}

    for data_type in os.listdir(labelled_data):

        for song in os.listdir(labelled_data + data_type):

            with open(labelled_data + data_type + "/" + song) as f:

                sentences = f.readlines()[1:]

                for sentence in sentences:
                    label_dict = {}

                    split_sentence = sentence.split("%")

                    text = split_sentence[0].replace("\n", "")
                    label = split_sentence[1].replace("\n", "")

                    label_dict["label"] = label
                    label_dict["text"] = text

                    dataset[data_type].append(label_dict)

    tokenizer = RobertaTokenizerFast.from_pretrained("tae898/emoberta-large")

    tokenized_datasets = {"train": [], "test": [], "validation": []}

    label_map = {
        "neutral": 0,
        "joy": 1,
        "surprise": 2,
        "anger": 3,
        "sadness": 4,
        "disgust": 5,
        "fear": 6
    }

    for data_type in dataset:

        for example in dataset[data_type]:
            tokenized_example = tokenizer(example["text"], padding="max_length", truncation=True)
            tokenized_example["label"] = label_map[example["label"]]
            tokenized_datasets[data_type].append(tokenized_example)

    random.shuffle(tokenized_datasets["train"])
    random.shuffle(tokenized_datasets["test"])
    random.shuffle(tokenized_datasets["validation"])

    model = AutoModelForSequenceClassification.from_pretrained('tae898/emoberta-large')

    training_args = TrainingArguments(
        output_dir="data/BERT_fine_tuned/",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=500,
        save_strategy="epoch"
    )

    wandb.login(key="f98f0b7ea0fe9cdabbe5e59c532b2260512d004b")

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    output_dir = "data/BERT_fine_tuned/"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def test_fine_tuned(after_fine_tune=True):

    if after_fine_tune:
        classifier = pipeline("text-classification", model='data/BERT_fine_tuned/', return_all_scores=True)
    else:
        classifier = pipeline("text-classification", model='tae898/emoberta-large', return_all_scores=True)

    test_data_directory = "data/BERT_labelled_data/test/"

    total = 0
    correct = 0

    for song in os.listdir(test_data_directory):

        with open(test_data_directory + song) as f:

            lines = f.readlines()

            for line in lines:

                split_line = line.split("%")
                text = split_line[0].replace("\n", "")
                label = split_line[1].replace("\n", "")

                prediction = classifier(text)

                sentiments = prediction[0]

                # Get the highest scoring sentiment for the sentence
                max_score_sentiment = max(sentiments, key=lambda x: x['score'])
                max_label = max_score_sentiment['label']

                total += 1

                if label == max_label:
                    correct += 1

    print(f"Accuracy: {correct/total}")

def plot_custer_diagram(title, song_dict, output_path):

    # Create an empty graph
    G = nx.Graph()

    final_emotions = []
    # Add nodes for each emotion
    for emotion in model_emotions:

        for song in song_dict.keys():

            found_emotion = False

            for emos in song_dict[song].keys():

                if emotion in emos:
                    final_emotions.append(emotion)
                    G.add_node(emotion)
                    break

            if found_emotion:
                break

    # Iterate over songs and emotions
    for song, emotions in song_dict.items():
        # Calculate the total count of emotions in the song
        total_emotions = sum(emotions.values())

        # Add edges from emotions to songs
        for emotion, count in emotions.items():
            # Calculate the normalized weight as the fraction of the emotion count in the song
            weight = count / total_emotions

            # Add edge with the normalized weight
            G.add_edge(emotion, song, weight=weight)

            # Set the 'title' attribute for song nodes
            G.nodes[song]['title'] = song.replace(".txt", "").replace("_", " ")

    # Set positions for the nodes in the clu-ster diagram
    pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=final_emotions, node_color='lightgray', node_size=2000)
    nx.draw_networkx_nodes(G, pos, nodelist=song_dict.keys(), node_color='skyblue', node_size=800)

    # Set node labels
    labels = {node: node if node in final_emotions else song for node, song in G.nodes(data='title')}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Remove edges
    plt.axis('off')
    plt.title(title)
    # Display the plot
    plt.savefig(output_path)
    plt.clf()

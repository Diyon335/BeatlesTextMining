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

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

classifier = pipeline("text-classification", model='tae898/emoberta-large', return_all_scores=True)

data_folder = "data/pre_processed_data/sentences/all/"
labelled_data = "data/BERT_labelled_data/"
model_emotions = ['joy', 'love', 'anger', 'fear', 'surprise', 'sadness']
output_folder = "data/sentiment_cluster_diagrams/"

def emotion_classification(dict):
    for key in dict:
        sentence_labels = []
        for item in dict[key]:
            prediction = classifier(item, )
            max = 0
            for element in prediction:
                for em in element:
                    if (em['score'] > max):
                        max = em['score']
                        emotion = em['label']
            sentence_labels.append(emotion)
            dict[key] = sentence_labels

    for song in dict:
        sentiment_dict = {}
        sent_list = dict[song]

        for sent in sent_list:

            if sent not in sentiment_dict:
                sentiment_dict[sent] = 0

            sentiment_dict[sent] += 1

        dict[song] = sentiment_dict
    return dict


def embeddings(dict):
    tokens = tokenizer.tokenize(dict["Anna_(Go_To_Him)"][0])
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = torch.tensor(token_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    last_hidden_state, pooler_output = model(token_ids, attention_mask=attention_mask).to_tuple()
    print(last_hidden_state)
    print(token_ids)
    print(attention_mask)


def pos_tagging(dict):
    temp = []
    final_list = []
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
    for key in dict:
        for item in dict[key]:
            outputs = pipeline(item)
            for element in outputs:
                temp.append(element["entity"])
                temp.append(element["word"])
    final_list = [temp[n:n + 2] for n in range(0, len(temp), 2)]
    return (final_list)


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

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

def fine_tune(sentence_dict, label_dict):
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
        "dissgust": 5,
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

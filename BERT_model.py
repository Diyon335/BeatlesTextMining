import os
from transformers import AutoTokenizer, AutoModelWithLMHead, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
import numpy as np
path = "data/pre_processed_data/sentences"

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

song_dict = {}
def dict_creation():
    for folder in os.listdir(path):
        for file in os.listdir(path + "/" + folder):
            with open(path + "/" + folder + "/" + file) as f:
                lyrics = f.readlines()
                lyrics = [lyric.replace("\n", "") for lyric in lyrics]
                song_dict[file.replace(".txt", "")] = lyrics

    return song_dict

def sentences_emotion_classification(dict):
    for key in dict:
        if key == "Because" or key == "Carry_That_Weight":
            sentence_labels = []
            for item in dict[key]:
                input_ids = tokenizer.encode(item + '</s>', return_tensors='pt')

                output = model.generate(input_ids=input_ids, max_length=2)

                dec = [tokenizer.decode(ids) for ids in output]
                label = dec[0]
                label = label.replace('<pad> ', '')
                sentence_labels.append(label)
                dict[key] = sentence_labels

    return(dict)
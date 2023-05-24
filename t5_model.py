import os
from transformers import AutoTokenizer, AutoModelWithLMHead, BertForSequenceClassification, BertTokenizerFast, Trainer, TrainingArguments
import torch
import numpy as np
import os
path = "data/pre_processed_data/sentences/all"

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")




def sentences_emotion_classification(dict):
    for key in dict:
        if key == "Anna_(Go_To_Him)":
            sentence_labels = []
            for item in dict[key]:
                input_ids = tokenizer.encode(item + '</s>', return_tensors='pt')

                output = model.generate(input_ids=input_ids, max_length=2)

                dec = [tokenizer.decode(ids) for ids in output]
                label = dec[0]
                label = label.replace('<pad> ', '')
                sentence_labels.append(label)
                dict[key] = sentence_labels

    for song in dict:
        sentiment_dict = {}
        sent_list = dict[song]

        for sent in sent_list:

            if sent not in sentiment_dict:
                sentiment_dict[sent] = 0

            sentiment_dict[sent] += 1

        dict[song] = sentiment_dict


    return(dict)

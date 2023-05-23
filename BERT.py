from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline, pipeline
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

classifier = pipeline("text-classification", model='tae898/emoberta-large', return_all_scores=True)

def emotion_classification(dict):
    for key in dict:
        if key == "Anna_(Go_To_Him)" or key == "Let_It_Be":
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
        if key == "Anna_(Go_To_Him)" or key == "Let_It_Be":
            for item in dict[key]:
                outputs = pipeline(item)
                for element in outputs:
                    temp.append(element["entity"])
                    temp.append(element["word"])
    final_list = [temp[n:n + 2] for n in range(0, len(temp), 2)]
    return(final_list)


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

def fine_tune(sentence_dict, label_dict):
    dataset = {}
    print(sentence_dict)
    print(label_dict)

    dataset = Dataset.from_dict(sentence_dict)
    print(dataset)
    #tokenized_datasets = sentence_dict.map(tokenize_function, batched=True)
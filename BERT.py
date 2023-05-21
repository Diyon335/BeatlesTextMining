from transformers import BertTokenizer, BertModel,AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
SEED = 1234

model_name = "QCRI/bert-base-multilingual-cased-pos-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
outputs = pipeline("A test example")
print(outputs)



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
max_length = 120

#config = AutoConfig.from_pretrained(bert-base-uncased)
#config.num_labels = 2

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def emotion_classificatio(dict):
    '''To implement'''

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
    list1 = []
    final_list = []
    list2 = []
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
    outputs = pipeline(dict["Anna_(Go_To_Him)"][0])
    for element in outputs:
        list1.append(element["entity"])
        list1.append(element["word"])
    final_list = [list1[n:n + 2] for n in range(0, len(list1), 2)]
    final_list
    print(final_list)

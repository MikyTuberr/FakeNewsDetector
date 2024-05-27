import pandas as pd
import numpy as np
import pickle
import os
from Plotter import Plotter
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import time

TRAIN_PATH = "../data/WELFake_Dataset.csv"
TEST_PATH = "../data/fake_or_real_news.csv"
SERIALIZED_PATH = "../serialized/bert_data.pkl"
PLOT_PATH = "../plots/bert.png"

if os.path.exists(SERIALIZED_PATH):
    print("Loading processed data...")
    with open(SERIALIZED_PATH, 'rb') as f:
        X_train, X_train_attention_mask, y_train, X_test, X_test_attention_mask, y_test = pickle.load(f)
else:
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def preprocess_text(title, text):
        if pd.isnull(title):
            title = ""
        if pd.isnull(text):
            text = ""
        combined_text = title + " " + text
        tokens = tokenizer.encode_plus(combined_text,
                                       add_special_tokens=True,
                                       max_length=128,
                                       padding='max_length',
                                       truncation=True,
                                       return_attention_mask=True,
                                       return_token_type_ids=False,
                                       return_tensors='pt')
        return tokens['input_ids'], tokens['attention_mask']


    X_train = np.vstack(train_data.apply(lambda row: preprocess_text(row['title'], row['text'])[0], axis=1))
    X_train_attention_mask = np.vstack(train_data.apply(lambda row: preprocess_text(row['title'], row['text'])[1], axis=1))
    y_train = train_data['label'].values

    X_test = np.vstack(test_data.apply(lambda row: preprocess_text(row['title'], row['text'])[0], axis=1))
    X_test_attention_mask = np.vstack(test_data.apply(lambda row: preprocess_text(row['title'], row['text'])[1], axis=1))
    y_test = test_data['label'].map({'FAKE': 0, 'REAL': 1}).values

    with open('../serialized/bert_data.pkl', 'wb') as f:
        pickle.dump((X_train, X_train_attention_mask, y_train, X_test, X_test_attention_mask, y_test), f)

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

X_train = X_train[:100]
X_train_attention_mask = X_train_attention_mask[:100]
y_train = y_train[:100]

X_test = X_test[:100]
X_test_attention_mask = X_test_attention_mask[:100]
y_test = y_test[:100]

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

num_epochs = 3
batch_size = 20
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.train()

start_time = time.time()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    for batch in range(0, len(X_train), batch_size):
        inputs = {
            'input_ids': torch.tensor(X_train[batch:batch+batch_size]).to(device),
            'attention_mask': torch.tensor(X_train_attention_mask[batch:batch+batch_size]).to(device),
            'labels': torch.tensor(y_train[batch:batch+batch_size]).to(device)
        }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"  Batch {batch//batch_size + 1}/{len(X_train)//batch_size}: Loss: {loss.item()}")

end_time = time.time()
training_time = end_time - start_time
print(f"[BERT] Training time: {training_time} seconds")

model.eval()
eval_loss = 0
predictions = []


start_time = time.time()

with torch.no_grad():
    for batch in range(0, len(X_test), batch_size):
        inputs = {
            'input_ids': torch.tensor(X_test[batch:batch+batch_size]).to(device),
            'attention_mask': torch.tensor(X_test_attention_mask[batch:batch+batch_size]).to(device),
            'labels': torch.tensor(y_test[batch:batch+batch_size]).to(device)
        }
        outputs = model(**inputs)
        logits = outputs.logits
        eval_loss += outputs.loss.item()

        predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())

end_time = time.time()
classifying_time = end_time - start_time
print(f"[BERT] Classifying time: {training_time} seconds")


eval_accuracy = accuracy_score(y_test, predictions)
print("[BERT] Accuracy:", eval_accuracy)

matrix = confusion_matrix(y_test, predictions)
print(matrix)

Plotter.show_confusion_matrix(matrix, PLOT_PATH)
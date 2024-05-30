import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from torch.nn import DataParallel

from Plotter import Plotter
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from torch.optim import AdamW
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
import time


class BERTClassifier:
    def __init__(self, data_path, serialized_path, plot_path, train_size, test_size):
        self.DATA_PATH = data_path
        self.SERIALIZED_PATH = serialized_path
        self.PLOT_PATH = plot_path

        if os.path.exists(self.SERIALIZED_PATH):
            print("Loading processed data...")
            with open(self.SERIALIZED_PATH, 'rb') as f:
                (self.X_train, self.X_train_attention_mask, self.y_train,
                 self.X_test, self.X_test_attention_mask, self.y_test) = pickle.load(f)
        else:
            self._load_and_process_data()

        self.X_train = self.X_train[:train_size]
        self.X_train_attention_mask = self.X_train_attention_mask[:train_size]
        self.y_train = self.y_train[:train_size]

        self.X_test = self.X_test[:test_size]
        self.X_test_attention_mask = self.X_test_attention_mask[:test_size]
        self.y_test = self.y_test[:test_size]

        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        self.loss_history = []

    def _load_and_process_data(self):
        data = pd.read_csv(self.DATA_PATH)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def preprocess_text(title, text):
            combined_text = title if pd.isnull(text) else text if pd.isnull(title) else title + " " + text
            tokens = tokenizer.encode_plus(combined_text,
                                           max_length=512,
                                           padding='max_length',
                                           truncation=True,
                                           add_special_tokens=True,
                                           return_attention_mask=True,
                                           return_token_type_ids=False,
                                           return_tensors='pt')
            return tokens['input_ids'], tokens['attention_mask']

        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

        self.X_train = np.vstack(train_data.apply(lambda row: preprocess_text(row['title'], row['text'])[0], axis=1))
        self.X_train_attention_mask = np.vstack(
            train_data.apply(lambda row: preprocess_text(row['title'], row['text'])[1], axis=1))
        self.y_train = train_data['label'].values

        self.X_test = np.vstack(test_data.apply(lambda row: preprocess_text(row['title'], row['text'])[0], axis=1))
        self.X_test_attention_mask = np.vstack(
            test_data.apply(lambda row: preprocess_text(row['title'], row['text'])[1], axis=1))
        self.y_test = test_data['label'].values

        with open(self.SERIALIZED_PATH, 'wb') as f:
            pickle.dump((self.X_train, self.X_train_attention_mask, self.y_train,
                         self.X_test, self.X_test_attention_mask, self.y_test), f)

        print("Shape of X_train:", self.X_train.shape)
        print("Shape of X_test:", self.X_test.shape)
        print("Shape of y_train:", self.y_train.shape)
        print("Shape of y_test:", self.y_test.shape)

    def train(self, num_epochs=3, batch_size=32):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = DataParallel(self.model)

        self.model.to(device)
        self.model.train()

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0  # add this line to calculate epoch loss
            for batch in range(0, len(self.X_train), batch_size):
                inputs = {
                    'input_ids': torch.tensor(self.X_train[batch:batch + batch_size]).to(device),
                    'attention_mask': torch.tensor(self.X_train_attention_mask[batch:batch + batch_size]).to(device),
                    'labels': torch.tensor(self.y_train[batch:batch + batch_size]).to(device)
                }

                self.optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs.loss
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                print(f"  Batch {batch // batch_size + 1}/{len(self.X_train) // batch_size}: Loss: {loss.item()}")

            self.loss_history.append(epoch_loss)

        end_time = time.time()
        training_time = end_time - start_time
        print(f"[BERT] Training time: {training_time} seconds")

        Plotter.plot_loss(self.loss_history, self.PLOT_PATH + "_loss")

    def evaluate(self, batch_size=32):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        eval_loss = 0
        predictions = []

        start_time = time.time()

        with torch.no_grad():
            for batch in range(0, len(self.X_test), batch_size):
                inputs = {
                    'input_ids': torch.tensor(self.X_test[batch:batch + batch_size]).to(device),
                    'attention_mask': torch.tensor(self.X_test_attention_mask[batch:batch + batch_size]).to(device),
                    'labels': torch.tensor(self.y_test[batch:batch + batch_size]).to(device)
                }
                outputs = self.model(**inputs)
                logits = outputs.logits
                eval_loss += outputs.loss.item()

                predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())

        end_time = time.time()
        classifying_time = end_time - start_time
        print(f"[BERT] Classifying time: {classifying_time} seconds")

        eval_accuracy = accuracy_score(self.y_test, predictions)
        print("[BERT] Accuracy:", eval_accuracy)

        matrix = confusion_matrix(self.y_test, predictions)
        print(matrix)

        Plotter.show_confusion_matrix(matrix, self.PLOT_PATH)



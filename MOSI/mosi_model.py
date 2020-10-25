import numpy as np
import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.attention import Attention
from modules.coattention import Coattention
import time
import json, os, ast, h5py

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
import sys

print(torch.__version__)

def load_saved_data():

    h5f = h5py.File('X_train.h5', 'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('y_train.h5', 'r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('X_valid.h5', 'r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('y_valid.h5', 'r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('X_test.h5', 'r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('y_test.h5', 'r')
    y_test = h5f['data'][:]
    h5f.close()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


class Net(nn.Module):
    def __init__(self,config):
        super(Net, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config['input_dims']
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]

        self.final_dims = config["final_dims"]

        self.h_dim = config["h_dim"]

        self.wordLSTM = nn.LSTM(self.d_l, self.dh_l, bidirectional=False)
        self.covarepLSTM = nn.LSTM(self.d_a, self.dh_a, bidirectional=False)
        self.facetLSTM = nn.LSTM(self.d_v, self.dh_v, bidirectional=False)

        self.coattention = Coattention(self.dh_l, self.dh_a)
        self.coattention1 = Coattention(self.dh_l, self.dh_v)
        self.coattention2 = Coattention(self.dh_v, self.dh_a)

        self.aggregateLSTM = nn.LSTM(self.dh_a*2 + self.dh_v * 2 + self.dh_l, self.h_dim, bidirectional=False, batch_first=True)

        self.dropout1 = nn.Dropout(config["dropout1"])
        self.dropout2 = nn.Dropout(config["dropout2"])

        self.wordLSTMlinear = nn.Linear(self.dh_l, self.dh_l)
        self.wordLSTMfinal = nn.Linear(self.dh_l, 1)

        self.attention = Attention(self.dh_l)
        self.outputlinear = nn.Linear(self.h_dim*2, self.final_dims)
        self.finallinear = nn.Linear(self.final_dims, 1)
        #
    def squash(self, x, dim=-1):
        squared = torch.sum(x * x, dim=dim, keepdim=True)
        scale = torch.sqrt(squared) / (1.0 + squared)
        return scale * x

    def forward(self, x):
        x_l = x[:, :, :self.d_l] #20 686 300
        x_a = x[:, :, self.d_l:self.d_a+self.d_l] #20 686 5
        x_v = x[:, :, self.d_a+self.d_l:] #20 686 20

        word_LSTM, h = self.wordLSTM(x_l)
        covarep_LSTM, h1 = self.covarepLSTM(x_a)
        facet_LSTM, h2 = self.facetLSTM(x_v)

        # print(lens)
        word_LSTM = word_LSTM.permute(1, 0, 2)
        covarep_LSTM = covarep_LSTM.permute(1, 0, 2)
        facet_LSTM = facet_LSTM.permute(1, 0, 2)

        coatt_LA = self.coattention(covarep_LSTM, word_LSTM)
        coatt_LV = self.coattention1(facet_LSTM, word_LSTM)

        coatt_LAV = torch.cat([coatt_LA, coatt_LV, word_LSTM], dim=2)

        aggregate_LSTM, h3 = self.aggregateLSTM(coatt_LAV)

        #word-level classification
        word_LSTM = self.dropout1(word_LSTM)
        word_LSTM_linear = self.wordLSTMlinear(word_LSTM).squeeze()
        word_LSTM_linear = self.squash(word_LSTM_linear)
        word_LSTM_final = self.wordLSTMfinal(word_LSTM_linear)
        #print(word_LSTM_final.size())

        #sentiment classification
        LSTM_att, self.weight = self.attention(word_LSTM_linear, aggregate_LSTM)
        fusion = torch.mean(aggregate_LSTM, dim=1)
        self.fusion = torch.cat([fusion, LSTM_att], dim=-1)
        final = F.relu(self.outputlinear(self.fusion))
        final = self.dropout2(final)
        final = self.finallinear(final)
        # print(word_LSTM_final.squeeze(1).size())
        return final.squeeze(1), word_LSTM_final.squeeze(2)

def train_net(X_train, y_train, y_train_sentiment,  X_valid, y_valid, y_valid_sentiment, X_test, y_test, y_test_sentiment, config):
    torch.manual_seed(111)
    model = Net(config)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])

    criterion = nn.L1Loss()
    criterion1 = nn.BCELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    criterion1 = criterion1.to(device)

    def train(model, batchsize, X_train, y_train, y_train_sentiment,  optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[1]
        num_batches = int(total_n / batchsize) + 1
        for batch in range(num_batches):
            start = batch * batchsize
            end = (batch + 1) * batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[:, start:end])
            batch_y = torch.Tensor(y_train[start:end])
            batch_y_s = torch.Tensor(y_train_sentiment[start:end])
            mask = torch.BoolTensor(train_mask[start:end])
            predictions, predictions1 = model.forward(batch_X)
            loss = criterion(predictions, batch_y)
            predictions1 = F.sigmoid(predictions1)
            loss1 = criterion1(predictions1, batch_y_s)
            loss1 = loss1.masked_select(mask).mean()
            loss_total = (1 - config['a']) * loss + config['a'] * loss1
            loss_total.backward()
            # loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            epoch_loss += loss_total.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion, batchsize=64):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            total_n = X_valid.shape[1]
            num_batches = int(total_n / batchsize) + 1
            for batch in range(num_batches):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X = torch.Tensor(X_valid[:, start:end])
                batch_y = torch.Tensor(y_valid[start:end])
                predictions, _ = model.forward(batch_X)
                loss = criterion(predictions, batch_y)
                epoch_loss += loss.item()
        return epoch_loss / num_batches

    def predict(model, X_test, batchsize=64):
        batch_preds = []
        model.eval()
        with torch.no_grad():
            total_n = X_test.shape[1]
            num_batches = int(total_n / batchsize) + 1
            for batch in range(num_batches):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X = torch.Tensor(X_test[:, start:end])
                predictions, _ = model.forward(batch_X)
                predictions = predictions.cpu().data.numpy()
                batch_preds.append(predictions)
            batch_preds = np.concatenate(batch_preds, axis=0)
        return batch_preds

    # timing
    start_time = time.time()

    end_time = time.time()
    print(end_time - start_time)

    best_valid = 999999.0

    for epoch in range(config["num_epochs"]):
        train_loss = train(model,config["batchsize"],X_train, y_train, y_train_sentiment, optimizer, criterion)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        # scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            torch.save(model, 'mosi_model.pt')
        else:
            print(epoch, train_loss, valid_loss)

    model = torch.load('mosi_model.pt', map_location='cpu')

    predictions = predict(model, X_test)

    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
    print("MAE: ", np.mean(np.absolute(predictions - y_test)))
    print("Corr: ", np.corrcoef(predictions, y_test)[0][1])
    mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    print("7-class: ", mult)
    print(config)
    sys.stdout.flush()

if __name__ == '__main__':

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_saved_data()

    y_train_sentiment = np.load('y_train_sentiment.npy')
    y_valid_sentiment = np.load('y_valid_sentiment.npy')
    y_test_sentiment = np.load('y_test_sentiment.npy')

    train_mask = np.load('train_mask.npy')
    valid_mask = np.load('valid_mask.npy')
    test_mask = np.load('test_mask.npy')

    #
    X_train = X_train.swapaxes(0, 1)
    X_valid = X_valid.swapaxes(0, 1)
    X_test = X_test.swapaxes(0, 1)

    #
    print(X_train.shape)
    print(y_train.shape)
    print(X_valid.shape)
    print(y_valid.shape)
    print(X_test.shape)
    print(y_test.shape)

    config = dict()
    config["input_dims"] = [300, 5, 20]
    hl = 100
    ha = 50
    hv = 30

    config["h_dims"] = [hl, ha, hv]

    config["final_dims"] = 100
    config["batchsize"] = 16
    config["num_epochs"] = 20
    config["lr"] = 0.0006
    config["h_dim"] = 128
    config['dropout1'] = 0.5
    config['dropout2'] = 0.2
    config['a'] = 0.3

    train_net(X_train, y_train, y_train_sentiment, X_valid, y_valid, y_valid_sentiment, X_test, y_test,
              y_test_sentiment, config)



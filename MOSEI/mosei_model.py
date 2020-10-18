import numpy as np
# from process_data import *

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from modules.coattention import Coattention
from modules.attention import Attention
import time
import json, os, ast, h5py


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys


def load_saved_data():
    h5f = h5py.File('text_train_emb.h5', 'r')
    X_train_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('video_train.h5', 'r')
    X_train_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('audio_train.h5', 'r')
    X_train_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('y_train.h5', 'r')
    y_train = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('text_valid_emb.h5', 'r')
    X_valid_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('video_valid.h5', 'r')
    X_valid_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('audio_valid.h5', 'r')
    X_valid_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('y_valid.h5', 'r')
    y_valid = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('text_test_emb.h5', 'r')
    X_test_emb = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('video_test.h5', 'r')
    X_test_vedio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('audio_test.h5', 'r')
    X_test_audio = h5f['d1'][:]
    h5f.close()

    h5f = h5py.File('y_test.h5', 'r')
    y_test = h5f['d1'][:]
    h5f.close()

    new_y_train, new_y_valid, new_y_test = [], [], []
    for i, x in enumerate(X_train_emb):

        if y_train[i] > 0:
            new_y_train.append(2)
        elif y_train[i] == 0:
            new_y_train.append(1)
        else:
            new_y_train.append(0)

    for i, x in enumerate(X_valid_emb):

        if y_valid[i] > 0:
            new_y_valid.append(2)
        elif y_valid[i] == 0:
            new_y_valid.append(1)
        else:
            new_y_valid.append(0)


    for i, x in enumerate(X_test_emb):

        if y_test[i] > 0:
            new_y_test.append(2)
        elif y_test[i] == 0:
            new_y_test.append(1)
        else:
            new_y_test.append(0)
            # s.append(sum_)

    print(y_test[:20])
    print(len(X_train_emb) + len(X_valid_emb) + len(X_test_emb))

    return X_train_emb, X_train_vedio, X_train_audio, np.array(new_y_train), \
          X_valid_emb, X_valid_vedio, X_valid_audio, np.array(new_y_valid), \
           X_test_emb, X_test_vedio, X_test_audio, np.array(new_y_test)

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

        self.aggregateLSTM = nn.LSTM(self.dh_a * 2 + self.dh_v * 2 + self.dh_l, self.h_dim, bidirectional=False,
                                     batch_first=True)

        self.dropout1 = nn.Dropout(config["dropout1"])
        self.dropout2 = nn.Dropout(config["dropout2"])

        self.wordLSTMlinear = nn.Linear(self.dh_l, self.dh_l)
        self.wordLSTMfinal = nn.Linear(self.dh_l, 1)

        self.attention = Attention(self.dh_l)
        self.outputlinear = nn.Linear(self.h_dim * 2, self.final_dims)
        self.finallinear = nn.Linear(self.final_dims, 3)
        #

    def squash(self, x, dim=-1):
        squared = torch.sum(x * x, dim=dim, keepdim=True)
        scale = torch.sqrt(squared) / (1.0 + squared)
        return scale * x

    def forward(self, x_l, x_v, x_a):
        word_LSTM, h = self.wordLSTM(x_l)
        covarep_LSTM, h1 = self.covarepLSTM(x_a)
        facet_LSTM, h2 = self.facetLSTM(x_v)

        word_LSTM = word_LSTM.permute(1, 0, 2)
        covarep_LSTM = covarep_LSTM.permute(1, 0, 2)
        facet_LSTM = facet_LSTM.permute(1, 0, 2)

        coatt_LA = self.coattention(covarep_LSTM, word_LSTM)
        coatt_LV = self.coattention1(facet_LSTM, word_LSTM)

        coatt_LAV = torch.cat([coatt_LA, coatt_LV, word_LSTM], dim=2)

        aggregate_LSTM, h3 = self.aggregateLSTM(coatt_LAV)

        # word-level classification
        word_LSTM = self.dropout1(word_LSTM)
        word_LSTM_linear = self.wordLSTMlinear(word_LSTM).squeeze()
        word_LSTM_linear = self.squash(word_LSTM_linear)
        word_LSTM_final = self.wordLSTMfinal(word_LSTM_linear)
        # print(word_LSTM_final.size())

        # sentiment classification
        LSTM_att, _ = self.attention(word_LSTM_linear, aggregate_LSTM)
        fusion = torch.mean(aggregate_LSTM, dim=1)
        self.fusion = torch.cat([fusion, LSTM_att], dim=-1)
        final = F.relu(self.outputlinear(self.fusion))
        final = self.dropout2(final)
        final = self.finallinear(final)
        # print(word_LSTM_final.squeeze(1).size())
        return final.squeeze(1), word_LSTM_final.squeeze(2)


def train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, X_valid_emb, X_valid_vedio, X_valid_audio,
              y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config):

    torch.manual_seed(111)  #
    model = Net(config)

    # optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    criterion1 = criterion1.to(device)

    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5, verbose=True)

    def train(model, batchsize, X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train_emb.shape[1]
        num_batches = total_n / batchsize
        for batch in range(int(num_batches)):
            start = batch * batchsize
            end = (batch + 1) * batchsize
            optimizer.zero_grad()
            batch_X_embed = torch.Tensor(X_train_emb[:, start:end])
            batch_X_v = torch.Tensor(X_train_vedio[:, start:end])
            batch_X_a = torch.Tensor(X_train_audio[:, start:end])
            batch_y = torch.LongTensor(y_train[start:end])
            batch_y_s = torch.Tensor(y_train_sentiment[start:end])
            mask = torch.BoolTensor(train_mask[start:end])
            predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
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

    def evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion, batchsize=64):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            total_n = X_valid_emb.shape[1]
            num_batches = total_n / batchsize
            for batch in range(int(num_batches) + 1):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X_embed = torch.Tensor(X_valid_emb[:, start:end])
                batch_X_v = torch.Tensor(X_valid_vedio[:, start:end])
                batch_X_a = torch.Tensor(X_valid_audio[:, start:end])
                batch_y = torch.LongTensor(y_valid[start:end])
                predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
                loss = criterion(predictions, batch_y).item()
                epoch_loss += loss
        return epoch_loss

    def predict(model, X_test_emb, X_test_vedio, X_test_audio, batchsize=64):
        batch_preds = []
        model.eval()
        with torch.no_grad():
            total_n = X_test_emb.shape[1]
            num_batches = total_n / batchsize
            for batch in range(int(num_batches) + 1):
                start = batch * batchsize
                end = (batch + 1) * batchsize
                batch_X_embed = torch.Tensor(X_test_emb[:, start:end])
                batch_X_v = torch.Tensor(X_test_vedio[:, start:end])
                batch_X_a = torch.Tensor(X_test_audio[:, start:end])

                predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
                predictions = F.softmax(predictions, 1)
                predictions = predictions.cpu().data.numpy()
                batch_preds.append(predictions)
            batch_preds = np.concatenate(batch_preds, axis=0)
        return batch_preds


    # timing
    start_time = time.time()
    predictions = predict(model, X_test_emb, X_test_vedio, X_test_audio)

    print(predictions.shape)
    end_time = time.time()
    print(end_time - start_time)

    best_valid = 999999.0
    rand = random.randint(0, 100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model,config["batchsize"], X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, optimizer, criterion)
        valid_loss = evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion)
        # scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            torch.save(model, 'mosei_model.pt')
        else:
            print(epoch, train_loss, valid_loss)
    print('rand:',rand)
    model = torch.load('mosei_model.pt', map_location='cpu')
    predictions = predict(model, X_test_emb, X_test_vedio, X_test_audio)

    true_label = y_test
    predicted_label = np.argmax(predictions, axis=1)

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("3-Accuracy ", accuracy_score(true_label, predicted_label))
    print("3-F1 ", f1_score(true_label, predicted_label, average='weighted'))
    # print("MAE: ", np.mean(np.absolute(predictions - y_test)))
    # print("Corr: ", np.corrcoef(predictions, y_test)[0][1])
    # mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    # print("7-class: ", mult)
    print(config)
    sys.stdout.flush()

if __name__ == '__main__':
    X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, X_test_emb, \
    X_test_vedio, X_test_audio, y_test = load_saved_data()

    y_train_sentiment = np.load('y_train_sentiment.npy')
    y_valid_sentiment = np.load('y_valid_sentiment.npy')
    y_test_sentiment = np.load('y_test_sentiment.npy')

    train_mask = np.load('train_mask.npy')
    valid_mask = np.load('valid_mask.npy')
    test_mask = np.load('test_mask.npy')

    #
    X_train_emb = X_train_emb.swapaxes(0, 1)
    X_valid_emb = X_valid_emb.swapaxes(0, 1)
    X_test_emb = X_test_emb.swapaxes(0, 1)

    print(X_train_vedio.shape)
    print(X_valid_emb.shape)
    print(X_test_emb.shape)
    print(y_train.shape)

    X_train_vedio = X_train_vedio.swapaxes(0, 1)
    X_valid_vedio = X_valid_vedio.swapaxes(0, 1)
    X_test_vedio = X_test_vedio.swapaxes(0, 1)

    X_train_audio = X_train_audio.swapaxes(0, 1)
    X_valid_audio = X_valid_audio.swapaxes(0, 1)
    X_test_audio = X_test_audio.swapaxes(0, 1)

    config = dict()
    config["input_dims"] = [300, 74, 35]
    hl = 128
    ha = 20
    hv = 10

    config["h_dims"] = [hl, ha, hv]

    config["final_dims"] = 100
    config["batchsize"] = 64
    config["num_epochs"] = 8
    config["lr"] = 0.0004
    config["h_dim"] = 100
    config['dropout1'] = 0.3
    config['dropout2'] = 0.3

    config['a'] = 0.25
    train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, X_valid_emb, X_valid_vedio, X_valid_audio,
              y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config)


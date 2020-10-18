import numpy as np
# from process_data import *

import random
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import json, os, ast, h5py
from modules.attention import Attention
from modules.coattention import Coattention

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys

print(torch.__version__)

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
    y_train_onehot = h5f['d1'][:]
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
    y_valid_onehot = h5f['d1'][:]
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
    y_test_onehot = h5f['d1'][:]
    h5f.close()

    print(X_train_audio.shape, X_train_vedio.shape, y_train_onehot.shape)

    y_train = np.argmax(y_train_onehot, axis=1)
    y_valid = np.argmax(y_valid_onehot, axis=1)
    y_test = np.argmax(y_test_onehot, axis=1)
    print(y_train[:10])
    return X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, \
           X_test_emb, X_test_vedio, X_test_audio, y_test, y_train_onehot, y_valid_onehot, y_test_onehot

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

        self.attention = Attention(self.h_dim)
        self.outputlinear = nn.Linear(self.h_dim*2, self.final_dims)
        self.finallinear = nn.Linear(self.final_dims, 3)

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
        LSTM_att, _ = self.attention(aggregate_LSTM, aggregate_LSTM)
        fusion = torch.mean(aggregate_LSTM, dim=1)
        self.fusion = torch.cat([fusion, LSTM_att], dim=-1)
        final = F.relu(self.outputlinear(self.fusion))
        final = self.dropout2(final)
        final = self.finallinear(final)
        # print(word_LSTM_final.squeeze(1).size())
        return final.squeeze(1), word_LSTM_final.squeeze(2)
        # return final.squeeze(1), None


def train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, X_valid_emb, X_valid_vedio, X_valid_audio,
              y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config):
    torch.manual_seed(111)
    model = Net(config)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    # optimizer = optim.RMSprop(model.parameters(), lr=config["lr"])

    criterion = nn.CrossEntropyLoss()
    criterion1 = nn.BCELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    criterion1 = criterion1.to(device)

    def train(model, batchsize, X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment,  optimizer, criterion):
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
            # mask = torch.BoolTensor(train_mask[start:end])
            predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            loss = criterion(predictions, batch_y)
            predictions1 = F.sigmoid(predictions1)
            loss1 = criterion1(predictions1, batch_y_s)
            # loss1 = loss1.masked_select(mask).mean()
            loss_total = (1 - config['a']) * loss + config['a'] * loss1
            loss_total.backward()
            # loss.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            epoch_loss += loss_total.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion):

        model.eval()
        with torch.no_grad():
            batch_X_embed = torch.Tensor(X_valid_emb)
            batch_X_v = torch.Tensor(X_valid_vedio)
            batch_X_a = torch.Tensor(X_valid_audio)
            batch_y = torch.LongTensor(y_valid)
            # batch_y_s = torch.Tensor(y_valid_sentiment)
            predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            loss = criterion(predictions, batch_y)
            # predictions1 = F.sigmoid(predictions1)
            # loss1 = criterion1(predictions1, batch_y_s)
            # loss_total = (1 - config['a']) * loss + config['a'] * loss1
            epoch_loss = loss.item()
        return epoch_loss

    def predict(model, X_test_emb, X_test_vedio, X_test_audio):

        model.eval()
        with torch.no_grad():
            batch_X_embed = torch.Tensor(X_test_emb)
            batch_X_v = torch.Tensor(X_test_vedio)
            batch_X_a = torch.Tensor(X_test_audio)
            predictions, predictions1 = model.forward(batch_X_embed, batch_X_v, batch_X_a)
            predictions = F.softmax(predictions, 1)
            predictions1 = F.sigmoid(predictions1)
            predictions = predictions.cpu().data.numpy()
            predictions1 = predictions1.cpu().data.numpy()
        return predictions, predictions1

    # timing
    start_time = time.time()
    predictions, predictions1 = predict(model, X_test_emb, X_test_vedio, X_test_audio)

    print(predictions.shape)
    end_time = time.time()
    print(end_time - start_time)

    best_valid = 999999.0

    for epoch in range(config["num_epochs"]):
        train_loss = train(model,config["batchsize"],X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, optimizer, criterion)
        valid_loss = evaluate(model, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, criterion)
        # scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            torch.save(model, 'youtube_model.pt')
        else:
            print(epoch, train_loss, valid_loss)

    model = torch.load('youtube_model.pt', map_location='cpu')

    predictions, predictions1 = predict(model, X_test_emb, X_test_vedio, X_test_audio)

    true_label = y_test
    predicted_label = np.argmax(predictions, axis=1)

    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    print("F1 ", f1_score(true_label, predicted_label, average='weighted'))
    # print("MAE: ", np.mean(np.absolute(predictions - y_test)))
    # print("Corr: ", np.corrcoef(predictions, y_test)[0][1])
    # mult = round(sum(np.round(predictions) == np.round(y_test)) / float(len(y_test)), 5)
    # print("7-class: ", mult)
    print(config)
    sys.stdout.flush()

if __name__ == '__main__':
    X_train_emb, X_train_vedio, X_train_audio, y_train, X_valid_emb, X_valid_vedio, X_valid_audio, y_valid, X_test_emb, \
    X_test_vedio, X_test_audio, y_test, y_train_onehot, y_valid_onehot, y_test_onehot = load_saved_data()
    y_train_sentiment = np.load('y_train_sentiment.npy')
    y_valid_sentiment = np.load('y_valid_sentiment.npy')
    y_test_sentiment = np.load('y_test_sentiment.npy')

    train_mask = np.load('train_mask.npy')
    valid_mask = np.load('valid_mask.npy')
    test_mask = np.load('test_mask.npy')

    X_train_emb = X_train_emb.swapaxes(0, 1)
    X_valid_emb = X_valid_emb.swapaxes(0, 1)
    X_test_emb = X_test_emb.swapaxes(0, 1)

    X_train_vedio = X_train_vedio.swapaxes(0, 1)
    X_valid_vedio = X_valid_vedio.swapaxes(0, 1)
    X_test_vedio = X_test_vedio.swapaxes(0, 1)

    X_train_audio = X_train_audio.swapaxes(0, 1)
    X_valid_audio = X_valid_audio.swapaxes(0, 1)
    X_test_audio = X_test_audio.swapaxes(0, 1)

    #
    print(X_train_emb.shape)
    print(X_valid_audio.shape)
    print(X_test_emb.shape)
    print(y_train.shape)

    config = dict()
    config["input_dims"] = [300, 74, 36]
    hl = 100
    ha = 20
    hv = 20

    config["h_dims"] = [hl, ha, hv]

    config["final_dims"] = 200
    config["batchsize"] = 16
    config["num_epochs"] = 25
    config["lr"] = 0.0005
    config["h_dim"] = 100
    config['dropout1'] = 0.5
    config['dropout2'] = 0.1

    config['a'] = 0.25

    train_net(X_train_emb, X_train_vedio, X_train_audio, y_train, y_train_sentiment, X_valid_emb, X_valid_vedio,
              X_valid_audio,
              y_valid, X_test_emb, X_test_vedio, X_test_audio, y_test, config)




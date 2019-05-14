#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
from random import shuffle
from itertools import cycle
from scipy import interp
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import pickle
import argparse
import sys
import os


from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import rbf_kernel
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import balanced_accuracy_score

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from IPython.display import display
#get_ipython().run_line_magic('matplotlib', 'inline')

class SequenceDLModelling:

    def __init__(self, in_df_path=None, out_df_path=None, max_length=None, embedding_length=None):
        self.in_df_path       = in_df_path
        self.out_df_path      = out_df_path
        self.max_length       = max_length
        self.embedding_length = embedding_length

    def getData(self, path):
        print("path: ", path)
        df= pd.read_csv(path)
        print(df.shape)
        return df

    def splitData(self, df, path):
        # don't forget to shuffle the data first
        features = df[df.columns[0:-1]].values
        y = df['label'].values
        x_train, x_test, y_train, y_test = train_test_split(
            features, y, test_size=0.2, random_state=1000)

        return  x_train, x_test, y_train, y_test

    def computeDescriptors(self, df):

        sentences = df['composition'].values
        y = df['label'].values

        tokenizer = Tokenizer(num_words=539, filters=",")
        tokenizer.fit_on_texts(sentences)

        X_sentences = tokenizer.texts_to_sequences(sentences)

        vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

        #print(len(tokenizer.word_counts))
        #print((tokenizer.word_index))
        #tokenizer.word_counts
        print("vocab_size: ", vocab_size)
        print("Example: ", sentences[2])
        print("Example: ", X_sentences[2])

        ###### Padding ##########
        X_padded = pad_sequences(X_sentences, padding='post', maxlen=self.max_length)
        df_out = pd.DataFrame(X_padded)
        print("df_out: ", df_out.shape)
        #df_out['composition'] = sentences
        df_out['label'] = y
        print("df_out: ", df_out.shape)
        return df_out, vocab_size

    def computeDLModel(self, X_train, X_test, y_train, y_test, out_path, suffix, vocab_size):
        # create the model
        model = Sequential()
        model.add(Embedding(vocab_size, self.embedding_length, input_length=self.max_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        history=model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
        if (suffix==1): suffix="reg"
        else: suffix = "rand"
        model.save(out_path + "DNN_model_" + suffix)
        return history, scores

    def plot_history(self, history, model_path,  suffix):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(x, acc, 'b', label='Training acc')
        plt.plot(x, val_acc, 'r', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(x, loss, 'b', label='Training loss')
        plt.plot(x, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        if (suffix==1): suffix="reg"
        else: suffix = "rand"
        plt.savefig(model_path + "DL_DNN_" + suffix +".png", dpi=400, format='png')


    def predict_hidden_testSet(self, X_test, path, class_names):
        loaded_model = load_model(path + "DNN_model_reg")
        return loaded_model.predict_classes(X_test)


    def prepOutputFolder(self, out_df_path, max_length, embedding_length):
            path = out_df_path
            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)

            path = out_df_path + "maxLen_" + str(max_length) +  "_embedLen_" + str(embedding_length) + "/"

            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
            return path


    def runMM(self):
        print("Setting output Folder")
        out_path = self.prepOutputFolder(self.out_df_path, self.max_length, self.embedding_length)
        print("done")
        print()
        print("Reading input file")
        df = self.getData(self.in_df_path)
        print("done")
        print()
        print("Computing descriptors")
        df_desc, vocab_size = self.computeDescriptors(df)
        print("done")
        print()
        print("Splitting data into trainingset and test")
        x_train, x_test, y_train, y_test = self.splitData(df_desc, out_path,)
        print("done")
        print()
        print("Building models and running internal 5 FCV")
        history, scores = self.computeDLModel(x_train, x_test, y_train, y_test, out_path, 1, vocab_size)
        print("Accuracy_regular: %.2f%%" % (scores[1]*100))
        self.plot_history(history, out_path, 1)
        print("done")
        print()
        print("Predict hidden test set")
        class_names=['Antibiotic', "Not_Antibiotic"]
        y_pred = self.predict_hidden_testSet(x_test, out_path , class_names)
        print("done")
        print()
        print("Y-Randomized Models")
        print('############ Model Building #######')
        shuffle(y_train)
        history_rand, scores_rand = self.computeDLModel( x_train, x_test, y_train, y_test, out_path, 0, vocab_size)
        print("Accuracy_random: %.2f%%" % (scores_rand[1]*100))
        self.plot_history(history_rand, out_path, 0)
        return history, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-in','--in_df_path',
                        help='Path of input file path.',  required=True, default=None)
    parser.add_argument('-o', '--out_df_path',
                        help='Path to new .csv file for saving the potential peptide dataframe.',  required=True, default=None)
    parser.add_argument('-ml', '--max_length', type=int,
                        help='determine the maximum length of the input vector.',  required=True, default=None)
    parser.add_argument('-el', '--embedding_length', type=int,
                        help='determine the length of the enbedding vector.',  required=True, default=None)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    mm = SequenceDLModelling(   in_df_path       = args.in_df_path,
                                out_df_path      = args.out_df_path,
                                max_length       = args.max_length,
                                embedding_length = args.embedding_length)
    mm.runMM()

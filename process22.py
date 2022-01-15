# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:11:24 2018

@author: lenovo-pc
"""

import os
import re
import numpy as np

from gensim.models import FastText
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense,Dropout
from keras.layers import Bidirectional,Flatten,RepeatVector,Activation
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
#from keras.layers import Permute

from random import shuffle

maxlen=300
window_size=5
batch_size=32
n_epoch=2
input_length=600

#Accessing the dataset and appending them to lists, also preprocessing at the same time
def init_lists(folder):
    var=1
    a_list=[]
    file_list=os.listdir(folder)
    for file in file_list:
        print(str(var)+'\n')
        var=var+1
        f=open(os.path.join(folder,file),encoding="utf-8")
        text=f.read()
        words=re.split(r'\s+',re.sub(r'[,/\-!?.I?"\]\[<>]', ' ',text).strip())
        for word in words:
            if word == "br":
                word = ""
            word = word.lower()
        a_list.append(words)
    f.close()
    return a_list

pos_train=init_lists("pos_train")
neg_train=init_lists("neg_train")
pos_test=init_lists("pos_test")
neg_test=init_lists("neg_test")
unsup=init_lists("unsup")
total = pos_train + neg_train + pos_test + neg_test + unsup
train = pos_train + neg_train
test = pos_test + neg_test
mya1 = np.zeros(12500)
mya2 = np.ones(12500)
label_train = np.append(mya2, mya1)
label_test = np.append(mya2, mya1)

t = Tokenizer(lower=True, split=' ', char_level=False, oov_token=None)
t.fit_on_texts(train)
encoded_docs = t.texts_to_sequences(train)
padded_docs = sequence.pad_sequences(encoded_docs, maxlen = 600, padding = 'post')

#Training a Fasttext Model
model=FastText(size=300,alpha=0.025,window=5,min_count=1,workers=4)
model.build_vocab(train + unsup)
model.train(train + unsup,total_examples=model.corpus_count,epochs=model.iter)
model.save("FastText_Eng_2.bin")

model=FastText.load("FastText_Eng_2.bin")

vocab_size = len(t.word_index) + 1 #114153
print(vocab_size)
embedding_matrix = np.zeros((vocab_size, 300))
for word,i in t.word_index.items():
    try:
        embedding_vector = model.wv[word]
    except:
        print(word, 'not found')
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


print('Defining a Simple Keras Model...')
lstm_model=Sequential()  # or Graph 
lstm_model = Sequential()
lstm_model.add(Embedding(output_dim=300,input_dim=vocab_size,
                    weights=[embedding_matrix],input_length=input_length)) 
lstm_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
lstm_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Adding Input Length

lstm_model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping=EarlyStopping(monitor='val_loss',min_delta=0,patience=0,
                                    verbose=0,mode='auto')

print("Train...")

t2 = Tokenizer(lower=True, split=' ', char_level=False, oov_token=None)
t2.fit_on_texts(test)

encoded_docs2 = t2.texts_to_sequences(test)

padded_docs2 = sequence.pad_sequences(encoded_docs2, maxlen = 600, padding = 'post')

print(padded_docs.size / 600)
print(label_train.size)
lstm_model.fit(padded_docs, label_train,batch_size=batch_size,epochs=20,
          validation_data = (padded_docs2, label_test),callbacks=[earlyStopping])

print("Evaluate...")
score,acc=lstm_model.evaluate(padded_docs2,label_test,batch_size=batch_size)
print('Test score:',str(score*100))
print('Test accuracy:',str(acc*100))
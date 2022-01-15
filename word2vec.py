import os
from gensim.models import Word2Vec
from cltk.tokenize.sentence import TokenizeSentence
import multiprocessing
import numpy as np

np.random.seed(0)
# accessing each file and add them to list
def init_lists(folder):
    print('starting')
    a_list=[]
    file_list=os.listdir(folder)
    for file in file_list:
	    f=open(os.path.join(folder,file),encoding="utf-8")
	    text=f.read()
	    text=text.replace(' <br /> <br />','')
	    tokenizer=TokenizeSentence('hindi')
	    tokenized_text=tokenizer.tokenize(text)   #Tokenization
	    a_list.append(tokenized_text)
    f.close()
    return a_list
   
# All hindi, bengali, english datasets
pos_train1=init_lists("pos_train")
neg_train1=init_lists("neg_train")
pos_test1=init_lists("pos_test")
neg_test1=init_lists("neg_test")
unsup1=init_lists("unsup")

#Combining all the words together.
all_reviews=pos_train1+neg_train1+pos_test1+neg_test1+unsup1+pos_train2+neg_train2+pos_test2+neg_test2+unsup2+pos_train3+neg_train3+pos_test3+neg_test3+unsup3

#Training FastText model
cpu_count=multiprocessing.cpu_count()
model=Word2Vec(size=300,window=5,min_count=1,alpha=0.025,workers=cpu_count,max_vocab_size=None,negative=10)
model.build_vocab(all_reviews)
model.train(all_reviews,total_examples=model.corpus_count,epochs=model.iter)
model.save("word2vec_hindi_50.bin")
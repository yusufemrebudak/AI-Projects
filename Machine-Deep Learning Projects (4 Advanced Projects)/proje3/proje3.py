import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences  # girdi olacak olan cümlelerdeki boyutları fikslemek.
from keras.models import Sequential # to create sequential model
from keras.layers.embeddings import Embedding  # integırları belirli boyutlarda yoğunluk vektörlerine çevirecek.  
from keras.layers import SimpleRNN, Dense, Activation

import warnings
warnings.filterwarnings("ignore")

#%%
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
   path = "imdb.npz", # npz numpy ın zipli hali
   num_words= None, # kaç
   skip_top = 0, # en sık kullanılan kelimeleri ignore edilip edilemeyeceğini belirtir.
   maxlen = None, # yorumlar birden fazla kelime olabilir, herhangi bir kelime sınırı, tüm yorumları istiyorum
   seed = 113, # aynı shuffle ile alabilmek için
   start_char = 1,  # dökümantasyonda bu şekilde yapılması isteniyor.
   oov_char = 2,  # dökümantasyonda bu şekilde yapılması isteniyor.
   index_from = 3  # dökümantasyonda bu şekilde yapılması isteniyor.
   ) 

print("Type: ", type(X_train))
print("Type: ", type(Y_train))

print("X train shape: ",X_train.shape)
print("Y train shape: ",Y_train.shape)

#%% EDA

print("Y train values: ",np.unique(Y_train)) 
print("Y test values: ",np.unique(Y_test))
# Y train values:  [0 1]
# Y test values:  [0 1]

unique, counts = np.unique(Y_train, return_counts=True)
print("Y train distribution",dict(zip(unique,counts))) # Y train distribution {0: 12500, 1: 12500}

unique, counts = np.unique(Y_test, return_counts=True)
print("Y test distribution",dict(zip(unique,counts))) # Y train distribution {0: 12500, 1: 12500}
plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

d = X_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []
for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))

sns.distplot(review_len_train, hist_kws = {"alpha":0.3})
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})

print("Train mean",np.mean(review_len_train)) # Train mean 238.71364
print("Train median",np.median(review_len_train)) # Train median 238.71364
print("Train mod",stats.mode(review_len_train)) # Train median 238.71364


# number of words 
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index)) # 88584

for keys, values in word_index.items():
    if values == 126:
        print(keys)
        
reverse_index = dict( [(value, key) for (key, value) in word_index.items()] )

def whatItSay(index = 24):
    reverse_index = dict( [(value, key) for (key, value) in word_index.items()] )
    decode_review = " ".join([reverse_index.get( i - 3 , "!") for i in X_train[index]])
    
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatItSay()


#%%PreProcess

num_words = 15000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words = num_words) # 15000 en sık kullanılan kelime 


max_len = 130
X_train = pad_sequences(X_train, maxlen = max_len)
X_test = pad_sequences(X_test, maxlen = max_len)

print(len(X_train[5]))
for i in X_train[0:10]:
    print(len(i))

decoded_review = whatItSay(5)

#%% RNN
rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length=len(X_train[0]))) # integırlara belirli boyutlarda yoğunluk vektörlerine çevirmeyi sağlar
# num_words -> input dimension
# output dimention -> 32
rnn.add(SimpleRNN(16,input_shape=(num_words,max_len), return_sequences = False, activation="relu")) 
# input_shape=(num_words,max_len) 15000 kelimem olucak ve bu review ların içindeki max uzunluk 130 olucak diyorum
rnn.add(Dense(1)) # flatten layer ekliyorum bunu dense layer ile yapıyorum
rnn.add(Activation("sigmoid")) # sigmoid for binary classification
print(rnn.summary())
rnn.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics = ["accuracy"])

#%%rnn fit

history = rnn.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=5,batch_size = 128, verbose=1)

#%% evaluation
score = rnn.evaluate(X_test,Y_test)
print("accuracy: ", score[1]*100) # accuracy:  86.14400029182434
plt.figure()

plt.plot(history.history["accuracy"],label="Train")
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()


plt.figure()

plt.plot(history.history["loss"],label="Train")
plt.plot(history.history["val_loss"], label="Test")
plt.title("Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()


















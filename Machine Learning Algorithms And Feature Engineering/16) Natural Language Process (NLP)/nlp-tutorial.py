
import pandas as pd
#%% import data 
data = pd.read_csv(r"gender-classifier.csv",encoding = "latin1")
data = pd.concat([data.gender,data.description],axis=1)
# önce data cleaning  yapmak lazım nan lar ve saçma ifadeler var
data.dropna(axis=0,inplace=True) # axis=0 demek yani drop et bunu axis=0 olarak yani satırı sil 
# inplace=True doğrudan data variable ının içine set ediyor değişiklikleri
# classificiation string sevmez kadınları 1, erkekleri 0 yapalım.
data.gender=[1 if each=="female" else 0 for each in data.gender]
# twitter datamız var, bir describtion yazmışlar, bunu öğreticez modelimize,
# yeni bir desc verdiğimizde bunu kadın mı erkek mi yazmış bunu tahmin edecek.
#%% cleaning data
# regular expression RE
# RE -> textin içinde farklı karakterleri bulmamızı sağlayan expressionlara regular expression denir.
import re

first_description = data.description[4]
description =  re.sub("[^a-zA-Z]"," ",first_description) # a dan z ye olan harfleri alma diğerlerini al
# ve o ifadeleri " " boşluk ile değiştir.
# 2. işlem data hazırlama, tam cleaning gibi değil, tüm harfleri küçük harfe çevireceğim
description = description.lower() # preprocess gibi tüm karakterleri küçük harfe çevirdik
#%% stopwords (irrevelant words) gereksiz kelimeler
# I go to the school and home -> mesela burada the gereksizdir, grammerle alakalıdır.
# yani benim bunu kadın mı erkek mi yazdı bunu tahmin etmem de bana hiç bir faydası yoktur.
# and de aynı şekilde çıkartmamız lazım datadan. bize kadın veya erkek olması hakkında bilgi vermez.
import nltk
nltk.download("stopwords") # gereksiz kelimeleri verecek, corpus diye bir classore indiricek bilgisayarımdaki.
from nltk.corpus import stopwords # sonra ben corpus klasörümden import ediyorum

# bu text i string string (kelime-kelime) ayırmak istiyorum.
# description2 = description.split()# kelimelerine ayırıyor 
# split yerine tokenizer da kullanabiliriz.

description = nltk.word_tokenize(description)
#%% gereksiz kelimeleri çıkar
description = [word for word in description if not word in set(stopwords.words("english"))]

# lemmatazation ,   loved=>love   gitmeyeceğim=>git, kelimelerin köklerini bulmalıyız

import nltk as nlp
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description] # her kelimemi verip bunların kökünü buluyorum
description = " ".join(description)

#%% 
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description) # a dan z ye olan harfleri alma diğerlerini al
    description = description.lower()
    description = nltk.word_tokenize(description) # cümleleri parçaladık ve kellimeleri oluşturduk
    # description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description] # her kelimemi verip bunların kökünü buluyorum
    description = " ".join(description)
    description_list.append(description)
#%% bag of words (kelimelerin çantası)
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak için kullandığım metodum 
# toplam 16000 tane sample ım var her sample da birbirinden farklı farklı kelimeler de olsa 
# toplamda binlerce kelime ortaya çıkabilir .max_features 500 demek 
# en sık kullanılan 500 kelimeyi feature olarak kullanıcam demek
max_features = 25000
# normalde 29144 tane kelime var :)
count_vectorizer = CountVectorizer(max_features=max_features,stop_words="english")
# ingilizce de gereksiz kelimeleri buluyor  ve CountVectorizer metodunu uygularken 
# gereksiz kelimeleri cümlelerden çıkarıp atacak.

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() 
# bütüm sample larımın içindeki cümlelerdeki kelimelerin matris olarak ifadesi gibidir.
print("en sik kullanilan {} kelimeler: {}".format(max_features,count_vectorizer.get_feature_names()))
#%%
# sparce_matrix bizim datamız yani x imiz
x = sparce_matrix
y = data.iloc[:,0].values # kadın ve erkekleri alıyorum
# train test split 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)

# %%naive bayes modeli kullanacağız
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
# prediction
y_pred= nb.predict(x_test)
# y_pred = y_pred.reshape(-1,1)
print("acc: {}".format(nb.score(y_pred.reshape(-1,1),y_test)))




















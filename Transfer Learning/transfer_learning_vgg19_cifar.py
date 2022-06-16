# from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical # classları categorical hale getirmek için
from keras.datasets import cifar10 # 0 dan 9 a kadar farklı farklı classlara ait resimler var
import cv2 # 
import numpy as np

#  you can download the dataset on "https://www.kaggle.com/datasets/moltean/fruits"


#%%
(x_train,y_train),(x_test,y_test) =   cifar10.load_data()
#%%
print("x_train shape:", x_train.shape)
numberOfClass = 10
y_train = to_categorical(y_train,numberOfClass)
y_test = to_categorical(y_test,numberOfClass)

input_shape=x_train.shape[1:]
#%% visualize
plt.imshow(x_train[35].astype(np.uint8))
plt.axis("off")
plt.show()
#%% increase dimension

def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

x_train = resize_img(x_train)
x_test = resize_img(x_test)
#%%
print("increased dim x_train: ",x_train.shape)
plt.figure()
plt.imshow(x_train[5511].astype(np.uint8))
plt.axis("off")
plt.show()
#%%vgg19

vgg = VGG19(include_top = False, weights = "imagenet",input_shape=(48,48,3))
# weights="imagenet" ne demek;
# bu VGG19 modeli bir transfer learning modeli vee bir yerde eğitilmiştir
# weights="imagenet" demek eğitildiği yer ise "imagenet" dataseti olsun diyorum
# include_top = False demek ise fully connected layer ım eklenmesin diyorum,
# modeli bana orası eksik ver diyorum.
print(vgg.summary())

vgg_layer_list = vgg.layers
print(vgg_layer_list)
#%% 
model = Sequential()
for layer in vgg_layer_list:
    model.add(layer)
    
print(model.summary())

for layer in model.layers:
    layer.trainable = False

# fully con layers
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(numberOfClass, activation= "softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])
hist = model.fit(x_train, y_train, validation_split = 0.2, epochs = 2, batch_size=500)
# Model, eğitim verilerinin bu kısmını ayıracak, 
#  eğitim olarak kullanmaycak ve her epoch sonunda bu veriler ile 
# loss ve diğer metrikleri hesaplayacak. 



#%%  model save
model.save_weights("example.h5")

#%%
print(hist.history.keys())

plt.plot(hist.history["loss"], label = "train loss")
plt.plot(hist.history["val_loss"], label = "val loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "train acc")
plt.plot(hist.history["val_accuracy"], label = "val acc")
plt.legend()
plt.show()

#%% load
import json, codecs
with codecs.open("deneme5.json","r",encoding = "utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["accuracy"], label = "train acc")
plt.plot(n["val_accuracy"], label = "val acc")
plt.legend()
plt.show()


#%% save
with open('deneme5.json', 'w') as f:
    json.dump(hist.history, f)
























    
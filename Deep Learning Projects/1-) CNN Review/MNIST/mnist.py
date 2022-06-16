
from keras.models import Sequential # sırali sequential demek, layerları eklicem işte 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten,Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import pandas as pd

# you can download the dataset on "https://www.kaggle.com/competitions/digit-recognizer"

def load_and_preprocess(data_path):
    data = pd.read_csv(data_path)
    data = data.to_numpy()
    np.random.shuffle(data)
    x = data[:,1:].reshape(-1,28,28,1)/255.0
    y = data[:,0].astype(np.int32)
    y = to_categorical(y,num_classes=len(set(y)))
    return x,y
    

data = pd.read_csv("mnist_train.csv") # (60000,785)



train_data_path = "mnist_train.csv"
test_data_path = "mnist_test.csv"

x_train, y_train = load_and_preprocess(train_data_path)
x_test, y_test = load_and_preprocess(test_data_path)
#%% visualization
index = 2122
vis = x_train.reshape(60000,28,28)
plt.imshow(vis[index,:,:])
plt.legend()
plt.axis("off")
plt.show()
print(np.argmax(y_train[index]))
#%% CNN
number_of_class = y_train.shape[1]
model = Sequential()
model.add(Conv2D(filters = 16,kernel_size= (3,3), input_shape = (28,28,1) )) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Conv2D(filters = 64,kernel_size= (3,3) )) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Conv2D(filters = 128,kernel_size= (3,3) )) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Flatten())
model.add(Dense(units=512))
model.add(Activation("relu"))
model.add(Dropout(0.25)) 
model.add(Dense(units=number_of_class)) # output layer
model.add(Activation("softmax"))


model.compile(loss = "categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"]) # softmax kullandığım ve multiclassım olduğu için bunu kullanıyorum.
hist = model.fit(x_train,y_train, validation_data=(x_test,y_test),epochs = 30, batch_size = 4000)
# bir batchde 4000 tane train edilsin diyorum toplamda, epoch basına batch sayısı 15 çıkar çünkü toplamda 60000 tane resmim var.
#%% 
model.save_weights("cnn_mnist_model.h5")
#%%
print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"],label="Train acc")
plt.plot(hist.history["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()


#%% save history
import json
with open('cnn_mnist_model.json','w') as f:
    json.dump(hist.history,f)
    
#%%
import codecs
with codecs.open("cnn_mnist_model.json","r",encoding='utf-8') as f:
    h = json.loads(f.read())
    
plt.figure()
plt.plot(h["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(h["accuracy"],label="Train acc")
plt.plot(h["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()


# predictions1 = model.predict(x_test)
# predictions2 = model.predict(x_test).max(axis=1)

score = model.evaluate(x_test, y_test, verbose = 0)
print("Test Loss : %f \nTest Accuracy : %f "%(score[0],score[1]))















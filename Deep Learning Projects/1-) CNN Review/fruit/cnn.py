from keras.models import Sequential # sırali sequential demek, layerları eklicem işte 
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import matplotlib.pyplot as plt
from glob import glob


#  you can download the dataset on "https://www.kaggle.com/datasets/moltean/fruits"


train_path = "fruits-360/Training/"
test_path =  "fruits-360/Test/"
img = load_img(train_path+"Apple Braeburn/0_100.jpg")
#plt.imshow(img)
#plt.axis("off")
# plt.show()
x = img_to_array(img)
print(x.shape)
className = glob(train_path+'/*') # train_path içine gir herhangi bir dosyanın hepsini al.
number_of_class = len(className)

#%% CNN Model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape= x.shape )) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Conv2D(32, (3,3))) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Conv2D(64, (3,3) )) # 32 filtre olsun, yani 32 feature map olsun
model.add(Activation("relu"))
model.add(MaxPooling2D()) # default olarak (2,2)

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5)) # 1024 taneden yüzden 50 sini kapatıyorum,
# her seferinde 512 tanesini aktif oluyor.  Overfitting i engellemek için 
model.add(Dense(number_of_class)) # output layer
model.add(Activation("softmax"))


model.compile(loss = "categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"]) # softmax kullandığım ve multiclassım olduğu için bunu kullanıyorum.
batch_size = 32 # batch size da belirlediğimiz sayı kadar resmi train edicez.her bir iterasyonda 32 resim train edilecek.
#%% 
# her bir nesne için bende 490-500 tane img falan var, bu çok az sayıdır bunun için 
# data augmentain yapmamız lazım yani farklı rotasyon ve kaydırmalarla resimlerimin sayılarını 
# yani farklı farklı elmalar muzlar yaratıcaz, böylece data sayımız artıcak ve model daha iyi öğrencek.
train_datagen = ImageDataGenerator(rescale=1./255,
                   shear_range=0.3, # shear_range -> resmi çevirmek gibi bişey
                    horizontal_flip = True,
                    zoom_range=0.3)
# train datam için kullanılacak bir imagegenerator yarattım.
test_datagen =ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical", # birden fazla class ım var demek. 
    )
# Found 67692 images belonging to 131 classes.
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=x.shape[:2],
    batch_size=batch_size,
    color_mode = "rgb",
    class_mode = "categorical", # birden fazla class ım var demek. 
    )
# Found 22688 images belonging to 131 classes.
##########
# fit etmek istediğimiz dataseti içeren generator 'train_generator'
hist = model.fit_generator(
    generator = train_generator,
    steps_per_epoch = 1600 // batch_size,#epoch başına yapılması gereken batch sayısı 50 taneymiş. 
    # 1600 burada resim sayısını ifade eder. bir class için 1600 resim olsun diyorum. 
    # ama bu kadar resim yok 50*batch_size = 1600 eder zaten bende 500 tane falan resim var
    # geri kalan resimlerde generatorlar ile çalıştığımızdan generator bize sağlar. 
    # 1600/batch_size = 50
    # diyorumki bu 32 farklı farklı paketleri 50 kere train et,
    validation_data = test_generator,
    epochs=10,
    validation_steps=800//batch_size)
#%%
model.save_weights("deneme.h5")
#%% model evaluation
print(hist.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])
# val_loss ve val_accuracy test üzerinde, loss ve accuracy benim train set üzerinde elde ettiğim sonuçlar. 
# loss ve accuracy model her eğitim iterayonunda hesaplanır, val olanlar epoch bitiminde train üzerinde test edilir. 
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"],label="Train acc")
plt.plot(hist.history["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()
#%% save history, kaydedip 
import json 
with open("deneme.json","w") as f:
    json.dump(hist.history,f)

#%% load history, tekrar yükleyebiliyoruz. 
import codecs
with codecs.open("deneme.json","r",encoding="utf-8") as f:
    h = json.loads(f.read())    
plt.plot(h["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(h["accuracy"],label="Train acc")
plt.plot(h["val_accuracy"],label="Validation acc")
plt.legend()
plt.show()

















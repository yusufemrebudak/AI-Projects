
import torch 
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import os
import time

# # you cab download the dataset on "https://www.kaggle.com/datasets/muhammeddalkran/lsi-far-infrared-pedestrian-dataset"


#%% Device config Ekstra, default cpu but gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)
# biz normal cpu ile devam edeceğiz ama
#%%
def read_images(path, num_img):
    array=np.zeros([num_img,64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path, mode="r")
        data = np.asarray(img, dtype="uint8") # elemanları int e çevirdim
        data = data.flatten()
        array[i,:] = data
        i += 1
    return array
# read train negative
train_negative_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Train\neg" 
num_train_negative_img = 43390
train_neg_array = read_images(train_negative_path,num_train_negative_img)
# train_neg_array = read_images(train_negative_path,num_train_negative_img)
x_train_negative_tensor = torch.from_numpy(train_neg_array)
# 1 boyutlu vektör, 2 boyutlu matrix, 3 boyutlu,4,5,6, boyutluların hepsinde tensor olarak adlandırıyoruz.
print("x_train_negative_tensor:", x_train_negative_tensor.size()) # torch.Size([43390, 2048])
y_train_negative_tensor = torch.zeros(num_train_negative_img, dtype=torch.long)

#%% read train positive

train_positive_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Train\pos" 
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)
x_train_positive_tensor = torch.from_numpy(train_positive_array)
print("x_train_positive_tensor:", x_train_positive_tensor.size()) 
y_train_positive_tensor = torch.ones(num_train_positive_img, dtype=torch.long)
 # bu iki veri setimi neg ve posları  train olarak  concat ile birleştirmem gerek
#%% concat train
x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor),0) # 0 anlamı satır bazında birleştir demektir.
y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor),0)
print("x_train:",x_train.size())
print("y_train:",y_train.size())
#%% read test negative 22050
test_negative_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Test\neg" 
num_test_negative_img = 22050
test_neg_array = read_images(test_negative_path,num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_neg_array)
print("x_train_negative_tensor:", x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(num_test_negative_img, dtype=torch.long)

#%% read test positive 5944

test_positive_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Test\pos" 
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path,num_test_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_train_positive_tensor:", x_test_positive_tensor.size()) 
y_test_positive_tensor = torch.ones(num_test_positive_img, dtype=torch.long)
#%%concat test
x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor),0) # 0 anlamı satır bazında birleştir demektir.
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor),0)
print("x_test:",x_test.size())
print("y_test:",y_test.size())
#%% visualize
plt.imshow(x_train[42000,:].reshape(64,32),cmap="gray")
#%%
num_epoch = 50
num_classes = 2
batch_size = 8933
learning_rate=0.00001
# import torch.nn as nn
# Module sınıfından bir şeyleri kullanıcam, kullanacağım modelleri(Conv, pool) initilizate etmem lazım 
# __init içinde
class Net(nn.Module):
    def __init__(self): #constructor-initilizear
        super(Net,self).__init__() ## nn.Module un inheritance gerçekleştirmesi için yaptığım şey.
        self.conv1=nn.Conv2d(1,10,5) # 1->input image channel, 10->output channel, 5 -> 5x5 lik filtreler.
        self.pool = nn.MaxPool2d(2,2) # size ı 2x2 lik bir pencere ile pooling yapsın.
        self.conv2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5,520) # input 16*3*5, 520 output 
        self.fc2 = nn.Linear(520,130)
        self.fc3 = nn.Linear(130,num_classes)
        
        
    def forward(self,x): # layerları birbirine bağladığım yapı olacak
        x = self.pool( F.relu( (self.conv1(x)) ) )
        # input önce conv layera girer sonra relu sonra pool yapar
        x = self.pool( F.relu((self.conv2(x))) )
        x = x.view(-1,16*13*5)
        x = F.relu((self.fc1(x)))
        x = F.relu((self.fc2(x)))
        x = self.fc3(x)
        return x

import torch.utils.data
train = torch.utils.data.TensorDataset(x_train,y_train)
 # x_train ve y_train zaten tensördü tensordataset metoduyla tekrardan tensöre çevirdim
trainloader = torch.utils.data.DataLoader(train,batch_size = batch_size,shuffle=True) # şimdide traini data ya çevirmem lazım
# böyle bir train den data  packet yap diyorum, batch_size a göre yap tabiki

test = torch.utils.data.TensorDataset(x_test, y_test)
testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle=False) 

net = Net() # Net sınfından nesne üretiliyor.
# net = Net().to(device) for GPU
#%% loss and optimizer
criterion = nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.SGD(net.parameters(),lr = learning_rate , momentum = 0.8)

#%% train a network
start = time.time() # şuanki zamanı depola
train_acc = []
test_acc = []
loss_list = []
use_gpu = False 

# modeli eğitmek için bu dataseti tam dataset üzerinde
for epoch in range(num_epoch): # 5000 epoch için for döngüsü 
    for i, data in enumerate(trainloader,0): # 
        # trainloader içinde batch_size belirlenmiştir bundan dolayı batch_size 
        # train datalarımı bitirene kadar burada bir döngü oluşacak.
        inputs, labels = data
        inputs = inputs.view(batch_size,1,64,32) # reshape
        inputs = inputs.float()
        if use_gpu:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
        # zero gradient
        
        optimizer.zero_grad() # her eğitimde gradientleri sıfırlıyorum
        
        # forward
        outputs = net(inputs)
        #loss
        loss = criterion(outputs,labels)
        # backward
        loss.backward() # find the gradients
        # update weights
        optimizer.step()
        
    # test
    correct = 0
    total = 0
    with torch.no_grad(): # backpropagation yapmıyoruz demek 
        for data in testloader:
            images, labels = data 
            images = images.view(images.size(0),1,64,32)
            images = images.float()
            # gpu
            if use_gpu: 
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1) # indeksini return ediyor anlamında
            total += labels.size(0) # ne kadar datamın olduğu
            correct += (predicted == labels).sum().item()
    acc1 = 100*(correct/total)
    print("accuracy test: ",acc1)
    test_acc.append(acc1)
            

    # train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data 
            images = images.view(images.size(0),1,64,32)
            images = images.float()
            # gpu
            if use_gpu: 
                if torch.cuda.is_available():
                    inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1) # indeksini return ediyor anlamında
            total += labels.size(0)
            correct += (predicted==labels).sum().item()
    acc2 = 100*(correct/total)
    print("accuracy train: ",acc2)
    test_acc.append(acc2)


print("train is done.")




end = time.time()
process_time = (end-start) / 60
print("Process Time : ", process_time)


"""
Bizim örneğimizde image shape 64x32. Padding değeri girmiyoruz, varsayılan değerini alıyor ki o da 0'a eşit. Stride değeri girmiyoruz, varsayılan değeri 1. Dilation değeri girmiyoruz, varsayılan değeri 1. Bu bilgileri toparladığımızda;

o = output = bilinmeyenimiz

p = padding = 0

k = kernel_size = 5

s = stride = 1

d = dilation = 1

i =input shape = 64x32

Bir Conv2d metoduna giren bir input aşağıdaki formüle göre çıkış boyutu verir.

o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
Kendi parametre değerlerimizle çözüm yaptığımızda;

i=64 için o=60,

i=32 için o=28 çıkar. Böylece inputumuz ilk Conv2d metodundan çıktığında 60x28 shape'ini alıyor.

Ardından MaxPooling2d yapıyoruz. Örneğimizde pool_size=(2,2) belirlediğimiz için 60x28 shape'i 30x14'e indiriyoruz.

Ardından 30x14 için tekrar Conv2d uyguluyoruz.

i=30 için o=26,

i=14 için o=10, böylece inputumuz ikinci Conv2d metodundan çıktığında 26x10 shape'ini alıyor.

Ardından tekrar MaxPooling2d yapıyoruz. Örneğimizde pool_size=(2,2) belirlediğimiz için 26x10 shape'i 13x5'e indiriyoruz.

Son Conv2d metodunda feature map sayısını 16 olarak belirlediğimiz için elimizde 16 adet 13x5 image oluyor. Bu yüzden nn.Linear(16*13*5,520) yazıyoruz. Umarım faydalı olur.
"""
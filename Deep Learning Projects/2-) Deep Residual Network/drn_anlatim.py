import torch 
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os 
import torch.utils.data
#%% Device config

# you cab download the dataset on "https://www.kaggle.com/datasets/muhammeddalkran/lsi-far-infrared-pedestrian-dataset"


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: ",device)

#%% Dataset
def read_images(path, num_img):
    array = np.zeros([num_img,64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path, mode = 'r')
        data = np.asarray(img,dtype = "uint8")
        data = data.flatten()
        array[i,:] = data
        i += 1      
    return array
        
# read train negative  43390
train_negative_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Train\neg" 
num_train_negative_img = 43390
train_negative_array = read_images(train_negative_path,num_train_negative_img)
x_train_negative_tensor = torch.from_numpy(train_negative_array[:42000,:])
print("x_train_negative_tensor: ",x_train_negative_tensor.size())
y_train_negative_tensor = torch.zeros(42000,dtype = torch.long)
print("y_train_negative_tensor: ",y_train_negative_tensor.size())

# read train positive 10208
train_positive_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Train\pos" 
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)
x_train_positive_tensor = torch.from_numpy(train_positive_array[:10000,:])
print("x_train_positive_tensor: ",x_train_positive_tensor.size())
y_train_positive_tensor = torch.ones(10000,dtype = torch.long)
print("y_train_positive_tensor: ",y_train_positive_tensor.size())

# concat train
x_train = torch.cat((x_train_negative_tensor, x_train_positive_tensor), 0)
y_train = torch.cat((y_train_negative_tensor, y_train_positive_tensor), 0)
print("x_train: ",x_train.size())
print("y_train: ",y_train.size())

# read test negative  22050
test_negative_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Test\neg" 
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path,num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_negative_array[:18056,:])
print("x_test_negative_tensor: ",x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(18056,dtype = torch.long)
print("y_test_negative_tensor: ",y_test_negative_tensor.size())

# read test positive 5944
test_positive_path = r"C:\Users\yusuf\Desktop\ImageProcessing-DeepLearning\Deep Learning ve Python İleri Seviye Derin Öğrenme\2-) Deep Residual Network\LSIFIR\LSIFIR\Classification\Test\pos" 
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path,num_test_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_test_positive_tensor: ",x_test_positive_tensor.size())
y_test_positive_tensor = torch.zeros(num_test_positive_img,dtype = torch.long)
print("y_test_positive_tensor: ",y_test_positive_tensor.size())

# concat test
x_test = torch.cat((x_test_negative_tensor, x_test_positive_tensor), 0)
y_test = torch.cat((y_test_negative_tensor, y_test_positive_tensor), 0)
print("x_test: ",x_test.size())
print("y_test: ",y_test.size())


#%% visualize
plt.imshow(x_train[1001,:].reshape(64,32), cmap='gray') # 45002 ve 1001

# %% 
num_classes = 2
# Hyper parameters
num_epochs = 5
batch_size = 2000
learning_rate = 0.0001

train = torch.utils.data.TensorDataset(x_train,y_train) # bu ikisini birleştirip train datamı oluştur 
trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = True)

test = torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


# %% Deep Residual Network, kullanılacak olan conv lar yazılır ve basic bloklar yazılır

def conv3x3(in_planes, out_planes, stride = 1):# kernel size 3 olacak anlamında
    # in planes : input channel, input image daki chaneel sayısı mesela gray scale ise 1 dir rgb ise 3 dür.
    # out_planes : output channel ım, number of noron, layerdaki noron sayısıdır. 
    # out_planes yani output chaneels corresponds the number of conv kernel(filter)
    # 5x5 filter var input image: 3, out planes:7 ise size 7 x 5 x 5 x 3 
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride=stride, padding = 1, bias=False)
    # padding filtre dolaştırdıktan sonra feature map küçülür bunu istemeyiz bundan dolayı
    # gelen inputa filtre uygulamadan önce padding ile boyutlarını büyütürüz padding 1 ise
    # 5x5 input 7x7 haline gelir.
def conv1x1(in_planes, out_planes, stride = 1):# kernel size 3 olacak anlamında
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride=stride, bias=False)

class BasicBlock(nn.Module): # nn.Module classından inherit ediyorum.
    expansion =1
    def __init__(self, inplanes, planes, stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        # BasicBlock inherit edebilmesi için yani  nn.Module clasındakileri kullanabilmem için yapıyorum

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)# her layerda normalization yap demek
        self.relu = nn.ReLU(inplace = True) # relu aktivasyon fonksiyonunu çağırdıktan sonra sonucunu kendisine eşitle demek
        self.drop = nn.Dropout(0.9)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
            
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=num_classes):
        super(ResNet,self).__init__()
        self.inplanes=64
        self.conv1 = nn.Conv2d(1,64,kernel_size=7, stride=2, padding=3, bias= False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
        self.layer1 = self._make_layer(block,64, layers[0], stride = 1)       
        self.layer2 = self._make_layer(block,128, layers[1], stride = 2)        
        self.layer3 = self._make_layer(block,256, layers[2], stride = 2)        
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # biz sana outputu söyleyelim 
        # sen bu pool penceresini kendi içinde ne yapıyosan yap,bana (1,1) verde
        self.fc = nn.Linear(256*block.expansion, num_classes)
        
        # resnet olduğundan bir sürü layer ve 
        # bir sürü weight vardır bunları mantıklı bir şekilde initilize etmemiz gerek.
        for m in self.modules(): # modules nn classımdan geliyor.
        # bu m lerin içine sahip olduğum conv1, bn1, gibi layerları yüklüyor.
            if isinstance(m, nn.Conv2d): # ben eğer conv2d layerımdaysam buraya gir demek
                nn.init.kaiming_normal_(m.weight,mode="fan_out", nonlinearity="relu")
                # kaiming_normal 0 a yakın sayıları weight lerimize initial değer olarak atıyor.
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1) # tüm weightleri 1 e eşitle
                nn.init.constant_(m.bias,0) # bias yukarıda false yapıldı hiç bir anlamı yok burada zaten 0 lıyorum.
        
    
    def _make_layer(self, block, planes, blocks, stride = 1): 
        # bu resnet classı bildiğimiz gerçek modellerin nasıl yazıldığının bi tutorial ı gibidir.
        # amaç block oluşturmak,
        # basic blockları art arta koyarak yeni bir yapı oluşturacak.
        # BB1 -> BB2 -> BB3 ->..... ->BBn
        downsample = None
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride),
                    nn.BatchNorm2d(planes*block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.inplanes, planes))
        
        return nn.Sequential(*layers)
         
        
            
    
    def forward(self,x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x
        
model = ResNet(BasicBlock, [2,2,2])


#%% 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#%%  train 


loss_list = []
train_acc = []
test_acc = []
use_gpu = False
total_step = len(trainloader)

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(trainloader):
        images = images.view(batch_size,1,64,32)
        images = images.float() # floata çeviriyorum 
        if use_gpu:
            if torch.cuda.is_available():
                #images, labels = images.to(device), labels.to(device)
                print("gpu ")
                
        outputs = model(images)
        
        loss = criterion(outputs,labels)
        # backward and optimization
        optimizer.zero_grad() # pytroch da graidnetler i bizim her güncellemeden sonra 0 lamamız lazım.
        loss.backward() # backpropagation yap
        optimizer.step() # weightleri güncelle
        if i%2==0:
            print("Epoch: {} {}/{}".format(epoch,i,total_step))
        
    #train
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            images = images.view(images.size(0),1,64,32)
            images=images.float()
        
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total +=labels.size(0)
            correct += (predicted==labels).sum().item()
    print("Accuracy Train %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)
    # test
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.view(images.size(0),1,64,32)
            images=images.float()
        
            outputs = model(images)
            _,predicted = torch.max(outputs.data,1)
            total +=labels.size(0)
            correct += (predicted==labels).sum().item()
    print("Accuracy Test %d %%"%(100*correct/total))
    train_acc.append(100*correct/total)
    
    loss_list.append(loss.item)

        
        
        
        
        
        
        
        
        
        
        




















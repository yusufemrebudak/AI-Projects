import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#  read csv
data = pd.read_csv("framingham.csv")

print(data.head())
y = data.TenYearCHD.values
x = data.drop(["TenYearCHD"],axis=1) # bunu da düşürdükten sonra artık
X = x.apply(lambda x: x.fillna(x.mean()),axis=0)
# data.drop(["User ID","Gender"],axis=1,inplace = True)
X_data = (X - np.min(X)) / (np.max(X) - np.min(X)).values
# TRAIN/TEST SPLIT
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_data,y,test_size=0.2,random_state=42)

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

def initilizate_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01) # dimensiona 1 lik 0.01 lerden oluşan bir weight matrix yap diyorum.
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b # matris çarpımı yapıyorum,
    # print("z shape",z.shape)
    # weightlerimle her bir featurem ı çarpıyorum ve bias ekliyorum her birine 
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head)-(1-y_train)*np.log(1-y_head)
    print("loss: ",loss.shape)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    # print("Cost :",cost)
    # cost bulundu buraya kadar
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head - y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    print("gradients: ",gradients)
    
    return cost,gradients

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # print("parameteres : {}, bias:{}".format(gradients["derivative_weight"] ,gradients["derivative_bias"]))
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    print("parameteres : {}, bias:{}".format( w ,b))
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 15
    w,b = initilizate_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    # np.abs mutlak değerdir.
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 500)  

####################################################################################################
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()  
lr.fit(x_train.T,y_train.T)
print("text accuracy: {}".format(lr.score(x_test.T, y_test.T))) 
# lr.score() --> predict et daha sonra bana ne kadar doğru predict etmişim onu ver dedi.











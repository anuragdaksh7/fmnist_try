import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\anura\OneDrive\Desktop\fashion mnist\fashion-mnist_train.csv')
data2 = pd.read_csv(r'C:\Users\anura\OneDrive\Desktop\fashion mnist\fashion-mnist_test.csv')

ll = [f"pixel{i}" for i in range(1,785)]
inputs = pd.DataFrame(data, columns= ll)
inputs = np.asarray(inputs)
#print(len(inputs))
new = []
neww = []
tmp = []
for i in inputs:
    new = []
    for j in range(28):
        tmp = []
        tmp.append(i[28*j:28*(j+1)])
        #tmp = np.array([i for i in tmp])
        #np.resize(tmp,(28))
        new.append(tmp)
        
    new = [np.asarray(i,dtype="float64") for i in new]
    new = [np.resize(i,(28)) for i in new]
    new = np.asarray(new,dtype="float64")
    #np.resize(new,(28,28))
    neww.append(new)
neww = np.asarray(neww)

output = pd.DataFrame(data, columns= ['label'])
output = np.asarray(output)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(512,activation="sigmoid"),
    keras.layers.Dense(512,activation = "sigmoid"),
    keras.layers.Dense(512,activation = "sigmoid"),
    keras.layers.Dense(512,activation = "sigmoid"),
    keras.layers.Dense(1024,activation = "sigmoid"),
    keras.layers.Dense(10,activation="softmax")
    ])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(neww,output,epochs=5)

dd = pd.DataFrame(data2)
ll = [f"pixel{i}" for i in range(1,785)]
inputss = pd.DataFrame(data2, columns= ll)
inputss = np.asarray(inputss)

new = []
new2 = []
newww = []
neww2 = []
tmp = []
for i in inputss:
    new = []
    new2 = []
    for j in range(28):
        tmp = []
        
        tmp.append(i[28*j:28*(j+1)])
        new2.append(tmp)
        #tmp = np.array([i for i in tmp])
        #np.resize(tmp,(28))
        new.append(tmp)
        
    neww2.append(new2)
    new = [np.asarray(i,dtype="float64") for i in new]
    new = [np.resize(i,(28)) for  i in new]
    #print(new[0].shape)
    new = np.asarray(new,dtype="float64")
    
    #np.resize(new,(28,28))
    newww.append(new)

newww = np.asarray(newww,dtype = "float64")
outputss = pd.DataFrame(data2, columns= ['label'])
outputss = np.asarray(outputss)

test_loss, test_acc = model.evaluate(newww, outputss)
prediction=model.predict(neww)

i = 0
while i<100:
    plt.grid(False)
    plt.imshow(neww[i],cmap=plt.cm.binary)

    
    plt.title("prediction: "+conv(np.argmax(prediction[i])))

    plt.show()
    i+=1

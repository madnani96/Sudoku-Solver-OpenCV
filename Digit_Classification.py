import numpy as np
import cv2
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,Dropout
from sklearn.model_selection import train_test_split


from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

path='Data'

lst=os.listdir(path)
print(lst)
numberofclasses=len(lst)
imagelist=[]
classlabel=[]

x=0
while(x<numberofclasses):
    piclist=os.listdir(path+"/"+str(x))
    for y in piclist:
        currentImage=cv2.imread(path+"/"+str(x)+"/"+y)
        currentImage=cv2.resize(currentImage,(32,32))
        imagelist.append(currentImage)
        classlabel.append(x)
    x+=1

print("Number of Images",len(imagelist))
imagelist=np.array(imagelist)
classlabel=np.array(classlabel)
print("BeforeSplit train shape",imagelist.shape)
print("BeforeSplit label shape",classlabel.shape)
trainX,testX,trainY,testY=train_test_split(imagelist,classlabel,test_size=0.2)
trainX,validationX,trainY,validationY=train_test_split(trainX,trainY,test_size=0.2)
print("After Splitting train",trainX.shape, trainY.shape)
print("After Splitting validation",validationX.shape, validationY.shape)
print("After Splitting test",testX.shape,testY.shape)

x=0
while(x<10):
    print("Number of",x, "classes in training:", len(np.where(trainY==x)[0]))
    x+=1

def preprocess(image):
    #Convert to grayscale
    grayscaleimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Equalization
    equalizedimage=cv2.equalizeHist(grayscaleimg)
    #Normalization
    normalizedimage=equalizedimage/255
    return normalizedimage
#To check out random preprocessed image uncomment below 2 lines
#cv2.imshow("Random image preprocessed",cv2.resize(preprocess(testX[5]),(350,350)))
#cv2.waitKey(0)

#Preprocessing all images using map function
trainX=np.array(list(map(preprocess,trainX)))
testX=np.array(list(map(preprocess,testX)))
validationX=np.array(list(map(preprocess,validationX)))

trainX=trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2],1)
print(trainX.shape)
testX=testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2],1)
validationX=validationX.reshape(validationX.shape[0],validationX.shape[1],validationX.shape[2],1)


newdata=ImageDataGenerator(shear_range=0.1,height_shift_range=0.1,zoom_range=0.2,rotation_range=10,width_shift_range=0.1)
newdata.fit(trainX)
trainY=to_categorical(trainY,numberofclasses)
testY=to_categorical(testY,numberofclasses)
validationY=to_categorical(validationY,numberofclasses)


def classificationmodel():

    numberoffilters=120
    model=Sequential()
    model.add((Conv2D(numberoffilters,(7,7),input_shape=(32,32,1),activation='relu')))
    model.add((Conv2D(numberoffilters,(7,7),activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add((Conv2D(numberoffilters//2, (3,3), activation='relu')))
    model.add((Conv2D(numberoffilters//2, (3,3), activation='relu')))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(numberofclasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
model1=classificationmodel()
print(model1.summary())

hst=model1.fit_generator(newdata.flow(trainX,trainY,batch_size=25),validation_data=(validationX,validationY),steps_per_epoch=200,epochs=20,shuffle=1)

plt.figure(1)
plt.plot(hst.history['loss'])
plt.plot(hst.history['val_loss'])
plt.legend(['Training','Validation'])
plt.title('Loss')
plt.xlabel('Epoch')
plt.figure(2)
plt.plot(hst.history['accuracy'])
plt.plot(hst.history['val_accuracy'])
plt.legend(['Training','Validation'])
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.show()
score = model1.evaluate(testX,testY,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy =', score[1])
model1.save('Myfirstmodel/')

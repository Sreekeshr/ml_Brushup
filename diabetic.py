import tensorflow
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json

data = loadtxt("/home/sreekesh/sreek/ML_projects/trial_project/pima-indians-diabetes.csv",delimiter=",")
X_val = data[:,0:8]
Y_val = data[:,8]

model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_val,Y_val,epochs=50,batch_size=10)

_,accuracy = model.evaluate(X_val,Y_val)
print("Accuracy : "+ str(accuracy))

json_model = model.to_json()
with open("/home/sreekesh/sreek/ML_projects/trial_project/Trial_model.json","w") as json_file:
    json_file.write(json_model)

model.save_weights("/home/sreekesh/sreek/ML_projects/trial_project/Trial_model.h5")
print('model saved to disk')


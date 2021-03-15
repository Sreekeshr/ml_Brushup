from tensorflow.keras.models import model_from_json
import numpy as np
from numpy import loadtxt

x =np.array([[8,183,64,0,0,23.3,0.672,32],[1,89,66,23,94,28.1,0.167,21]])



json_file = open('/home/sreekesh/sreek/ML_projects/trial_project/Trial_model.json','r')
json_model = json_file.read()
json_file.close()
loaded_model = model_from_json(json_model)
loaded_model.load_weights('/home/sreekesh/sreek/ML_projects/trial_project/Trial_model.h5')

print("loaded model....")

predict_val = loaded_model.predict_classes(x)
print(predict_val)


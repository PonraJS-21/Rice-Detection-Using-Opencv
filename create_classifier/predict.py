import numpy as np
import os
import cv2
from keras.models import model_from_json
import pickle

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

# load json and create model
json_file = open('model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_face.h5")
print("Model is now loaded in the disk")

path = 'predict/un_broken/'

img = os.listdir(path)

for im in img:
	image = np.array(cv2.imread(path+im))
	image = cv2.resize(image, (50, 50))
	image = np.array([image])
	image = image.astype('float32')
	image = image / 255.0
	# image = np.expand_dims(image, axis=-1)
	print(image.shape)

	prediction=loaded_model.predict(image)

	print(prediction)

	print(im)

	print(int_to_word_out[np.argmax(prediction)])

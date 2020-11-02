import numpy as np
from keras.models import model_from_json
import pickle
import cv2

classifier_f = open("model/int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

# load json and create model
json_file = open('model/model_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model/model_face.h5")
print("Model is now loaded in the disk")

def classify_image(img):
	image=np.array(img)
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (50, 50))
	image=np.array([image])
	image = image.astype('float32')
	image = image / 255.0
	# image = np.expand_dims(image, axis=-1)

	prediction = loaded_model.predict(image)
	probability = prediction[0][np.argmax(prediction)]
	return (int_to_word_out[np.argmax(prediction)], probability)

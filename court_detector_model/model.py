import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import sys
import cv2
import json
import numpy as np
import argparse
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split


IMAGE_HEIGHT = 360
MAX_IMAGES = 100


def get_model_path(model_name):
	return "MODELS/" + str(model_name)


def get_model_datafile_path(model_name):
	return get_model_path(model_name) + "/data.json"


def get_model_file(model_name):
	return "MODELS/" + str(model_name) + ".keras"


def read_model_data(model_name):
	with open(get_model_datafile_path(model_name)) as data_file:
		return json.load(data_file)


def store_model_data(model_name, model_data):
	with open(get_model_datafile_path(model_name), "w") as data_file:
		json.dump(model_data, data_file)


def model_ini(model_name):
	if not os.path.isfile(get_model_path(model_name)):
		os.makedirs(get_model_path(model_name))
	
	if not os.path.isfile(get_model_datafile_path(model_name)):
		with open(get_model_datafile_path(model_name), 'w') as data_file:
			json.dump({"count": 0,
					   "classification": {}}, data_file)


def img_preprocess(img):
	img_prop = img.shape[0] / img.shape[1]
	img = cv2.resize(img, (int(IMAGE_HEIGHT / img_prop), IMAGE_HEIGHT))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype('float32')
	img /= 255
	return img


def load_data(model_name, max_images=None):
	img_path_list = [get_model_path(model_name) + "/" + img
					 for img in os.listdir(get_model_path(model_name)) if img[-4:] == ".png"]

	img_list = [img_preprocess(cv2.imread(img_path))
				for img_path in sorted(img_path_list, key=lambda x: int(x.split("/")[-1].split(".")[0]))]
	
	#cv2.imwrite(img_path_list[0] + "_PROCESSED", img_list[0])
	
	model_data = list(read_model_data(model_name)["classification"].values())
	
	print("  Data readed: " + str(len(img_list)))
	
	img_list = np.array(img_list)
	model_data = np.array(model_data)
	
	if max_images is not None and len(img_list) > max_images:
		img_list, X_test, model_data, Y_test = train_test_split(img_list, model_data, test_size=(1 - (max_images / len(img_list))), stratify=model_data)
		#random_i = np.random.choice(len(img_list), max_data, replace=False)
		#img_list = img_list[random_i]
		#model_data = model_data[random_i]
	
	return (img_list, model_data)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model_name", type=str)
	args = parser.parse_args()
	
	if args.model_name is not None:
		if not os.path.isfile(get_model_datafile_path(args.model_name)):
			print("  Error: modelo '" + str(read_model_data(args.model_name)) + "' no encontrado")
			sys.exit(0)
	
	X_train, Y_train = load_data(args.model_name, max_images=MAX_IMAGES)
	X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
	
	data_size = X_train.shape[0]
	print("  Data size for train: " + str(data_size))
	
	width = X_train.shape[1]
	height = X_train.shape[2]
	print("  Img shape: " + str(width) + "x" + str(height))
	
	#sys.exit(0)
	
	model = Sequential()
	model.add(Convolution2D(32, (5, 5), data_format='channels_last', activation='relu', input_shape=(width,height,1)))
	model.add(MaxPooling2D(pool_size=(4,4)))
	model.add(Convolution2D(64, (3, 3), data_format='channels_last', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	print("  Model '" + str(args.model_name) + "' crerated")
	
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train, Y_train, batch_size=data_size, epochs=50, verbose=1)
	print("  Model '" + str(args.model_name) + "' fitted")
	
	score = model.evaluate(X_train, Y_train, verbose=0)
	print("  Model score: " + str(score))
	
	model.save(get_model_file(args.model_name))







































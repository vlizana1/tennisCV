import os
import sys
import cv2
import json
import numpy as np
import model as MDL
import argparse
from keras.models import load_model



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model_name", type=str)
	parser.add_argument("validation_set", type=str)
	args = parser.parse_args()
	
	# Carga los datos de validacion
	X_val, Y_val = MDL.load_data(args.validation_set)
	
	# Carga el modelo
	model = load_model(MDL.get_model_file(args.model_name))
	
	confusion_matrix = [[0, 0,], # VP, FP
						[0, 0]]  # FN, VN
	
	# Contrasta las predicciones
	for i_val, (x_val, y_val) in enumerate(zip(X_val, Y_val)):
		img_val = x_val.reshape(1, x_val.shape[0], x_val.shape[1], 1)
		y = model.predict(img_val, batch_size=1, verbose=0)
		
		print("Validacion " + str(i_val) + ": " + str(y_val) + " -> " + str(y)[:4], end="\r")
		
		if y < 0.5:
			if y_val == 0:	# VP
				confusion_matrix[0][0] += 1
			else:			# FP
				confusion_matrix[0][1] += 1
		else:
			if y_val == 1:	# VN
				confusion_matrix[1][1] += 1
			else:			# FN
				confusion_matrix[1][0] += 1


	print("Matriz de validacion:" + " "*25)
	print("\t" + str(confusion_matrix[0]))
	print("\t" + str(confusion_matrix[1]))
	
	
	a = (confusion_matrix[0][0] + confusion_matrix[1][1]) / len(X_val)
	print("Accuracy:\t" + str(a))
	
	
	p = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
	print("Precision:\t" + str(p))
	r = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
	print("Recall:\t\t" + str(r))
	
	
	F1 = 2 * (p * r) / (p + r)
	print("F1:\t\t" + str(F1))
	
	
	
	
	
	
	
	
	
	
	
	

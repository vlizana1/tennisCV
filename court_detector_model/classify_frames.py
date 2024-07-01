import os
import sys
import cv2
import json
import argparse
import model as MDL


def classify_frame(model_name, model_data, frame, cod):
	model_data["classification"][model_data["count"]] = cod
	cv2.imwrite(MDL.get_model_path(model_name) + "/" + str(model_data["count"]) + ".png", frame)
	model_data["count"] += 1
	return model_data


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model_name", type=str)
	parser.add_argument("video_name", type=str)
	args = parser.parse_args()
	
	if args.model_name is not None:
		if not os.path.isfile(MDL.get_model_datafile_path(args.model_name)):
			MDL.model_ini(args.model_name)
	
	if args.video_name is not None:
		if not os.path.isfile(args.video_name):
			print("ERR: video '" + str(args.video_name) + "' no encontrado")
			sys.exit(0)
		elif args.video_name[-4:] != ".mp4":
			print("ERR: '" + str(args.video_name) + "' tiene que ser un video")
			sys.exit(0)
	
	model_data = MDL.read_model_data(args.model_name)
	cap = cv2.VideoCapture(args.video_name)
	
	frame_i = 0
	paused = False
	while cap.isOpened():
		if not paused:
			ret, frame = cap.read()
		cv2.imshow('frame', frame)
		
		key = cv2.waitKey(25)
		if key & 0xFF == ord('q'):
			break
		elif key & 0xFF == ord(' '):
			if paused is True:
				paused = False
			else:
				paused = True
		elif key & 0xFF == ord('j'):
			model_data = classify_frame(args.model_name, model_data, frame, 0) # is frame
		elif key & 0xFF == ord('k'):
			model_data = classify_frame(args.model_name, model_data, frame, 1) # not frame
		elif key & 0xFF == ord('f'):
			for i in range(500):
				if cap.isOpened():
					_, _ = cap.read()
		
		print(frame_i, end="\r")
		frame_i += 1
	
	MDL.store_model_data(args.model_name, model_data)




















import os
import sys
import cv2
import json
import numpy as np
import model as MDL
import argparse
from keras.models import load_model

GREEN = (0,255,0)
RED = (0,0,255)

TAM_BUFFER = 61


def draw_points(img, pts):
	for x, y in pts:
		cv2.circle(img, (x, y), 5, GREEN, -1)


def manage_buffer(frame_buffer, new):
	if len(frame_buffer[1]) < TAM_BUFFER:
		frame_buffer[1].append(new)
		last = (None, None)
		frame_buffer[0] = (frame_buffer[0] + 1) % TAM_BUFFER
	else:
		last = frame_buffer[1][frame_buffer[0]]
		frame_buffer[1][frame_buffer[0]] = new
		frame_buffer[0] = (frame_buffer[0] + 1) % TAM_BUFFER
		
		"""
		"""
		r_ini = (frame_buffer[0] - (TAM_BUFFER - 1) / 2) % TAM_BUFFER
		sum_izq = 1
		sum_der = 1
		
		i = int(r_ini)
		while i != frame_buffer[0]:
			sum_izq += frame_buffer[1][i][1] * 2
			i = (i + 1) % TAM_BUFFER
		i = (i + 1) % TAM_BUFFER
		while i != r_ini:
			sum_der += frame_buffer[1][i][1] * 2
			i = (i + 1) % TAM_BUFFER
		
		sum_izq = round(sum_izq / (TAM_BUFFER / 2))
		sum_der = round(sum_der / (TAM_BUFFER / 2))
		
		if sum_izq == sum_der and sum_izq != 1:
			frame_buffer[1][frame_buffer[0]] = (frame_buffer[1][frame_buffer[0]][0], int(sum_izq / 2))
	
	return last


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("model_name", type=str)
	parser.add_argument("video_path", type=str)
	args = parser.parse_args()
	
	if args.model_name is not None:
		if not os.path.isfile(MDL.get_model_file(args.model_name)):
			print("  Error: modelo '" + str(MDL.read_model_data(args.model_name)) + "' no encontrado")
			sys.exit(0)
	
	if args.video_path is not None:
		if not os.path.isfile(args.video_path):
			print("ERR: video '" + str(args.video_path) + "' no encontrado")
			sys.exit(0)
		elif args.video_path[-4:] != ".mp4":
			print("ERR: '" + str(args.video_path) + "' tiene que ser un video")
			sys.exit(0)
	
	model = load_model(MDL.get_model_file(args.model_name))
	cap = cv2.VideoCapture(args.video_path)
	
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
	cutted_video = cv2.VideoWriter(args.video_path[:-4] + "_CUTTED" + args.video_path[-4:],
								   fourcc, fps, (frame_width, frame_height))
	
	green_count = 0
	red_count = 0
	frame_buffer = [0, []]
	paused = False
	while cap.isOpened():
		if not paused:
			for i in range(1):
				ret, frame = cap.read()
			
			img = MDL.img_preprocess(frame)
			img = img.reshape(1, img.shape[0], img.shape[1], 1)
			ans = model.predict(img, batch_size=1, verbose=0)
			
			
			#frame, ans = manage_buffer(frame_buffer, (frame, ans[0][0]))
			
			if frame is not None:
				if ans < 0.5:
					#cutted_video.write(frame)
					color = GREEN
					green_count += 1
					red_count = 0
				else:
					color = RED
					green_count = 0
					red_count += 1

				cv2.circle(frame, (50,50), 10, color, -1)
				cv2.imshow('frame', frame)
				print(str(green_count) + "\t" + str(red_count) + "\t|\t" + str(ans))

		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			cap.release()
		elif key & 0xFF == ord('j'):
			print("\t\tjumping 500...")
			for i in range(500):
				ret, frame = cap.read()
		elif key & 0xFF == ord(' '):
			if paused is True:
				paused = False
			else:
				paused = True
	
	cutted_video.release()

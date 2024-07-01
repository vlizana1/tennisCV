import os
import sys
import cv2
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import tensorflow as tf
import torchvision

import fotograma as FTGM
import detect_stroke as STRK
import methods.math_methods as MTH
import methods.draw_methods as DRW


TD_DETECT_BALL = "DETECT_BALL"
TD_POST_CHECKS = "POST_CHECKS_BALL"
TD_APROXIMATE = "APROXIMATE_BALL"

FRAMES_TO_COPY = 4
LAST_FRAMES_TO_MERGE = [-1, -3]
MAX_BALL_DIST = 20 * (FRAMES_TO_COPY + 1)
MIN_BALL_DIST = 3 * (FRAMES_TO_COPY + 1)
MAX_STATICS_FRAMES = 4


class ConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, pad, bias=True, bn=True):
		super().__init__()
		if bn:
			self.block = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
				nn.ReLU(),
				nn.BatchNorm2d(out_channels)
			)
		else:
			self.block = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, bias=bias),
				nn.ReLU()
			)

	def forward(self, x):
		return self.block(x)


class BallTrackerNet(nn.Module):
	def __init__(self, out_channels=256, bn=True):
		super().__init__()
		self.out_channels = out_channels

		# Encoder layers
		layer_1 = ConvBlock(in_channels=9, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_3 = nn.MaxPool2d(kernel_size=2, stride=2)
		layer_4 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_5 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_6 = nn.MaxPool2d(kernel_size=2, stride=2)
		layer_7 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_8 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_9 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_10 = nn.MaxPool2d(kernel_size=2, stride=2)
		layer_11 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_12 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_13 = ConvBlock(in_channels=512, out_channels=512, kernel_size=3, pad=1, bias=True, bn=bn)

		self.encoder = nn.Sequential(layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9,
									 layer_10, layer_11, layer_12, layer_13)

		# Decoder layers
		layer_14 = nn.Upsample(scale_factor=2)
		layer_15 = ConvBlock(in_channels=512, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_16 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_17 = ConvBlock(in_channels=256, out_channels=256, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_18 = nn.Upsample(scale_factor=2)
		layer_19 = ConvBlock(in_channels=256, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_20 = ConvBlock(in_channels=128, out_channels=128, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_21 = nn.Upsample(scale_factor=2)
		layer_22 = ConvBlock(in_channels=128, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_23 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3, pad=1, bias=True, bn=bn)
		layer_24 = ConvBlock(in_channels=64, out_channels=self.out_channels, kernel_size=3, pad=1, bias=True, bn=bn)

		self.decoder = nn.Sequential(layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, layer_21,
									 layer_22, layer_23, layer_24)

		self.softmax = nn.Softmax(dim=1)
		self._init_weights()

	def forward(self, x, testing=False):
		batch_size = x.size(0)
		features = self.encoder(x)
		scores_map = self.decoder(features)
		output = scores_map.reshape(batch_size, self.out_channels, -1)
		# output = output.permute(0, 2, 1)
		if testing:
			output = self.softmax(output)
		return output

	def _init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.Conv2d):
				nn.init.uniform_(module.weight, -0.05, 0.05)
				if module.bias is not None:
					nn.init.constant_(module.bias, 0)
			elif isinstance(module, nn.BatchNorm2d):
				nn.init.constant_(module.weight, 1)
				nn.init.constant_(module.bias, 0)

	def inference(self, frames: torch.Tensor):
		self.eval()
		with torch.no_grad():
			if len(frames.shape) == 3:
				frames = frames.unsqueeze(0)
			if next(self.parameters()).is_cuda:
				frames.cuda()
			# Forward pass
			output = self(frames) #testing=True ¿?
			output = output.argmax(dim=1).detach().cpu().numpy()
			if self.out_channels == 2:
				output *= 255
			x, y = self.get_center_ball(output)
		return x, y

	def get_center_ball(self, output):
		output = output.reshape((360, 640))

		# cv2 image must be numpy.uint8, convert numpy.int64 to numpy.uint8
		output = output.astype(np.uint8)

		# reshape the image size as original input image
		heatmap = cv2.resize(output, (640, 360))

		# heatmap is converted into a binary image by threshold method.
		ret, heatmap = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

		# find the circle in image with 2<=radius<=7
		circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
		maxRadius=7)
		# check if there have any tennis be detected
		if circles is not None:
			# if only one tennis be detected
			if len(circles) == 1:
				x = int(circles[0][0][0])
				y = int(circles[0][0][1])

				return x, y
		return None, None


def get_model():
	BALL_DETECTOR = BallTrackerNet(out_channels=2)
	saved_state_dict = torch.load("models/tracknet_weights_2_classes.pth", map_location=torch.device('cpu'))
	BALL_DETECTOR.load_state_dict(saved_state_dict['model_state'])
	BALL_DETECTOR.eval().to(torch.device("cpu"))
	return BALL_DETECTOR


def combine_three_frames(frame1, frame2, frame3, width, height):
	# Resize and type converting for each frame
	img = cv2.resize(frame1, (width, height))
	# input must be float type
	img = img.astype(np.float32)

	# resize it
	img1 = cv2.resize(frame2, (width, height))
	# input must be float type
	img1 = img1.astype(np.float32)

	# resize it
	img2 = cv2.resize(frame3, (width, height))
	# input must be float type
	img2 = img2.astype(np.float32)

	# combine three imgs to  (width , height, rgb*3)
	imgs = np.concatenate((img, img1, img2), axis=2)

	# since the odering of TrackNet  is 'channels_first', so we need to change the axis
	imgs = np.rollaxis(imgs, 2, 0)

	return (torch.from_numpy(np.array(imgs)) / 255).to(torch.device("cpu"))


def detect_ball(frame: FTGM.Fotograma, last_frame: FTGM.Fotograma, last_frames_to_merge: list):
	if last_frame is not None \
	and (last_frame.info.get("ball", None) is not None or last_frame.info.get("fix_ball", False) is True) \
	and last_frame.info.get("frames_to_copy_ball", -1) > 0:
		#frame.info["ball"] = last_frame.info["ball"]
		#frame.info["ball_copied"] = True
		frame.info["frames_to_copy_ball"] = last_frame.info["frames_to_copy_ball"] - 1

		frame.info["fix_ball"] = True

		#DRW.draw_points(frame, [frame.info["ball"]], DRW.GREEN)

		frame.push_to_do(STRK.TD_DETECT_STROKE)
	elif "models" in frame.info and "ball_detection" in frame.info["models"]:
		size_for_model = [640, 360]

		while len(last_frames_to_merge) < 2:
			last_frames_to_merge.append(frame)

		combined_frames = combine_three_frames(frame.original,
											last_frames_to_merge[0].original,
											last_frames_to_merge[1].original,
											size_for_model[0],
											size_for_model[1])

		x, y = frame.info["models"]["ball_detection"].inference(combined_frames)

		if x is not None:
			x = int(x * (frame.original.shape[1] / size_for_model[0]))
			y = int(y * (frame.original.shape[0] / size_for_model[1]))
			
			frame.info["ball"] = [x, y]
			
			#print("\t" + str(frame.info["ball"]))
			DRW.draw_points(frame, [frame.info["ball"]], DRW.GREEN)
			frame.info["frames_to_copy_ball"] = FRAMES_TO_COPY
			frame.push_to_do(STRK.TD_DETECT_STROKE)
		elif last_frame is not None \
		and (last_frame.info.get("ball", None) is not None or last_frame.info.get("fix_ball", False) is True):
			frame.info["fix_ball"] = True


def post_checks(frames_info, frame, detect_ball_list, detect_ball_list_2):
	# Guarda los frames a checkear
	if "ball" in frame.info:
		detect_ball_list.append(frame)
	#TODO: else: detect_ball_list.clear() ?
	
	# Checkea los freams
	if len(detect_ball_list) > 2:
		# Distancia con el anterior punto
		if "ball" in detect_ball_list[0].info:
			dist_1 = MTH.get_point_distance(detect_ball_list[0].info["ball"],
											detect_ball_list[1].info["ball"])
		else:
			dist_1 = None
		# Distancia con el siguiente punto
		dist_2 = MTH.get_point_distance(detect_ball_list[1].info["ball"],
										detect_ball_list[2].info["ball"])
		
		# Actualiza la lista
		detect_ball_list.pop(0)

		#TODO: recalcular MAX_BALL_DIST y MIN_BALL_DIST en funcion de la distancia entre frames

		# Comprueba la distancia con los frames de alrededor
		if (dist_1 is not None and dist_1 > MAX_BALL_DIST and dist_2 > MAX_BALL_DIST) \
		or (dist_1 is None and dist_2 > MAX_BALL_DIST * 3):
			# Elimina la pelota del frame
			frame_dst = detect_ball_list[0]
			frame_dst.info.pop("ball", None)
			frame_dst.info["fix_ball"] = True

			# Actualiza la informacion
			frames_info[frame_dst.position] = frame_dst.get_info_to_store()
			
			# Actualiza la lista
			#detect_ball_list.pop(0)
		elif dist_1 is not None and dist_1 > MAX_BALL_DIST and dist_2 < MIN_BALL_DIST:
			# Actualiza las listas
			detect_ball_list_2.clear()
			detect_ball_list_2.append(detect_ball_list[0])
		elif len(detect_ball_list_2) > 0 and dist_1 is not None and dist_1 < MIN_BALL_DIST and dist_2 < MIN_BALL_DIST:
			# Actualiza las listas
			detect_ball_list_2.append(detect_ball_list[0])
			if len(detect_ball_list_2) > MAX_STATICS_FRAMES:
				detect_ball_list_2.clear()
		elif len(detect_ball_list_2) > 0 and dist_1 is not None and dist_1 < MIN_BALL_DIST and dist_2 > MAX_BALL_DIST:
			detect_ball_list_2.append(detect_ball_list[0])
			# Elimina la pelota de los frames
			for aux_frame in detect_ball_list_2:
				"""
				"""
				aux_frame.info.pop("ball", None)
				aux_frame.info["fix_ball"] = True
				# Actualiza la informacion
				frames_info[aux_frame.position] = aux_frame.get_info_to_store()
			# Actualiza las listas
			detect_ball_list_2.clear()
			detect_ball_list.pop(0)


def approximate_ball(frame_list):
	new_points = MTH.approximate_points_between(frame_list[0].info["ball"],
												frame_list[-1].info["ball"],
												len(frame_list) - 2)
	for frame, new_point in zip(frame_list[1:-1], new_points):
		if "fix_ball" in frame.info:
			frame.info["ball"] = new_point
			frame.info.pop("fix_ball", None)


def approximate_balls(frames_info, frame, detect_ball_list):
	# Maneja la lista de frames
	if len(detect_ball_list) > 0:
		detect_ball_list.append(frame)
		
	# Maneja los frames para el recalculado de la pelota
	if frame.info.get("fix_ball", False) is False and "ball" in frame.info:
		if len(detect_ball_list) > 2:
			t_ini = time.time()

			# Recalculo de la osicion de los jugadores
			approximate_ball(detect_ball_list)

			t = (time.time() - t_ini) / (len(detect_ball_list) - 2)
			for frame_aux in detect_ball_list[1:-1]:
				# Actualiza el tiempo de procesado
				if "process_time" in frame_aux.info:
					frame_aux.info["process_time"] += t
				
				# Actualiza la info de los frames
				frames_info[frame_aux.position] = frame_aux.get_info_to_store()
		
		# Reinicia la lista
		detect_ball_list.clear()
		detect_ball_list.append(frame)


if __name__ == '__main__':
	fotograma = FTGM.Fotograma(115, cv2.imread("115.png"))
	
	fotograma.info["models"] = {}
	fotograma.info["models"]["ball_detection"] = get_model()
	
	last_frames = []
	last_frames.append(FTGM.Fotograma(113, cv2.imread("113.png")))
	last_frames.append(FTGM.Fotograma(114, cv2.imread("114.png")))

	detect_ball(fotograma, last_frames[0], last_frames)

	cv2.imwrite("FINISHED.png", fotograma.result)
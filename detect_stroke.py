import os
import sys
import cv2
import numpy as np
import torch
import imutils
import argparse
import torch.nn as nn
import tensorflow as tf
import torchvision

from torchvision import transforms

import fotograma as FTGM
import court_points as CRT
import methods.math_methods as MTH
import methods.draw_methods as DRW


TD_DETECT_STROKE = "DETECT_STROKE"
TD_CORRECT_STROKE = "CORRECT_STROKE_SECUENCE"
TD_NEXT_STROKE = "CALCULATE_NEXT_STROKE"

WEAK_POSITION_1 = 2
WEAK_POSITION_2 = 3.2

SERVING_STROKE_TYPE = "SERVING"
ATTACKING_STROKE_TYPE = "ATTACKING"
DEFENDING_STROKE_TYPE = "DEFENDING"
ENDING_STROKE_TYPE = "ENDING"
UNKNOWN_STROKE_TYPE = "UNKNOWN"

IN_BOX_METHOD = "ball_in_box"
HEIGHT_METHOD = "ball_at_same_height"
HEIGHT_AND_WIDTH_METHOD = "custom_box"
WIDTH_TO_SEARCH = 3
HEIGHT_TO_SEARCH = 1.4

STROKE_RECOGNITION_METHOD = HEIGHT_METHOD


num_classes = 3
input_size = 2048
num_layers = 3
hidden_size = 90

strokes_label = ['Forehand', 'Backhand', 'Service/Smash']


def get_models():
	#EXTRACTOR_MODEL = torchvision.models.inception_v3(pretrained=True)
	EXTRACTOR_MODEL = torchvision.models.inception_v3(weights="Inception_V3_Weights.IMAGENET1K_V1")
	EXTRACTOR_MODEL.type(torch.FloatTensor)
	EXTRACTOR_MODEL.eval()
	
	saved_state = torch.load("models/storke_classifier_weights.pth", map_location=torch.device('cpu'))
	model_state = saved_state["model_state"]

	keys = list(model_state.keys()).copy()
	for key in keys:
		if "LSTM." in key:
			new_key = key.replace("LSTM.", "")
			model_state[new_key] = model_state[key]
			del (model_state[key])
	del (model_state["fc.weight"])
	del (model_state["fc.bias"])

	LSTM = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True)
	LSTM.load_state_dict(saved_state["model_state"])
	LSTM.eval()
	LSTM.type(torch.FloatTensor)
	
	return EXTRACTOR_MODEL, LSTM


def get_image_from_box(image, box):
	margin = [int((box[2] - box[0]) * 1.5),
			  int((box[1] - box[3]) * 0.4)]

	box_center = [((box[2] - box[0]) / 2) + box[0],
				  ((box[1] - box[3]) / 2) + box[3]]

	"""
	box_coords = [[int(box_center[0] - margin[0]), int(box_center[1] + margin[1])],
				  [int(box_center[0] + margin[0]), int(box_center[1] + margin[1])],
				  [int(box_center[0] + margin[0]), int(box_center[1] - margin[1])],
				  [int(box_center[0] - margin[0]), int(box_center[1] - margin[1])]]
	new_image = image[int(box_center[1] - margin[1]) : int(box_center[1] + margin[1]),
					  int(box_center[0] - margin[0]) : int(box_center[0] + margin[0])].copy()
	"""
	
	box_coords = [[0, int(box_center[1] + margin[1])],
				  [image.shape[1], int(box_center[1] + margin[1])],
				  [image.shape[1], int(box_center[1] - margin[1])],
				  [0, int(box_center[1] - margin[1])]]

	new_image = image[int(box_center[1] - margin[1]) : int(box_center[1] + margin[1]),
					  0 : image.shape[1]].copy()

	return new_image, box_coords


def preprocess_image(image):
	new_image = imutils.resize(image, 299)
	new_image = new_image.transpose((2, 0, 1)) / 255
	image_tensor = torch.from_numpy(new_image).type(torch.FloatTensor)

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	image_tensor = normalize(image_tensor).unsqueeze(0)

	return image_tensor


def detect_player_stroke(frame: FTGM.Fotograma, player_box):
	if STROKE_RECOGNITION_METHOD == IN_BOX_METHOD:
		# La coordenada de la pelota debe estar dentro de la caja
		if MTH.is_points_inside_box(frame.info["ball"], player_box):
			return True
	
	elif STROKE_RECOGNITION_METHOD == HEIGHT_METHOD:
		box_h = player_box[1] - player_box[3]
		bottom_h = player_box[1] + (HEIGHT_TO_SEARCH - 1) * box_h
		top_h = player_box[3] - (HEIGHT_TO_SEARCH - 1) * box_h

		# La coordenada de la pelota debe estar a la misma altura que la caja
		if frame.info["ball"][1] < bottom_h \
		and frame.info["ball"][1] > top_h:
			return True
	
	elif STROKE_RECOGNITION_METHOD == HEIGHT_AND_WIDTH_METHOD:
		# La coordenada de la pelota debe estar dentro de la nueva caja ensanchada
		box_width = player_box[2] - player_box[0]
		aux_box = [player_box[0] - box_width * WIDTH_TO_SEARCH,
				   player_box[1],
				   player_box[2] + box_width * WIDTH_TO_SEARCH,
				   player_box[3]]
		#TODO: implementar HEIGHT_TO_SEARCH ?

		if MTH.is_points_inside_box(frame.info["ball"], aux_box):
			return True
	
	return False


"""
def detect_player_stroke(frame: FTGM.Fotograma, player_box):
	# Comprobacion de golpe
	player_image, player_image_box = get_image_from_box(frame.original, player_box)
	if MTH.is_point_inside(frame.info["ball"], player_image_box):
		STROKE_MODEL, LSTM = frame.info["stroke_clasification"]
		
		# Extraccion
		player_tensor = preprocess_image(player_image)
		
		with torch.no_grad():
			resultado_extr = STROKE_MODEL(player_tensor)
		resultado_extr = resultado_extr.unsqueeze(1)

		# Clasificado
		softmax = nn.Softmax(dim=1)
		fc = nn.Linear(hidden_size, num_classes)
		batch_size = resultado1.size(0) // 4
		
		with torch.no_grad():
			resultado2, _ = LSTM(resultado1)
			resultado2 = resultado2[:, -resultado1.size(1):, :]
			scores = fc(resultado2.squeeze(0))
			scores = scores[-1].unsqueeze(0)
			probs = softmax(scores).squeeze().cpu().numpy()
		
		prob = np.argmax(probs)
		print(strokes_label[prob])
		
		DRW.draw_area(frame, player_image_box, DRW.FADE_COLORS["b"]["color"])
		return True
"""


def detect_stroke(frame: FTGM.Fotograma):
	if "ball" in frame.info \
	and "models" in frame.info:
	#and "models" in frame.info and "stroke_clasification" in frame.info["models"]:
		for player_type in ["top_player", "bottom_player"]:
			if player_type + "_box" in frame.info:
				frame.info[player_type + "_stroke"] = detect_player_stroke(frame, frame.info[player_type + "_box"])

##################################

def correct_strokes_secuences(frame: FTGM.Fotograma, actual_stroke_list: list, between_stroke_list: list, next_stroke_list: list, frames_info):
	if frame_is_stroke(frame):
		if len(next_stroke_list) > 0:
			# Aniade el frame a la lista final
			next_stroke_list.append(frame)
		elif len(between_stroke_list) > 0:
			# Comprueba un error en la prediccion del golpe
			#if frames_are_same_stroke_type(actual_stroke_list[-1], frame) and len(between_stroke_list) < MAX_STROKE_DIST: #TODO implementar?
			if frames_are_same_stroke_type(actual_stroke_list[-1], frame):
				for aux_frame in between_stroke_list:
					# Arregla el golpe en los frames intermedios
					aux_frame.info[get_stroke_player(frame) + "_stroke"] = True
					frames_info[aux_frame.position] = aux_frame.get_info_to_store()
					
					# Aniade el frame a la lista inicial
					actual_stroke_list.append(aux_frame)
				
				# Aniade el frame a la lista inicial
				actual_stroke_list.append(frame)
				
				# Vacia la lista intermedia
				between_stroke_list.clear()
			else:
				# Aniade el frame a la lista final
				next_stroke_list.append(frame)
		else:
			actual_stroke_list.append(frame)

	else:
		if len(next_stroke_list) == 0 and len(actual_stroke_list) > 0:
			# Aniade el frame a la lista intermedia
			between_stroke_list.append(frame)
		elif len(next_stroke_list) > 0:
			# Reinicia las listas
			actual_stroke_list.clear()
			actual_stroke_list.extend(next_stroke_list)
			between_stroke_list.clear()
			between_stroke_list.append(frame)
			next_stroke_list.clear()

def correct_secuence(secuence, stroke_type, frames_info):
	for frame in secuence:
		frame.info[stroke_type] = True
		frames_info[frame.position] = frame.get_info_to_store()


def correct_strokes_secuences__(frame: FTGM.Fotograma, actual_stroke_list: list, between_stroke_list: list, next_stroke_list: list, frames_info):
	if frame_is_stroke(frame):
		#if len(between_stroke_list) == 0 and len(next_stroke_list) == 0:
		if len(between_stroke_list) == 0:
			if len(actual_stroke_list) == 0:
				# Comienza o continua al golpeo actual
				actual_stroke_list.append(frame)
			elif frames_are_same_stroke_type(actual_stroke_list[-1], frame):
				# Comienza o continua al golpeo actual
				actual_stroke_list.append(frame)
			else:
				# Reinicia el golpeo actual
				actual_stroke_list.clear()
				actual_stroke_list.append(frame)
		#if len(between_stroke_list) > 0 and len(next_stroke_list) == 0:
		elif len(between_stroke_list) > 0:
			if frames_are_same_stroke_type(actual_stroke_list[-1], frame):
				# Arregla el momento intermedio como golpeo
				correct_secuence(between_stroke_list, get_stroke_player(frame) + "_stroke", frames_info)
				actual_stroke_list.extend(between_stroke_list)
				actual_stroke_list.append(frame)
				# Reinicia el momento intermedio
				between_stroke_list.clear()
			else:
				# Reinicia el golpeo actual
				actual_stroke_list.clear()
				actual_stroke_list.append(frame)
				# Reinicia el momento intermedio
				between_stroke_list.clear()
	else:
		if frame.info["IS_COURT"] is False:
			actual_stroke_list.clear()
			between_stroke_list.clear()
			next_stroke_list.clear()
		if len(actual_stroke_list) > 0:
			between_stroke_list.append(frame)











##################################


def frame_is_stroke(frame: FTGM.Fotograma):
	return frame.info.get("top_player_stroke", False) is True \
		or frame.info.get("bottom_player_stroke", False) is True


def frames_are_same_stroke_type(frame_1: FTGM.Fotograma, frame_2: FTGM.Fotograma):
	if frame_1.info.get("top_player_stroke", False) is True \
	and frame_2.info.get("top_player_stroke", False) is True:
		return True
	elif frame_1.info.get("bottom_player_stroke", False) is True \
	and frame_2.info.get("bottom_player_stroke", False) is True:
		return True
	else:
		return False


def get_stroke_player(frame: FTGM.Fotograma):
	for player_type in ["top_player", "bottom_player"]:
		if frame.info.get(player_type + "_stroke", False) is True:
			return player_type


def get_track_of_ball(frame_list, frame_position):
	ball_track = []

	for frame_aux in frame_list:
		ball = frame_aux.info.get("ball", None)
		ball_track.append(ball)
		if frame_aux.position == frame_position:
			break
	
	return ball_track


"""
def stroke_is_winner(next_stroke_frame):
	# Calcula la distancia en la imagen entre jugador y pelota
	ball_dist = MTH.get_point_distance(
		MTH.get_center_of_square(next_stroke_frame.info[get_stroke_player(next_stroke_frame) + "_box"]),
   		next_stroke_frame.info["ball"])

	if ball_dist > 200:
		return True
	else:
		return False
"""


def get_stroke_type(actual_stroke_list, next_stroke_frame, next_stroke_list):
	next_stroke_player = get_stroke_player(next_stroke_frame)
	
	if next_stroke_player is None:
		return UNKNOWN_STROKE_TYPE + " (err)"
	
	weakenss_score = 0
	stroke_info = ""
	
	# Clasifica el golpe segun la posicion del jugador al final del golpe
	leftmost_pos = None
	rightmost_pos = None
	lowest_pos = None
	toppest_pos = None
	for next_frame in next_stroke_list:
		if next_frame.position >= next_stroke_frame.position and next_stroke_player in next_frame.info:
			future_player_position = MTH.find_homography(next_stroke_frame.info["to_topview"], [next_frame.info[next_stroke_player]], to_int=False)[0]
			if leftmost_pos is None or future_player_position[0] < leftmost_pos[0]:
				leftmost_pos = future_player_position
			if rightmost_pos is None or future_player_position[0] > rightmost_pos[0]:
				rightmost_pos = future_player_position
			if lowest_pos is None or future_player_position[1] > lowest_pos[1]:
				lowest_pos = future_player_position
			if toppest_pos is None or future_player_position[1] < toppest_pos[1]:
				toppest_pos = future_player_position
	
	court_middle_x = (CRT.TENNIS_COURT_TOPVIEW_POINTS[0][0] + CRT.TENNIS_COURT_TOPVIEW_POINTS[1][0]) / 2
	aux_left = abs(court_middle_x - leftmost_pos[0]) - 1 if leftmost_pos[0] <= court_middle_x else -1
	aux_right = abs(court_middle_x - rightmost_pos[0]) - 1 if rightmost_pos[0] >= court_middle_x else -1
	middle_distance = max(aux_left, aux_right) * 1.3

	if middle_distance >= 2:
		stroke_info += "\nhorizontal: ATK " + str(round(middle_distance, 2))
	elif middle_distance >= 1:
		stroke_info += "\nhorizontal: MBY " + str(round(middle_distance, 2))
	else:
		stroke_info += "\nhorizontal: DFN " + str(round(middle_distance, 2))
	weakenss_score += middle_distance

	if next_stroke_player == "top_player" and lowest_pos[1] > CRT.TENNIS_COURT_TOPVIEW_POINTS[3][1]:
		vertical_weak = lowest_pos[1] - CRT.TENNIS_COURT_TOPVIEW_POINTS[3][1]
		stroke_info += "\nvertical: DFN (" + str(round(vertical_weak, 2)) + ")"
	elif next_stroke_player == "bottom_player" and toppest_pos[1] < CRT.TENNIS_COURT_TOPVIEW_POINTS[0][1]:
		vertical_weak = CRT.TENNIS_COURT_TOPVIEW_POINTS[0][1] - toppest_pos[1]
		stroke_info += "\nvertical: DFN (" + str(round(vertical_weak, 2)) + ")"
	else:
		vertical_weak = 0
		stroke_info += "\nvertical: MBY"
	weakenss_score -= vertical_weak * 2
	
	# (Media de) distancia que tiene que recorrer el jugador contrario para golpear
	player_dist = 0
	n_bad_frames = 0
	for actual_frame in actual_stroke_list:
		if next_stroke_player in actual_frame.info:
			actual_player_position, future_player_position = MTH.find_homography(next_stroke_frame.info["to_topview"],
				[actual_frame.info[next_stroke_player], next_stroke_frame.info[next_stroke_player]], to_int=False)
			player_dist += MTH.get_point_distance(actual_player_position, future_player_position)
		else:
			n_bad_frames += 1
	player_dist = player_dist / (len(actual_stroke_list) + n_bad_frames)

	if player_dist >= 3:
		stroke_info += "\nplayer: ATK " + str(round(player_dist, 2))
	elif player_dist >= 2:
		stroke_info += "\nplayer: MBY " + str(round(player_dist, 2))
	elif player_dist >= 1:
		stroke_info += "\nplayer: DFN " + str(round(player_dist, 2))
	weakenss_score += player_dist

	if weakenss_score >= 4:
		stroke_type = ATTACKING_STROKE_TYPE
	else:
		stroke_type = DEFENDING_STROKE_TYPE
	
	return stroke_type
	#return stroke_type + " (" + str(round(weakenss_score, 2)) + ")" + stroke_info


def find_stroke_frame(stroke_frame_list):
	max_height_frame = None
	min_haight_frame = None

	for frame in stroke_frame_list:
		if "ball" in frame.info:
			if max_height_frame is None \
			or frame.info["ball"][1] < max_height_frame.info["ball"][1]:
				max_height_frame = frame
			if min_haight_frame is None \
			or frame.info["ball"][1] > min_haight_frame.info["ball"][1]:
				min_haight_frame = frame

	if "top_" in get_stroke_player(stroke_frame_list[0]):
		return max_height_frame
	else:
		return min_haight_frame


def write_destination_of_stroke(actual_stroke_list, between_stroke_list, next_stroke_list):
	# Obtiene el frame del proximo golpeo
	next_stroke_frame = find_stroke_frame(next_stroke_list)
	next_stroke_frame.info["frame_of_impact"] = True

	# Obtiene el tipo del golpe actual (puede que ya haya sido asignado como saque)
	stroke_type = actual_stroke_list[0].info.get("stroke_type", get_stroke_type(actual_stroke_list, next_stroke_frame, next_stroke_list))

	for i_frame in range(len(actual_stroke_list)):
		# Guarda el destino en el frame
		actual_stroke_list[i_frame].info["ball_destination"] = next_stroke_frame.info["ball"]
		actual_stroke_list[i_frame].info["ball_destination_frame"] = next_stroke_frame.position
		actual_stroke_list[i_frame].info["stroke_type"] = stroke_type

		# Calcula el track de la bola
		next_frames = [aux_frame for aux_frame in actual_stroke_list[i_frame:]]
		next_frames.extend(between_stroke_list)
		next_frames.extend(next_stroke_list)
		actual_stroke_list[i_frame].info["ball_track"] = get_track_of_ball(next_frames, next_stroke_frame.position)


def calculate_stroke_direction(frame: FTGM.Fotograma, last_stroke_list: list, actual_stroke_list: list, between_stroke_list: list, next_stroke_list: list, frames_info):
	if frame.info.get("IS_COURT", False) is False:
		# Establece los frames de finalizacion
		ending_stroke_list = []
		if len(next_stroke_list) > 0:
			ending_stroke_list = actual_stroke_list
		elif len(last_stroke_list) > 0:
			ending_stroke_list = last_stroke_list
		for ending_frame in ending_stroke_list:
			ending_frame.info["stroke_type"] = ENDING_STROKE_TYPE
			frames_info[ending_frame.position] = ending_frame.get_info_to_store()

		# Reinicia las listas
		last_stroke_list.clear()
		actual_stroke_list.clear()
		between_stroke_list.clear()
		next_stroke_list.clear()
	else:
		#print(str(frame.position) + "\tis stroke: " + str(frame_is_stroke(frame)) + " \tL:" + str(len(actual_stroke_list)) + "\tB:" + str(len(between_stroke_list)) + "\tN:" + str(len(next_stroke_list)))
		if frame_is_stroke(frame):
			if len(between_stroke_list) > 0 or len(next_stroke_list) > 0:
				# Aniade el frame a la lista final
				next_stroke_list.append(frame)
			else:
				# Establece el frame como de saque
				frame.info["stroke_type"] = SERVING_STROKE_TYPE
				# Aniade el frame a la lista inicial
				actual_stroke_list.append(frame)
		else:
			if len(next_stroke_list) == 0 and len(actual_stroke_list) > 0:
				# Aniade el frame a la lista intermedia
				between_stroke_list.append(frame)
			elif len(next_stroke_list) > 0:
				# Aniade mas frames a la lista
				frames_to_ad = 10
				if frames_to_ad > 0 and (len(next_stroke_list) < frames_to_ad + 1 or frame_is_stroke(next_stroke_list[-frames_to_ad]) is False):
					next_stroke_list.append(frame)
				else:
					# Guarda el destino de la pelota en los frames del golpeo inicial
					write_destination_of_stroke(actual_stroke_list, between_stroke_list, next_stroke_list)
					for actual_frame in actual_stroke_list:
						# Indica que se realentirazra la reproduccion del frame
						actual_frame.info["slow_frame"] = 3
						
						# Reescribe la informacion del frame
						frames_info[actual_frame.position] = actual_frame.get_info_to_store()

					# Reinicia las listas
					last_stroke_list.clear()
					last_stroke_list.extend(actual_stroke_list)
					actual_stroke_list.clear()
					actual_stroke_list.extend([next_frame for next_frame in next_stroke_list if frame_is_stroke(next_frame)])
					between_stroke_list.clear()
					between_stroke_list.append(frame)
					next_stroke_list.clear()
	
import os
import sys
import cv2
import time
import torch
import numpy as np
import argparse
import torchvision

import fotograma as FTGM
import methods.math_methods as MTH
import methods.draw_methods as DRW


PLAYERS_TYPES = ["top_player", "bottom_player"]

PERSON_LABEL_RCNN = 1
PERSON_LABEL_YOLO = 0

#MODEL_TO_USE = "RCNN"
MODEL_TO_USE = "YOLO"

PERSON_MIN_SCORE = 0.20
FRAMES_TO_COPY = 5
MAX_PLAYER_DIST = 60
TRACK_POINTS_TO_MISS = 25

TD_DETECT_PLAYERS = "DETECT_PLAYERS"
TD_POST_CHECKS = "POST_CHECKS_PLAYERS"
TD_APROXIMATE = "APROXIMATE_PLAYERS"
TD_TRACK_PLAYERS = "TRACK_PLAYERS"


def get_model_RCNN():
	PEOPLE_DETECTION_MODEL = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
	PEOPLE_DETECTION_MODEL.type(torch.FloatTensor)
	PEOPLE_DETECTION_MODEL.eval()
	return PEOPLE_DETECTION_MODEL


def get_model_YOLO():
	return torch.hub.load("ultralytics/yolov5:master", "custom", "models/yolov5s.pt")
	#return torch.hub.load("ultralytics/yolov5:master", "yolov5s", pretrained=True)

def get_model():
	if MODEL_TO_USE == "RCNN":
		return get_model_RCNN()
	elif MODEL_TO_USE == "YOLO":
		return get_model_YOLO()


def __mask_img(frame: FTGM.Fotograma):
	img_h, img_w = frame.original.shape[:2]
	
	mask_topview_points = MTH.find_homography(frame.info["to_cameraview"], [[4, 19.885], [14.97, 19.885], [18.97, 0], [0, 0], [0, 36]])
	mask_top_points = np.array([[0, mask_topview_points[4][1]],
								mask_topview_points[0], [mask_topview_points[3][0], mask_topview_points[3][1]/2],
								[mask_topview_points[2][0], mask_topview_points[2][1]/2], mask_topview_points[1],
								[img_w, mask_topview_points[4][1]], [img_w, 0], [0, 0]], dtype=np.int32)

	mask_bottom_points = np.array([[0, img_h], [img_w, img_h],
								   [img_w, mask_topview_points[4][1]], [0, mask_topview_points[4][1]]], dtype=np.int32)

	img_copy = np.copy(frame.original)
	cv2.fillPoly(img_copy , [mask_top_points] , (255,255,255))
	cv2.fillPoly(img_copy , [mask_bottom_points] , (255,255,255))
	return img_copy


def __img_preprocess_RCNN(image):
    processed = image.transpose((2, 0, 1)) / 255
    processed = torch.from_numpy(processed).unsqueeze(0).type(torch.FloatTensor)
    return processed


def detect_people_RCNN(image, model):
	image_proc = __img_preprocess_RCNN(image)

	with torch.no_grad():
		result = model(image_proc)

	persons_boxes = []
	for box, label, score in zip(result[0]['boxes'][:], result[0]['labels'], result[0]['scores']):
		if label == PERSON_LABEL_RCNN and score > PERSON_MIN_SCORE:
			persons_boxes.append(MTH.reshape_square(box.detach().cpu().numpy().astype(int)))
	return persons_boxes


def __img_preprocess_YOLO(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def detect_people_YOLO(image, model):
	image_proc = __img_preprocess_YOLO(image)
	
	result = model(image_proc)

	persons_boxes = []
	for xyxy in result.xyxy[0]:
		if xyxy[-1] == PERSON_LABEL_YOLO and xyxy[-2] > PERSON_MIN_SCORE:
			box = [int(c) for c in xyxy[:4]]
			persons_boxes.append(MTH.reshape_square(box))
	return persons_boxes


def sort_boxes_by_size(boxes):
    return sorted(boxes, key=MTH.get_square_size, reverse=True)


def sort_boxes_by_y(boxes):
	return sorted(boxes, key=lambda box: box[1], reverse=True)


def detect_players(frame: FTGM.Fotograma, last_frame):
	if last_frame is not None \
	and last_frame.info.get("frames_to_copy_players", -1) > 0 \
	and ((last_frame.info.get("top_player", None) is not None and last_frame.info.get("bottom_player", None) is not None) \
	or "fix_players" in last_frame.info):
		# Indica que el frame necesita arreglo
		frame.info["fix_players"] = True

		frame.info["frames_to_copy_players"] = last_frame.info["frames_to_copy_players"] - 1
	else:
		if "to_cameraview" in frame.info:
			# Crea una mascara sobre las partes no importantes de la imagen
			image = __mask_img(frame)
			# Obtiene la posicion de la mitad de la pista
			middle =  MTH.find_homography(frame.info["to_cameraview"], [[9.485, 19.885]])[0]
		else:
			image = frame.original
			middle = [int(image.shape[1] / 2), int(image.shape[0] / 2)]
		
		if MODEL_TO_USE == "RCNN":
			detect_people_function = detect_people_RCNN
		elif MODEL_TO_USE == "YOLO":
			detect_people_function = detect_people_YOLO
		
		# Aplica el modelo
		boxes = detect_people_function(image, frame.info["models"]["people_detection"])
		
		# Separa y ordena las cajas
		top_boxes = sort_boxes_by_y([box for box in boxes if box[1] <= middle[1]])
		bottom_boxes = sort_boxes_by_size([box for box in boxes if box[1] > middle[1]])
		
		#frame.info["top_boxes"] = top_boxes
		#frame.info["bottom_player"] = bottom_boxes
		
		DRW.draw_squares(frame, top_boxes, DRW.BLUE)
		DRW.draw_squares(frame, bottom_boxes, DRW.GREEN)
		if len(top_boxes) > 0:
			frame.info["top_player_box"] = top_boxes[0]
			frame.info["top_player"] = MTH.get_foot_from_square(top_boxes[0])
			DRW.draw_squares(frame, [top_boxes[0]], DRW.YELLOW)
			DRW.draw_points(frame, [frame.info["top_player"]], DRW.YELLOW)
		if len(bottom_boxes) > 0:
			frame.info["bottom_player_box"] = bottom_boxes[0]
			frame.info["bottom_player"] = MTH.get_foot_from_square(bottom_boxes[0])
			DRW.draw_squares(frame, [bottom_boxes[0]], DRW.YELLOW)
			DRW.draw_points(frame, [frame.info["bottom_player"]], DRW.YELLOW)

		if "top_player" in frame.info and "bottom_player" in frame.info:
			# Los siguientes dos fotogramas copiaran la posicion
			frame.info["frames_to_copy_players"] = FRAMES_TO_COPY
		else:
			frame.info["fix_players"] = True


def post_checks(frames_info, frame, detect_players_list):
	# Guarda los frames a checkear
	if "top_player" in frame.info or "bottom_player" in frame.info:
		detect_players_list.append(frame)
	
	# Checkea los freams
	if len(detect_players_list) > 1:
		pop_last = False
		for player_type in PLAYERS_TYPES:
			if player_type in detect_players_list[0].info \
			and player_type in detect_players_list[1].info:
				# Distancia con el anterior punto
				dist = MTH.get_point_distance(detect_players_list[0].info[player_type],
											  detect_players_list[1].info[player_type])
				
				# Numero de frames entre ambos
				n_frames = detect_players_list[1].position - detect_players_list[0].position
				#print("-- Frame [" + str(detect_players_list[0].position) + "][" + str(detect_players_list[1].position) + "] " + str(int(dist)))

				if dist > MAX_PLAYER_DIST * n_frames:
					# Elimina al jugador del frame
					frame_dst = detect_players_list[1]
					frame_dst.info.pop(player_type, None)
					frame_dst.info.pop(player_type + "_box", None)

					# Indica que el frame necesita arrreglo
					frame_dst.info["fix_players"] = True

					# Actualiza la informacion
					frames_info[frame_dst.position] = frame_dst.get_info_to_store()

					# Actualiza el frame malo de la lista
					pop_last = True
		
		# Elimina el frame adecuado de la lista
		if pop_last is True:
			detect_players_list.pop(1)
		else:
			detect_players_list.pop(0)


def approximate_player(frame_list):
	#TODO: probar
	for player in PLAYERS_TYPES:
		new_boxes = MTH.approximate_boxes_between(frame_list[0].info[player + "_box"],
												  frame_list[-1].info[player + "_box"],
												  len(frame_list) - 2)
		for frame, new_box in zip(frame_list[1:-1], new_boxes):
			if "fix_players" in frame.info:
				frame.info[player + "_box"] = new_box
				frame.info[player] = MTH.get_foot_from_square(new_box)

	"""
	for player in PLAYERS_TYPES:
		new_points = MTH.approximate_points_between(frame_list[0].info[player],
													frame_list[-1].info[player],
													len(frame_list) - 2)
		for frame, new_point in zip(frame_list[1:-1], new_points):
			if "fix_players" in frame.info:
				frame.info[player] = new_point
	
	for player_box in ["top_player_box", "bottom_player_box"]:
		new_boxes = MTH.approximate_boxes_between(frame_list[0].info[player_box],
												  frame_list[-1].info[player_box],
												  len(frame_list) - 2)
		for frame, new_box in zip(frame_list[1:-1], new_boxes):
			if "fix_players" in frame.info:
				frame.info[player_box] = new_box
	"""

	for frame in frame_list[1:-1]:
		frame.info.pop("fix_players", None)


def approximate_players(frames_info, frame, detect_players_list):
	# Actualiza las listas
	if len(detect_players_list) > 0:
		detect_players_list.append(frame)
	
	# Maneja los frames para el recalculado de los jugadores
	if frame.info.get("fix_players", False) is False \
	and "top_player" in frame.info \
	and "bottom_player" in frame.info:
		if len(detect_players_list) > 2:
			t_ini = time.time()

			# Recalculo de la posicion de los jugadores
			approximate_player(detect_players_list)

			t = (time.time() - t_ini) / (len(detect_players_list) - 2)
			for frame_aux in detect_players_list[1:-1]:
				# Actualiza el tiempo de procesado
				if "process_time" in frame_aux.info:
					frame_aux.info["process_time"] += t
				
				# Actualiza la info de los frames
				frames_info[frame_aux.position] = frame_aux.get_info_to_store()

		# Reinicia la lista
		detect_players_list.clear()
		detect_players_list.append(frame)


def get_distance_of_track(track_points: list):
	i = 0
	next_i = 0
	track_distance = 0

	# Calcula la distancia
	while i < len(track_points):
		next_i = i + TRACK_POINTS_TO_MISS

		if next_i < len(track_points) and track_points[i] is not None and track_points[next_i] is not None:
			track_distance += MTH.get_point_distance(track_points[i], track_points[next_i])
		
		i = next_i
	
	return track_distance


def track_players(frames_info, frame, track_players_list):
	if frame.info.get("IS_COURT", False) is False:
		# Reinicia la lista
		track_players_list.clear()
	else:
		for player_type in PLAYERS_TYPES:
			# Obtiene el track anterior si existe
			if len(track_players_list) == 0:
				player_track = player_point_track = []
			else:
				player_track = track_players_list[-1].info.get(player_type + "_track", [])
				player_point_track = track_players_list[-1].info.get(player_type + "_point_track", []).copy()

			if frame.info.get("IS_COURT", False) is False:
				player_track = player_point_track = []
			if frame.info.get("frame_of_impact", False) is True and frame.info.get(player_type + "_stroke", False):
				player_point_track = []
				
			# Aniade la posicion actual del jugador
			if player_type not in frame.info or "to_topview" not in frame.info:
				new_point = None
			else:
				new_point = MTH.find_homography(frame.info["to_topview"], [frame.info[player_type]], to_int=False)[0]
			player_track.append(new_point)
			player_point_track.append(new_point)

			# Calcula la distancia
			player_track_distance = get_distance_of_track(player_track)

			# Guarda la nueva informacion
			frame.info[player_type + "_track"] = player_track
			frame.info[player_type + "_point_track"] = player_point_track
			frame.info[player_type + "_distance"] = player_track_distance
			frames_info[frame.position] = frame.get_info_to_store()

		# Actualiza la lista
		track_players_list.clear()
		track_players_list.append(frame)


def track_players__ANT(frames_info, frame, track_players_list):
	if frame.info.get("IS_COURT", False) is True:
		for player_type in PLAYERS_TYPES:
			# Obtiene el track anterior si existe
			if len(track_players_list) == 0:
				track_points = []
			else:
				track_points = track_players_list[-1].info.get(player_type + "_track", [])
			
			# Aniade la posicion actual del jugador
			if player_type not in frame.info or "to_topview" not in frame.info:
				track_points.append(None)
			else:
				track_points.append(MTH.find_homography(frame.info["to_topview"], [frame.info[player_type]], to_int=False)[0])

			i = 0
			next_i = 0
			track_distance = 0
			# Calcula la distancia
			while i < len(track_points):
				next_i = i + TRACK_POINTS_TO_MISS

				# Actualiza next_i para no pasarse
				#if next_i >= len(track_points) and i != len(track_points) - 1:
				#	next_i = len(track_points) - 1
				
				if next_i < len(track_points) and track_points[i] is not None and track_points[next_i] is not None:
					track_distance += MTH.get_point_distance(track_points[i], track_points[next_i])
				
				i = next_i

			# Guarda la nueva informacion
			frame.info[player_type + "_track"] = track_points
			frame.info[player_type + "_distance"] = track_distance #TODO: mejorar/rehacer
			frames_info[frame.position] = frame.get_info_to_store()

	# Actualiza la lista
	track_players_list.clear()
	track_players_list.append(frame)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("img_path", type=str)
	args = parser.parse_args()
	
	if args.img_path is not None:
		if not os.path.isfile(args.img_path):
			print("  Error: imagen '" + args.img_path + "' no encontrada")
			sys.exit(0)
	
	img_original = cv2.imread(args.img_path)
	fotograma = FTGM.Fotograma(0, img_original)
	fotograma.info["models"] = {"people_detection": get_model()}
	
	detect_players(fotograma, [])
	cv2.imwrite("FINISHED.png", fotograma.result)

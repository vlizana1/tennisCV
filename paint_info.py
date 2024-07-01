import os
import sys
import cv2
import math
import numpy as np
import torch
import argparse

import fotograma as FTGM
import court_points as CRT
import detect_stroke as STRK
import detect_players as PLYRS
import methods.math_methods as MTH
import methods.draw_methods as DRW


TD_PAINT_MAP = "PAINT_MAP"

AREAS = [
	{"radius": 2, "a": 3, "b": 8,  "n": 15, "color": "r"},
	{"radius": 6,   "a": 5, "b": 10, "n": 15, "color": "y"}
]


def cut_area_with_court(area, court_points):
	area_lines = MTH.get_area_lines(area)
	court_lines = MTH.get_area_lines(court_points)

	new_area = []
	inside = MTH.is_point_inside(area[0], court_points)
	for line in area_lines:
		# Si esta dentro aniade el inicio de la linea
		if inside is True:
			new_area.append([line[0], line[1]])
		# Comprueba si hay corte con las lineas de pista
		for c_line in court_lines:
			cut_p = MTH.get_cut_point(line, c_line)
			# Si hay corte aniade el punto
			if cut_p is not None and (len(new_area) == 0 or (len(new_area) > 0 and cut_p != new_area[-1])):
				if inside is False:
					inside = True
					if MTH.is_point_inside(c_line[:2], area):
						new_area.append(c_line[:2])
					else:
						new_area.append(c_line[2:])
					new_area.append(cut_p)
				else:
					inside = False
					new_area.append(cut_p)
					if MTH.is_point_inside(c_line[:2], area):
						new_area.append(c_line[:2])
					else:
						new_area.append(c_line[2:])

	if len(new_area) == 0:
		if inside is True:
			new_area = area
		elif MTH.is_point_inside(court_points[0], area) is True:
			new_area = court_points

	return new_area


def paint_distance_map_for_player(frame, player_coord, court_points):
	if len(player_coord) == 2 and len(court_points) == 4:
		# Coordenadas del jugador en el plano
		player_topview_coord = MTH.find_homography(frame.info["to_topview"], player_coord, to_int=False)[0]

		areas = AREAS.copy()

		for area in areas:
			#topview_points = MTH.get_circle_points(player_topview_coord, radius=area["radius"], n_points=area["n"])
			topview_points = MTH.get_elipse_points(player_topview_coord, a=area["a"], b=area["b"], n_points=area["n"])
			area["points"] = MTH.find_homography(frame.info["to_cameraview"], topview_points)
			area["points"] = MTH.intersec_polygons(area["points"], court_points)
			#area["points"] = cut_area_with_court(area["points"], court_points)

		#maps.append({"points": court_points, "color": "g"})
		
		sig_not = None
		for area in reversed(areas):
			if len(area["points"]) > 2:
				if sig_not is not None:
					DRW.draw_area(frame, area["points"], sig_not)
				DRW.draw_area(frame, area["points"], DRW.FADE_COLORS[area["color"]]["color"])
				sig_not = DRW.FADE_COLORS[area["color"]]["not"]


def paint_distance_map(frame: FTGM.Fotograma):
	if "top_player" not in frame.info \
	or "bottom_player" not in frame.info \
	or "to_topview" not in frame.info \
	or "to_cameraview" not in frame.info \
	or "COURT_POINTS" not in frame.info:
		pass
	else:
		# Divide la pista en dos
		middle_court_points = MTH.find_homography(frame.info["to_cameraview"], [[5.37, 19.885], [13.6, 19.885]])
		bottom_court_points = [frame.info["COURT_POINTS"][0], frame.info["COURT_POINTS"][1], middle_court_points[1], middle_court_points[0]]
		top_court_points = [middle_court_points[0], middle_court_points[1], frame.info["COURT_POINTS"][2], frame.info["COURT_POINTS"][3]]
		
		# Pinta la info del jugador inferior
		paint_distance_map_for_player(frame, frame.info["top_player"], top_court_points)
		paint_distance_map_for_player(frame, frame.info["bottom_player"], bottom_court_points)


def paint_progress(frame: FTGM.Fotograma):
	"""
	## IS_COURT
	if frame.info.get("IS_COURT", False) is False:
		DRW.draw_points(frame, [[20, 20]], DRW.RED)
	else:
		DRW.draw_points(frame, [[20, 20]], DRW.GREEN)

	## COURT_POINTS
	for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
		if court_type in frame.info:
			DRW.draw_points(frame, frame.info[court_type], DRW.RED)
			DRW.draw_lines(frame, [frame.info[court_type][0] + frame.info[court_type][-1],
								   frame.info[court_type][1] + frame.info[court_type][-2]], DRW.RED)
	"""
	## PLAYERS
	for player_type in PLYRS.PLAYERS_TYPES:
		if player_type in frame.info:
			DRW.draw_points(frame, [frame.info[player_type]], DRW.YELLOW)
		if player_type + "_box" in frame.info:
			DRW.draw_squares(frame, [frame.info[player_type + "_box"]], DRW.YELLOW)
	

	## BALL
	if "ball" in frame.info:
		DRW.draw_points(frame, [frame.info["ball"]], DRW.GREEN)

	## STROKE
	for player_type in PLYRS.PLAYERS_TYPES:
		if frame.info.get(player_type + "_stroke", False) is True and player_type + "_box" in frame.info:
			DRW.draw_box(frame, frame.info[player_type + "_box"], DRW.FADE_COLORS["r"]["color"])
			#DRW.draw_points(frame, [frame.info[player_type]], DRW.RED)
			#DRW.draw_squares(frame, [frame.info[player_type + "_box"]], DRW.RED)
	
	"""
	## FRAME POSITION
	DRW.write_text(frame, "FRM: " + str(frame.position), [20, 60])
	"""


def show_players_tracks_on_point(frame: FTGM.Fotograma):
	for player_type in PLYRS.PLAYERS_TYPES:
		if player_type + "_point_track" in frame.info and "to_cameraview" in frame.info:
			track_points = frame.info[player_type + "_point_track"]

			track_lines = []
			for i in range(len(track_points) - 1):
				if track_points[i] is not None and track_points[i+1] is not None:
					aux_ini, aux_fin = MTH.find_homography(frame.info["to_cameraview"], [track_points[i], track_points[i+1]])
					track_lines.append([aux_ini[0], aux_ini[1], aux_fin[0], aux_fin[1]])

			DRW.draw_lines(frame, track_lines, DRW.YELLOW)


def show_players_tracks(frame: FTGM.Fotograma, last_frame: FTGM.Fotograma):
	for player_type in PLYRS.PLAYERS_TYPES[1:]:
		if player_type + "_track" in frame.info and "to_cameraview" in frame.info:
			track_points = frame.info[player_type + "_track"]

			i = 0
			track_lines = []
			# Calcula la distancia viajada
			while i < len(track_points):
				next_i = i + PLYRS.TRACK_POINTS_TO_MISS

				# Actualiza next_i para no pasarse
				if next_i >= len(track_points) and i != len(track_points) - 1:
					next_i = len(track_points) - 1
				
				if next_i < len(track_points) and track_points[i] is not None and track_points[next_i] is not None:
					aux_ini, aux_fin = MTH.find_homography(frame.info["to_cameraview"], [track_points[i], track_points[next_i]])
					track_lines.append([aux_ini[0], aux_ini[1], aux_fin[0], aux_fin[1]])
				
				i = next_i
			
			#TODO: marcar posiciones de golpeo?

			DRW.draw_lines(frame, track_lines, DRW.YELLOW)

			"""
			# Obtiene el track anterior
			if last_frame is None:
				track_points = []
			else:
				track_points = last_frame.info.get(player_type + "_track", [])
			
			# Aniade la posicion actual al track
			if player_type in frame.info:
				track_points.append(MTH.find_homography(frame.info["to_topview"], [frame.info[player_type]], to_int=False)[0])
				#track_points.append(frame.info[player_type])
			else:
				track_points.append(None)
			
			# Guarda el track nuevo en el frame
			frame.info[player_type + "_track"] = track_points

			i = 0
			distance = 0
			track_lines = []
			points_to_jump = 20
			# Calcula la distancia viajada
			while i < len(track_points):
				next_i = i + points_to_jump
				if next_i < len(track_points) and track_points[i] is not None and track_points[next_i] is not None:
					distance += MTH.get_point_distance(track_points[i], track_points[next_i])

					aux_ini, aux_fin = MTH.find_homography(frame.info["to_cameraview"], [track_points[i], track_points[next_i]])
					track_lines.append([aux_ini[0], aux_ini[1], aux_fin[0], aux_fin[1]])
				i = next_i
			
			frame.info[player_type + "_distance"] = distance
			DRW.draw_lines(frame, track_lines, DRW.YELLOW)
			"""


def show_ball_track(frame: FTGM.Fotograma):
	# Dibuja la trayectoria de la pelota hasta su destino
	if "ball_track" in frame.info:
		if frame.info.get("frame_of_impact", False) is True:
			DRW.draw_track(frame, frame.info["ball_track"], DRW.RED)
		else:
			DRW.draw_track(frame, frame.info["ball_track"], DRW.GREEN)
	
	# Dibuja el destino de la pelota
	if "ball" in frame.info and "ball_destination" in frame.info:
		DRW.draw_points(frame, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(frame, [frame.info["ball_destination"]], DRW.RED)


def show_stroke_predictions(frame: FTGM.Fotograma):
	# Indica el tipo de golpe realizado
	if "stroke_type" in frame.info:
		third_height = frame.original.shape[0] / 3
		aux_height = 0
		for player_type in PLYRS.PLAYERS_TYPES:
			aux_height += third_height
			if frame.info.get(player_type + "_stroke", False) is True:
				if player_type + "_box" in frame.info:
					position_to_write = frame.info[player_type + "_box"][2:]
					position_to_write[0] += 40
				else:
					position_to_write = [0, int(aux_height)]

				DRW.write_text(
					frame,
					text=frame.info["stroke_type"],
					position=position_to_write,
					background=True)


def show_statistics(frame: FTGM.Fotograma):
	img_height = frame.original.shape[0]

	for text_height, player_type in zip([img_height/3, img_height*2/3], PLYRS.PLAYERS_TYPES):
		if player_type + "_distance" in frame.info:
			text = player_type.upper() + ":\n" + \
				   " -Distancia recorrida: " + str(round(frame.info[player_type + "_distance"], 1))
			DRW.write_text(
				frame,
				text=text,
				position=[20, int(text_height)],
				background=True)


#TODO: sustituir por FTGM.copy_frame_to_mem ?
def copy_frame_to_mem(frame: FTGM.Fotograma):
	FRM_COPY = FTGM.Fotograma(-1, frame.original.copy())
	FRM_COPY.set_info(frame.get_info_to_store())
	FRM_COPY.info["to_topview"] = frame.info.get("to_topview", None)
	FRM_COPY.info["to_cameraview"] = frame.info.get("to_cameraview", None)
	return FRM_COPY


def get_images_to_mem(frame: FTGM.Fotograma):
	## PARA vid_r.m4
	"""
	if frame.position == 82:
		FRM_COPY = copy_frame_to_mem(frame)
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
			if court_type in frame.info:
				DRW.draw_points(FRM_COPY, frame.info[court_type], DRW.RED)
				DRW.draw_lines(FRM_COPY, [frame.info[court_type][0] + frame.info[court_type][-1],
								frame.info[court_type][1] + frame.info[court_type][-2]], DRW.RED)
		cv2.imwrite("MEM/court_points_cortados_por_cartel.png", FRM_COPY.result)
	"""

	## PARA vid_3.mp4
	"""
	if frame.position == 214:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		cv2.imwrite("MEM/trayectoria.png", FRM_COPY.result)



	if frame.position == 115:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/impacto_antes.png", FRM_COPY.result)
	if frame.position == 132:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.RED)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/impacto.png", FRM_COPY.result)
	if frame.position == 138:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/impacto_despues.png", FRM_COPY.result)
	


	if frame.position == 18:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		DRW.write_text(FRM_COPY,
						text=frame.info["stroke_type"].split("\n")[0],
						position=frame.info["bottom_player_box"][2:],
						background=True)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_saque.png", FRM_COPY.result)
	


	if frame.position == 282:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		DRW.write_text(FRM_COPY,
						text=frame.info["stroke_type"].split("\n")[0],
						position=frame.info["bottom_player_box"][2:],
						background=True)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_ataque_ini.png", FRM_COPY.result)
	if frame.position == 360:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_ataque_fin.png", FRM_COPY.result)
	
	if frame.position == 520:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		DRW.write_text(FRM_COPY,
						text=frame.info["stroke_type"].split("\n")[0],
						position=frame.info["top_player_box"][2:],
						background=True)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_ending_ini.png", FRM_COPY.result)
	if frame.position == 595:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_ending_fin.png", FRM_COPY.result)

	if frame.position == 77:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		DRW.write_text(FRM_COPY,
					text=frame.info["stroke_type"].split("\n")[0],
					position=frame.info["bottom_player_box"][2:],
					background=True)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_defensa_ini.png", FRM_COPY.result)
	if frame.position == 114:
		FRM_COPY = copy_frame_to_mem(frame)
		DRW.draw_track(FRM_COPY, frame.info["ball_track"], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [frame.info["ball_destination"]], DRW.RED)
		show_players_tracks_on_point(FRM_COPY)
		cv2.imwrite("MEM/golpeo_defensa_fin.png", FRM_COPY.result)



	if frame.position == 352:
		FRM_COPY = copy_frame_to_mem(frame)
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.YELLOW)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.YELLOW)
			if frame.info.get(player_type + "_stroke", False) is True and player_type + "_box" in frame.info:
				DRW.draw_box(FRM_COPY, frame.info[player_type + "_box"], DRW.FADE_COLORS["r"]["color"])
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/golpeo.png", FRM_COPY.result)
	if frame.position == 593:
		FRM_COPY = copy_frame_to_mem(frame)
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.YELLOW)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.YELLOW)
			if frame.info.get(player_type + "_stroke", False) is True and player_type + "_box" in frame.info:
				DRW.draw_box(FRM_COPY, frame.info[player_type + "_box"], DRW.FADE_COLORS["r"]["color"])
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/passing.png", FRM_COPY.result)
	if frame.position == 301:
		FRM_COPY = copy_frame_to_mem(frame)
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.YELLOW)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.YELLOW)
			if frame.info.get(player_type + "_stroke", False) is True and player_type + "_box" in frame.info:
				DRW.draw_box(FRM_COPY, frame.info[player_type + "_box"], DRW.FADE_COLORS["r"]["color"])
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/no_golpeo.png", FRM_COPY.result)



	if frame.position == 462:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/aprox_ball.png", FRM_COPY.result)
	if 462 < frame.position < 469:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/aprox_ball.png")
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.RED)
		cv2.imwrite("MEM/aprox_ball.png", FRM_COPY.result)
	if frame.position == 469:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/aprox_ball.png")
		DRW.draw_points(FRM_COPY, [frame.info["ball"]], DRW.GREEN)
		cv2.imwrite("MEM/aprox_ball.png", FRM_COPY.result)



	if frame.position == 558:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.YELLOW)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.YELLOW)
		cv2.imwrite("MEM/aprox_players.png", FRM_COPY.result)
	if 558 < frame.position < 564:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/aprox_players.png")
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.RED)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.RED)
		cv2.imwrite("MEM/aprox_players.png", FRM_COPY.result)
	if frame.position == 564:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/aprox_players.png")
		for player_type in PLYRS.PLAYERS_TYPES:
			DRW.draw_points(FRM_COPY, [frame.info[player_type]], DRW.YELLOW)
			DRW.draw_squares(FRM_COPY, [frame.info[player_type + "_box"]], DRW.YELLOW)
		cv2.imwrite("MEM/aprox_players.png", FRM_COPY.result)



	if frame.position == 172 - 4:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
			if court_type in frame.info:
				DRW.draw_points(FRM_COPY, frame.info[court_type], DRW.GREEN, fill=False)
		cv2.imwrite("MEM/smooth_points.png", FRM_COPY.result)
	if 172 - 4 < frame.position < 172 + 4:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/smooth_points.png")
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
			if court_type in frame.info:
				DRW.draw_points(FRM_COPY, frame.info[court_type], DRW.GREEN, fill=False)
		cv2.imwrite("MEM/smooth_points.png", FRM_COPY.result)
	if frame.position == 172 + 4:
		FRM_COPY = FTGM.copy_frame_to_mem(frame)
		FRM_COPY.result = cv2.imread("MEM/smooth_points.png")
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
			if court_type in frame.info:
				DRW.draw_points(FRM_COPY, frame.info[court_type], DRW.GREEN)
		DRW.draw_points(FRM_COPY, [[406, 814], [1524, 814], [1306, 292], [623, 293], [218, 817], [1713, 817], [1423, 292], [508, 293]], DRW.RED)
		cv2.imwrite("MEM/smooth_points.png", FRM_COPY.result)
	"""

	## PARA vid_7.m4
	if frame.position == 82:
		FRM_COPY = copy_frame_to_mem(frame)
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
			if court_type in frame.info:
				DRW.draw_points(FRM_COPY, frame.info[court_type], DRW.RED)
				DRW.draw_lines(FRM_COPY, [frame.info[court_type][0] + frame.info[court_type][-1],
								frame.info[court_type][1] + frame.info[court_type][-2]], DRW.RED)
		cv2.imwrite("MEM/court_points_cortados_por_cartel.png", FRM_COPY.result)
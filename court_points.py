import os
import sys
import cv2
import json
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

from typing import List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import splprep, splev

import fotograma as FTGM
import detect_ball as BLL
import detect_players as PLYRS
import methods.math_methods as MTH
import methods.draw_methods as DRW


IMAGE_HEIGHT = 460

TENNIS_COURT_TOPVIEW_POINTS = [[5.37, 30.97], [13.6, 30.97], [13.6, 8], [5.37, 8]]
TENNIS_COURT_TOPVIEW_POINTS_DOUBLES = [[4, 30.97], [14.97, 30.97], [14.97, 8], [4, 8]]

TD_COURT_POINTS = "COURT_POINTS"
TD_CHECK_BAD_POINTS = "CHECK_BAD_POINTS"
TD_COMPARE_COURT_LINES = "COMPARE_COURT_LINES"
TD_GET_HOMOGRAPHY = "GET_HOMOGRAPHY"
TD_CREATES_MISSING_POINTS = "CREATES_MISSING_POINTS"
TD_SMOOTH_POINTS = "SMOOTH_COURT_POINTS"


def get_h_v_lines(lines: np.ndarray,
				  min_vert_len: int = 70,
				  min_horiz_len: int = 100,
				  min_vert_slope: float = 1.5,
				  max_horiz_slope: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
	horizontal_lines = []
	vertical_lines = []
	for line in lines:
		distance = np.sqrt((line[0] - line[2])**2 + (line[1] - line[3])**2)
		slope = abs(MTH.get_line_slope(line))
		#print("-- d:" + str(distance) + ":" + str(min_vert_len) + " | s:" + str(abs(slope)) + ">" + str(min_vert_slope) + "<" + str(max_vert_slope))
		if distance > min_horiz_len and slope < max_horiz_slope:
			horizontal_lines.append(line)
		#elif distance > min_vert_len and abs(slope) > min_vert_slope and abs(slope) < max_vert_slope:
		elif distance > min_vert_len and slope > min_vert_slope:
			vertical_lines.append(line)
	horizontal_lines = np.array(horizontal_lines)
	vertical_lines = np.array(vertical_lines)
	return horizontal_lines, vertical_lines


def unify_lines(lines):
	u_line = None
	for line in lines:
		if u_line is None:
			if line[1] > line[3]:
				u_line = line
			else:
				u_line = [line[2], line[3], line[0], line[1]]
		else:
			if line[1] > u_line[1]:
				u_line[0], u_line[1] = line[0], line[1]
			if line[3] > u_line[1]:
				u_line[0], u_line[1] = line[2], line[3]
			if line[1] < u_line[3]:
				u_line[2], u_line[3] = line[0], line[1]
			if line[3] < u_line[3]:
				u_line[2], u_line[3] = line[2], line[3]
	return u_line


def unify_lines_by_cuts(lines, grid_tam, n_cutting_lines=10, sensitivity=10):
	cutting_lines = []
	cutting_line_h = grid_tam[1] / (n_cutting_lines + 1)
	for i in range(1, n_cutting_lines + 1):
		cutting_lines.append([0, int(cutting_line_h * i), grid_tam[0], int(cutting_line_h * i)])

	to_unify_dict = {}
	for i in range(len(lines)):
		to_unify_dict[i] = lines[i]

	for cutting_line in cutting_lines:
		for i in range(len(lines)):
			for j in range(i+1, len(lines)):
				line_i = to_unify_dict.get(i, None)
				line_j = to_unify_dict.get(j, None)
				if line_i is not None and line_j is not None:
					p1 = MTH.get_cut_point(line_i, cutting_line)
					p2 = MTH.get_cut_point(line_j, cutting_line)
					if p1 is not None and p2 is not None and MTH.get_point_distance(p1, p2) < sensitivity:
						to_unify_dict[i] = unify_lines([line_i, line_j])
						to_unify_dict[j] = None

	unifyed_list = []
	for unifyed_line in to_unify_dict.values():
		if unifyed_line is not None:
			unifyed_list.append(list(unifyed_line))
	return unifyed_list


def extract_court_points(frame: FTGM.Fotograma, show_progress=False):
	img_h, img_w = frame.original.shape[:2]

	# Image to gray
	img_gray = cv2.cvtColor(frame.original, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)[1]
	img_gray = cv2.dilate(img_gray, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=2)
	
	#cv2.imwrite("FINISHED_GRAY.png", img_gray) #TODO
	#cv2.imwrite("FINISHED_ORG.png", frame.original) #TODO

	# Apply HoughLines
	lines_hough = cv2.HoughLinesP(img_gray,
								  rho=1,
								  theta=np.pi / 180,
								  threshold=80,
								  lines=np.array([]),
								  minLineLength=img_h * 0.25,
								  maxLineGap=img_h * 0.03)
	if lines_hough is None or len(lines_hough) < 1:
		return
	lines_hough = lines_hough.squeeze()
	
	#DRW.draw_lines(frame, lines_hough, DRW.GREEN) #TODO
	
	# Extract vertical lines
	horizontal_lines, vertical_lines = get_h_v_lines(lines_hough,
													 min_vert_len=img_h * 0.25,
													 min_horiz_len=img_h * 0.25,
													 min_vert_slope=1.3,
													 max_horiz_slope=0.01)
	
	#DRW.draw_lines(frame, vertical_lines, DRW.RED) #TODO
	
	# Group
	groups_by_cuts = unify_lines_by_cuts(vertical_lines, [img_w, img_h])
	# Filter
	correct_size_lines = [line for line in groups_by_cuts if MTH.get_line_size(line) >= img_h * 0.35]
	# Sort de lines from left to right
	sorted_court_lines = sorted(correct_size_lines, key=lambda line: line[0])
	
	if len(sorted_court_lines) == 0:
		#frame.push_to_do("FINISH")
		pass
	elif len(sorted_court_lines) == 4:
		frame.info["COURT_POINTS"] = [sorted_court_lines[1][:2], sorted_court_lines[-2][:2],
									  sorted_court_lines[-2][2:], sorted_court_lines[1][2:]]
		frame.info["COURT_POINTS_DOUBLES"] = [sorted_court_lines[0][:2], sorted_court_lines[-1][:2],
											  sorted_court_lines[-1][2:], sorted_court_lines[0][2:]]
		
		if show_progress: DRW.draw_lines(frame, sorted_court_lines, DRW.RED)
		if show_progress: DRW.draw_points(frame, frame.info["COURT_POINTS"], DRW.RED)
		if show_progress: DRW.draw_points(frame, frame.info["COURT_POINTS_DOUBLES"], DRW.RED)
		
		#cv2.imwrite("FINISHED.png", frame.result) #TODO

		frame.push_to_do(TD_CHECK_BAD_POINTS)
	else:
		if show_progress: DRW.draw_lines(frame, sorted_court_lines, DRW.GREEN)
		frame.info["COURT_LINES"] = sorted_court_lines
		frame.push_to_do(TD_COMPARE_COURT_LINES)


def check_bad_points(frame, last_frame):
	if last_frame is not None and "COURT_POINTS" in last_frame.info:
		# Puntos actuales
		all_points = MTH.merge_lists(frame.info["COURT_POINTS"], frame.info["COURT_POINTS_DOUBLES"])
		# Puntos anteriores
		all_last = MTH.merge_lists(last_frame.info["COURT_POINTS"] , last_frame.info["COURT_POINTS_DOUBLES"])

		# Calcula distancias respecto de los puntos anteriores
		dist_list = []
		sum_avarage = 0
		for p1, p2 in zip(all_points, all_last):
			d = MTH.get_point_distance(p1, p2)
			sum_avarage += d
			dist_list.append(d)
		
		# Si hay puntos movidos los elimina
		some_to_create = False
		for i, dist in enumerate(dist_list):
			if dist > 3 * ((sum_avarage - dist) / (len(dist_list)-1)):
				all_points[i] = None
				some_to_create = True

		# Asigna los puntos sin los movidos
		if some_to_create is True:
			frame.info["COURT_POINTS"] = all_points[:4]
			frame.info["COURT_POINTS_DOUBLES"] = all_points[4:]

	frame.push_to_do(TD_GET_HOMOGRAPHY)


def check_missing_lines(frame, last_frame, show_progress=False):
	if "COURT_LINES" not in frame.info or last_frame is None or "COURT_POINTS" not in last_frame.info:
		#frame.push_to_do("FINISH")
		pass
	else:
		all_last_l = []
		all_last_l.append(list(last_frame.info["COURT_POINTS_DOUBLES"][0]) + list(last_frame.info["COURT_POINTS_DOUBLES"][-1]))
		all_last_l.append(list(last_frame.info["COURT_POINTS"][0]) + list(last_frame.info["COURT_POINTS"][-1]))
		all_last_l.append(list(last_frame.info["COURT_POINTS"][1]) + list(last_frame.info["COURT_POINTS"][-2]))
		all_last_l.append(list(last_frame.info["COURT_POINTS_DOUBLES"][1]) + list(last_frame.info["COURT_POINTS_DOUBLES"][-2]))

		n_good_lines = 0
		max_dist = frame.original.shape[0] * 0.02
		court_lines = []
		for last_line in all_last_l:
			finded = False
			for line in frame.info["COURT_LINES"]:
				if MTH.get_point_distance(line[:2], last_line[:2]) <= max_dist \
				or MTH.get_point_distance(line[2:], last_line[2:]) <= max_dist:
					court_lines.append(line)
					n_good_lines += 1
					finded = True
					break
			if finded is False:
				court_lines.append([None]*4)
		
		if n_good_lines >= 2:
			frame.info["COURT_POINTS"] = [court_lines[1][:2], court_lines[-2][:2],
										  court_lines[-2][2:], court_lines[1][2:]]
			frame.info["COURT_POINTS_DOUBLES"] = [court_lines[0][:2], court_lines[-1][:2],
										          court_lines[-1][2:], court_lines[0][2:]]

			if show_progress: DRW.draw_points(frame, frame.info["COURT_POINTS"], DRW.RED)
			if show_progress: DRW.draw_points(frame, frame.info["COURT_POINTS_DOUBLES"], DRW.RED)

			frame.push_to_do(TD_GET_HOMOGRAPHY)
		else:
			#frame.push_to_do("FINISH")
			pass


def create_homographys(frame, show_progress=False):
	if "COURT_POINTS" not in frame.info:
		#frame.push_to_do("FINISH")
		pass
	else:

		points_src = MTH.merge_lists(frame.info["COURT_POINTS"], frame.info["COURT_POINTS_DOUBLES"])
		points_dst = MTH.merge_lists(TENNIS_COURT_TOPVIEW_POINTS, TENNIS_COURT_TOPVIEW_POINTS_DOUBLES)
		
		for i in reversed(range(len(points_src))):
			if points_src[i] is None or points_src[i][0] is None or points_src[i][1] is None:
				points_src.pop(i)
				points_dst.pop(i)
		
		#TODO: aniadir comprobacion de puntos minimos
		
		frame.info["to_topview"] = MTH.create_homography(points_src, points_dst)
		frame.info["to_cameraview"] = MTH.create_homography(points_dst, points_src)
		
		middle =  MTH.find_homography(frame.info["to_cameraview"], [[9.485, 19.885]])[0]
		if show_progress: DRW.draw_points(frame, [middle], DRW.BLUE)
		
		missing = False
		for p in points_src:
			if p is None or p[0] is None or p[1] is None:
				missing = True

		if len(points_src) < 8 or missing is True:
			frame.push_to_do(TD_CREATES_MISSING_POINTS)
		else:
			frame.push_to_do(PLYRS.TD_DETECT_PLAYERS)
			frame.push_to_do(BLL.TD_DETECT_BALL)


def create_missing_points(frame, show_progress=False):
	for i in range(len(frame.info["COURT_POINTS"])):
		if frame.info["COURT_POINTS"][i] is None \
		or frame.info["COURT_POINTS"][i][0] is None \
		or frame.info["COURT_POINTS"][i][1] is None:
			frame.info["COURT_POINTS"][i] = list(MTH.find_homography(frame.info["to_cameraview"], TENNIS_COURT_TOPVIEW_POINTS[i])[0])
			if show_progress: DRW.draw_points(frame, [frame.info["COURT_POINTS"][i]], DRW.BLUE)

	for i in range(len(frame.info["COURT_POINTS_DOUBLES"])):
		if frame.info["COURT_POINTS_DOUBLES"][i] is None \
		or frame.info["COURT_POINTS_DOUBLES"][i][0] is None \
		or frame.info["COURT_POINTS_DOUBLES"][i][1] is None:
			frame.info["COURT_POINTS_DOUBLES"][i] = list(MTH.find_homography(frame.info["to_cameraview"], TENNIS_COURT_TOPVIEW_POINTS_DOUBLES[i])[0])
			if show_progress: DRW.draw_points(frame, [frame.info["COURT_POINTS_DOUBLES"][i]], DRW.BLUE)

	frame.push_to_do(PLYRS.TD_DETECT_PLAYERS)
	frame.push_to_do(BLL.TD_DETECT_BALL)


def smooth_court_points(frames_info, frame, court_points_list):
	if "COURT_POINTS" in frame.info and "COURT_POINTS_DOUBLES" in frame.info:
		court_points_list.append(frame)
	else:
		court_points_list.clear()

	if len(court_points_list) > 8:
		smoothed = {}
		for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
		#for court_type in ["COURT_POINTS"]:
			# Hace la media de los puntos de los frames
			smoothed[court_type] = [[0, 0], [0, 0], [0, 0], [0, 0]]
			for aux_frame in court_points_list:
				for i_point in range(len(aux_frame.info[court_type])):
					smoothed[court_type][i_point][0] += aux_frame.info[court_type][i_point][0]
					smoothed[court_type][i_point][1] += aux_frame.info[court_type][i_point][1]
			for i_point in range(len(smoothed[court_type])):
				smoothed[court_type][i_point][0] = int(smoothed[court_type][i_point][0] / len(court_points_list))
				smoothed[court_type][i_point][1] = int(smoothed[court_type][i_point][1] / len(court_points_list))
			
			# Guarda la media 
			i_midel = int(len(court_points_list) / 2)
			court_points_list[i_midel].info[court_type + "_smoothed"] = smoothed[court_type]

			# Guarda la nueva info del frame
			if court_type + "_smoothed" in court_points_list[0].info:
				court_points_list[0].info[court_type] = court_points_list[0].info[court_type + "_smoothed"]
				frames_info[court_points_list[0].position] = court_points_list[0].get_info_to_store()

		
		# Elimina el primer elemento de la lista
		court_points_list.pop(0)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("img_path", type=str)
	args = parser.parse_args()
	
	if args.img_path is not None and not os.path.isfile(args.img_path):
		print("  Error: imagen '" + args.img_path + "' no encontrada")
		sys.exit(0)
	
	img_original = cv2.imread(args.img_path)
	frame = FTGM.Fotograma(0, img_original)
	
	
	last_frame = FTGM.Fotograma(0, None)
	last_frame.info["COURT_POINTS"] = [[414, 815], [1500, 769], [1310, 293], [622, 297]]
	last_frame.info["COURT_POINTS_DOUBLES"] = [[220, 817], [1715, 817], [1419, 292], [508, 293]]
	
	to_do = "COURT_POINTS"
	while to_do != "FINISH":
		if to_do == "COURT_POINTS":
			extract_court_points(frame)
		elif to_do == "CHECK_BAD_POINTS":
			check_bad_points(frame, last_frame)
		elif to_do == "COMPARE_COURT_LINES":
			check_missing_lines(frame, last_frame)
		elif to_do == "GET_HOMOGRAPHY":
			create_homographys(frame)
		elif to_do == "CREATES_MISSING_POINTS":
			create_missing_points(frame)
		to_do = frame.pop_to_do()
	
	cv2.imwrite("FINISHED.png", frame.result)
















import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import io
import sys
import cv2
import json
import time
import torch
import numpy as np
import pprint
import argparse

import is_court as ISCRT
import fotograma as FTGM
import paint_info as PNT
import detect_ball as BLL
import court_points as CRT
import detect_stroke as STRK
import detect_players as PLYRS
import methods.draw_methods as DRW
import methods.math_methods as MTH


FINISHED_VIDEO_NAME_SUFIX = "_FINISHED"
FRAMES_INFO_JSON_SUFIX = "_INFO"
FRAMES_TRUTH_JSON_SUFIX = "_TRUTH"

MAX_FRAMES_TO_MANAGE = 5000 # ?

SHOW_PROGRESS = True

ALL_TO_DOS = [
	ISCRT.TD_IS_COURT, ISCRT.TD_CORRECT_WITH_LAST,
	CRT.TD_COURT_POINTS, CRT.TD_CHECK_BAD_POINTS, CRT.TD_COMPARE_COURT_LINES, CRT.TD_GET_HOMOGRAPHY, CRT.TD_CREATES_MISSING_POINTS,
	PLYRS.TD_DETECT_PLAYERS,
	BLL.TD_DETECT_BALL, STRK.TD_DETECT_STROKE,
	PLYRS.TD_POST_CHECKS, BLL.TD_POST_CHECKS,
	PLYRS.TD_APROXIMATE, BLL.TD_APROXIMATE,
	STRK.TD_CORRECT_STROKE,
	PLYRS.TD_TRACK_PLAYERS,
	STRK.TD_NEXT_STROKE,
	CRT.TD_SMOOTH_POINTS,
]


def prepare_models():
	all_models = {}

	if ISCRT.TD_IS_COURT in ALL_TO_DOS:
		all_models["is_court"] = ISCRT.get_model()
	
	if PLYRS.TD_DETECT_PLAYERS in ALL_TO_DOS:
		all_models["people_detection"] = PLYRS.get_model()
	
	if PLYRS.TD_DETECT_PLAYERS in ALL_TO_DOS:
		all_models["ball_detection"] = BLL.get_model()
		#all_models["stroke_clasification"] = STRK.get_model()

	return all_models


def create_video_to_write(video_path, new_name=None):
	# Abre el video de origen
	source_video = cv2.VideoCapture(video_path)

	# Obtiene sus caracteristicas
	fps = int(source_video.get(cv2.CAP_PROP_FPS))
	frame_width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

	# Cierra el video
	source_video.release()

	# Obtiene el nuevo nombre
	if new_name is None:
		new_name = "".join(video_path.split(".")[:-1]) + FINISHED_VIDEO_NAME_SUFIX + ".mp4"

	# Crea el objeto de escritura para el nuevo video
	finished_video = cv2.VideoWriter(new_name, fourcc, fps, (frame_width, frame_height))
	
	return finished_video


def process_frame(frame: FTGM.Fotograma, frame_buffer: FTGM.Buffer, all_models={}, to_dos=ALL_TO_DOS):
	t_ini = time.time()
	
	frame.push_to_do(ISCRT.TD_IS_COURT)
	frame.info["models"] = all_models
	
	to_do = frame.pop_to_do()
	while to_do != "FINISH":
		if to_do in to_dos:
			## IS COURT
			if to_do == ISCRT.TD_IS_COURT:
				ISCRT.is_court(frame, show_progress=SHOW_PROGRESS)
			if to_do == ISCRT.TD_CORRECT_WITH_LAST:
				ISCRT.correct_with_last(frame, frame_buffer.get([-1]), show_progress=SHOW_PROGRESS)
			if to_do == ISCRT.TD_CORRECT_APROXIMATING:
				ISCRT.correct_approximating(frame)
			
			## COURT POINTS
			if to_do == CRT.TD_COURT_POINTS:
				CRT.extract_court_points(frame, show_progress=SHOW_PROGRESS)
			if to_do == CRT.TD_CHECK_BAD_POINTS:
				CRT.check_bad_points(frame, frame_buffer.get(-1))
			if to_do == CRT.TD_COMPARE_COURT_LINES:
				CRT.check_missing_lines(frame, frame_buffer.get(-1), show_progress=SHOW_PROGRESS)
			if to_do == CRT.TD_GET_HOMOGRAPHY:
				CRT.create_homographys(frame, show_progress=SHOW_PROGRESS)
			if to_do == CRT.TD_CREATES_MISSING_POINTS:
				CRT.create_missing_points(frame, show_progress=SHOW_PROGRESS)
			
			## DETECT PLAYERS
			if to_do == PLYRS.TD_DETECT_PLAYERS:
				PLYRS.detect_players(frame, frame_buffer.get(-1))
			
			## DETECT BALL
			if to_do == BLL.TD_DETECT_BALL:
				BLL.detect_ball(frame, frame_buffer.get(-1), frame_buffer.get(BLL.LAST_FRAMES_TO_MERGE))
			if to_do == STRK.TD_DETECT_STROKE:
				STRK.detect_stroke(frame)
		
		to_do = frame.pop_to_do()
	
	frame.info["process_time"] = time.time() - t_ini

	return frame


def process_video(video_path, finished_video=None, all_models={}, to_dos=ALL_TO_DOS, show=True, verbose=True):
	captured_video = cv2.VideoCapture(video_path)

	paused = False
	time_all = 0
	frames_info = {}
	frame_buffer = FTGM.Buffer()
	frame_position = 0

	## START
	if verbose: print("Procesando video...")
	time_all = time.time()
	while captured_video.isOpened():
		if paused is False:
			# Lee el frame
			ret, readed_frame = captured_video.read()
			if ret is False:
				captured_video.release()

			if readed_frame is not None:
				
				## LECTURA
				frame = FTGM.Fotograma(frame_position, readed_frame)
				frame_position += 1

				## PROCESADO
				process_frame(frame, frame_buffer, all_models=all_models, to_dos=to_dos)

				## ESCRITURA
				if finished_video is not None:
					finished_video.write(frame.get_image_to_write())
				if show is True:
					cv2.imshow('frame', frame.get_image_to_write())
				
				## INFO
				frames_info[frame.position] = frame.get_info_to_store()

				frame_buffer.push(frame)
				if verbose: print("\t" + str(frame_position), end="\r")
		
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			captured_video.release()
		elif key & 0xFF == ord('j'):
			if verbose: print("\t\tjumping 500...")
			for _ in range(500):
				ret, frame = captured_video.read()
		elif key & 0xFF == ord(' '):
			if paused is True:
				paused = False
			else:
				paused = True
		elif key & 0xFF == ord('s'):
			cv2.imwrite(str(frame_position) + ".png", frame.original)
		
		#if frame_position >= MAX_FRAMES_TO_MANAGE:
		#	captured_video.release()

	## FIN
	frames_info[frame_position-1]["END"] = True
	cv2.destroyAllWindows() #TODO

	all_info = {"process_time": time.time() - time_all,
				"n_frames": frame_position,
				"frames_info": frames_info}

	if verbose: print("Procesados:\t" + str(all_info["n_frames"]))
	if verbose: print("Total:\t\t" + str(all_info["process_time"]))

	return all_info


def second_round_process(all_info, all_models={}, to_dos=ALL_TO_DOS, verbose=True):
	court_points_list = []
	detect_players_list = []
	detect_ball_list = []
	detect_ball_list_2 = []

	frames_info = all_info["frames_info"]

	if verbose: print("\nComenzando segunda vuelta...")
	
	rounds_organization = [
		[PLYRS.TD_POST_CHECKS, BLL.TD_POST_CHECKS, CRT.TD_SMOOTH_POINTS],
		[PLYRS.TD_APROXIMATE, BLL.TD_APROXIMATE],
	]

	time_all = time.time()
	for round_to_dos in rounds_organization:
		# Reinicia las listas
		court_points_list.clear()
		detect_players_list.clear()
		detect_ball_list.clear()
		detect_ball_list_2.clear()

		frame_position = 0
		while True:
			# Informacion del frame actual
			frame_info = frames_info.get(frame_position, None)
			if frame_info is None:
				break

			# Recrea el frame
			frame = FTGM.Fotograma.from_info(frame_position, frame_info)
			frame.info["models"] = all_models
			
			if frame.info.get("IS_COURT", False) is False:
				# Reinicia las listas
				detect_players_list.clear()
				detect_ball_list.clear()

			## COURT POINTS
			if CRT.TD_SMOOTH_POINTS in to_dos and CRT.TD_SMOOTH_POINTS in round_to_dos:
				CRT.smooth_court_points(frames_info, frame, court_points_list)

			## DETECT PLAYERS
			if PLYRS.TD_POST_CHECKS in to_dos and PLYRS.TD_POST_CHECKS in round_to_dos:
				PLYRS.post_checks(frames_info, frame, detect_players_list)
			if PLYRS.TD_APROXIMATE in to_dos and PLYRS.TD_APROXIMATE in round_to_dos:
				PLYRS.approximate_players(frames_info, frame, detect_players_list)

			## BALL
			if BLL.TD_POST_CHECKS in to_dos and BLL.TD_POST_CHECKS in round_to_dos:
				BLL.post_checks(frames_info, frame, detect_ball_list, detect_ball_list_2)
			if BLL.TD_APROXIMATE in to_dos and BLL.TD_APROXIMATE in round_to_dos:
				BLL.approximate_balls(frames_info, frame, detect_ball_list)

			# Comprueba el frame de finalizacion
			if frame.info.get("END", False) is True:
				break

			#if verbose: print("\t" + str(frame_position), end="\r")
			frame_position += 1

	## FIN
	all_info["process_time"] += time.time() - time_all
	if verbose: print("Procesados:\t" + str(frame_position))

	return all_info


def upper_level_process(all_info, all_models={}, to_dos=ALL_TO_DOS, verbose=True):
	last_stroke_list = []
	actual_stroke_list = []
	between_stroke_list = []
	next_stroke_list = []
	track_players_list = []

	frames_info = all_info["frames_info"]

	if verbose: print("\nComenzando procesado de alto nivel...")
	
	rounds_organization = [
		[STRK.TD_DETECT_STROKE],
		[STRK.TD_CORRECT_STROKE],
		[STRK.TD_NEXT_STROKE],
		[PLYRS.TD_TRACK_PLAYERS],
	]

	time_all = time.time()
	for round_to_dos in rounds_organization:
		# Reinicia las listas
		track_players_list.clear()
		last_stroke_list.clear()
		actual_stroke_list.clear()
		between_stroke_list.clear()
		next_stroke_list.clear()

		frame_position = 0
		while True:
			try:
				# Informacion del frame actual
				frame_info = frames_info.get(frame_position, None)
				if frame_info is None:
					break

				# Recrea el frame
				frame = FTGM.Fotograma.from_info(frame_position, frame_info)
				frame.info["models"] = all_models

				## STROKE
				if STRK.TD_DETECT_STROKE in to_dos and STRK.TD_DETECT_STROKE in round_to_dos:
					STRK.detect_stroke(frame)
					frames_info[frame.position] = frame.get_info_to_store()
				if STRK.TD_CORRECT_STROKE in to_dos and STRK.TD_CORRECT_STROKE in round_to_dos:
					STRK.correct_strokes_secuences(frame, actual_stroke_list, between_stroke_list, next_stroke_list, frames_info)

				## TRACK PLAYERS
				if PLYRS.TD_TRACK_PLAYERS in to_dos and PLYRS.TD_TRACK_PLAYERS in round_to_dos:
					CRT.create_homographys(frame)
					PLYRS.track_players(frames_info, frame, track_players_list)

				## NEXT_STROKE
				if STRK.TD_NEXT_STROKE in to_dos and STRK.TD_NEXT_STROKE in round_to_dos:
					CRT.create_homographys(frame)
					STRK.calculate_stroke_direction(frame, last_stroke_list, actual_stroke_list, between_stroke_list, next_stroke_list, frames_info)

				# Comprueba el frame de finalizacion
				if frame.info.get("END", False) is True:
					break

				#if verbose: print("\t" + str(frame_position), end="\r")
				frame_position += 1
			except Exception as e:
				print("========= " + str(frame_position))
				raise e

	## FIN
	all_info["process_time"] += time.time() - time_all
	if verbose: print("Procesados:\t" + str(frame_position))

	return all_info


def write_on_video(video_path, frames_info, new_name=None, show=True, verbose=True):
	# Crea el video para la escritura
	finished_video = create_video_to_write(video_path, new_name)

	# Abre el video de lectura
	captured_video = cv2.VideoCapture(video_path)

	if verbose: print("\nEscribiendo en video...")

	frame_buffer = FTGM.Buffer()
	frame_position = 0
	while captured_video.isOpened():
		# Lee el frame
		ret, readed_frame = captured_video.read()
		if ret is False:
			captured_video.release()

		if readed_frame is not None:
			# Crea el objeto
			frame = FTGM.Fotograma(frame_position, readed_frame)
			if frame_position in frames_info:
				# Craga la informacion
				frame.set_info(frames_info[frame_position])

				# Realiza un procesamiento minimo
				CRT.create_homographys(frame)

				# Dibuja el progreso
				if SHOW_PROGRESS is True:
					PNT.paint_progress(frame)

				# Dibuja la informacion resultante
				#PNT.paint_distance_map(frame)
				
				#PNT.show_players_tracks(frame, frame_buffer.get(-1))
				PNT.show_players_tracks_on_point(frame)
				
				PNT.show_ball_track(frame)

				PNT.show_stroke_predictions(frame)

				#PNT.show_statistics(frame)

				#PNT.get_images_to_mem(frame) #TODO

			# Escribe el frame en el video
			for _ in range(frame.info.get("slow_frame", 1)):
				finished_video.write(frame.get_image_to_write())
			if show is True:
				cv2.imshow('frame', frame.get_image_to_write())

			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				captured_video.release()
			
			# Aniade el frame al buffer
			frame_buffer.push(frame)

			# Detecta el ultimo frame con info
			if frame.info.get("END", False) is True:
				captured_video.release()
		
		if verbose: print("\t" + str(frame_position), end="\r")
		frame_position += 1

	if verbose: print(" "*20, end="\r")
	if verbose: print("Leidos:\t\t" + str(frame_position - 1))
	if verbose: print("Escritos:\t" + str(len(frames_info)))

	# Cierra el video de escritura
	finished_video.release()
	cv2.destroyAllWindows() #TODO


def info_path_from_video(video_path):
	return video_path.replace(".mp4", FRAMES_INFO_JSON_SUFIX + ".json")


def truth_path_from_video(video_path):
	return video_path.replace(".mp4", FRAMES_TRUTH_JSON_SUFIX + ".json")


def load_info(file_path):
	with open(file_path, "r") as f:
		all_info = json.load(f)
	
	if "frames_info" in all_info:
		for position in list(all_info["frames_info"].keys()):
			if isinstance(position, str):
				value = all_info["frames_info"].pop(position)
				try:
					all_info["frames_info"][int(position)] = value
				except ValueError:
					all_info["frames_info"][position] = value
	
	return all_info


def store_all_info(all_info, video_path):
	for pos in all_info["frames_info"].keys():
		for info_key in all_info["frames_info"][pos]:
			all_info["frames_info"][pos][info_key] = FTGM.to_serializable(all_info["frames_info"][pos][info_key])

	with open(str(info_path_from_video(video_path)).replace(".json", "__prueba.json"), "w") as f:
		json.dump(all_info, f, indent=4)


def main(video_path, file_path=None, all_models=None, write_video=True, store_info=True, to_dos=ALL_TO_DOS, post_process=True, upper_process=True):
	if file_path is not None:
		# Lee la informacion de un archivo json
		all_info = load_info(file_path)

		if "frames_info" not in all_info:
			print("\tERR:\tarchivo '" + str(file_path) + "' con contenido erroneo")
			sys.exit(0)

		print("Archivo con " + str(len(all_info["frames_info"])) + " frames leido")

		"""
		# Segundo procesamiento a la informacion de los frames
		second_round_process(all_info,
								all_models={},
								to_dos=to_dos,
								verbose=True)

		if upper_process is True:
			# Procesamiento de alto nivel de la informacion de los frames
			upper_level_process(all_info,
								all_models=all_models,
								to_dos=to_dos,
								verbose=True)
		"""
	else:
		# Prepara los modelos
		if all_models is None:
			all_models = prepare_models()
		#all_models = prepare_models()

		# Procesa el video indicado
		all_info = process_video(video_path,
								 finished_video=None,
								 #finished_video=create_video_to_write(video_path),
								 all_models=all_models,
								 to_dos=to_dos,
								 show=SHOW_PROGRESS,
								 verbose=True)
		
		if post_process is True:
			# Segundo procesamiento a la informacion de los frames
			second_round_process(all_info,
								 all_models=all_models,
								 to_dos=to_dos,
								 verbose=True)
		
		if upper_process is True:
			# Procesamiento de alto nivel de la informacion de los frames
			upper_level_process(all_info,
								all_models=all_models,
								to_dos=to_dos,
								verbose=True)
	
	# Escribe la informacion del procesamiento en el video
	if write_video is True:
		write_on_video(video_path,
					   all_info["frames_info"],
					   show=SHOW_PROGRESS,
					   verbose=True)
	
	if file_path is None and store_info is True:
		# Guarda la informacion del procesamiento en un json
		store_all_info(all_info, video_path)
	
	return all_info


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("video_path", type=str)
	parser.add_argument("--from_file", action="store_true")
	parser.add_argument("--from_truth", action="store_true")
	args = parser.parse_args()

	if not os.path.isfile(args.video_path):
		print("\tERR:\tvideo '" + str(args.video_path) + "' no encontrado")
		sys.exit(0)
	elif not args.video_path.endswith(".mp4"):
		print("\tERR:\t'" + str(args.video_path) + "' tiene que ser un archivo mp4") #TODO: ¿seguro?
		sys.exit(0)
	
	if args.from_file is True:
		info_file = info_path_from_video(args.video_path)
		if os.path.isfile(info_file) is False:
			print("\tERR:\tarchivo de info '" + str(info_file) + "' no encontrado")
			sys.exit(0)
	elif args.from_truth is True:
		info_file = truth_path_from_video(args.video_path)
		if os.path.isfile(info_file) is False:
			print("\tERR:\tarchivo de truth '" + str(info_file) + "' no encontrado")
			sys.exit(0)
	else:
		info_file = None
	
	main(args.video_path, info_file)
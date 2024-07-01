import os
import cv2
import numpy as np

import methods.validation_methods as VAL


TAM_BUFFER = 10

COURT_VAL_UMBRAL = 0.85
PLAYER_VAL_UMBRAL = 0.5
PLAYER_MAX_DIST = 70
BALL_MAX_DIST = 70
STROKE_UMBRAL = 0


INFO_TO_KEEP_LIST = [
	"IS_COURT",
	"COURT_POINTS", "COURT_POINTS_DOUBLES",
	"top_player", "top_player_box", "top_player_track", "top_player_point_track", "top_player_stroke", "top_player_distance",
	"bottom_player", "bottom_player_box", "bottom_player_track", "bottom_player_point_track", "bottom_player_stroke", "bottom_player_distance",
	"fix_players",
	"ball", "fix_ball", "ball_destination", "ball_destination_frame", "ball_track", "frame_of_impact",
	"stroke_type",
	"slow_frame",
	"END", "process_time", "validation",
]


def to_serializable(src):
	if isinstance(src, np.ndarray) or isinstance(src, list):
		new_list = []
		for i in src:
			new_list.append(to_serializable(i))
		return new_list
	if isinstance(src, np.int32) or isinstance(src, np.int64):
		return int(src)
	elif isinstance(src, np.float32):
		return float(src)
	else:
		return src


class Fotograma():
	def __init__(self, position: int, original: np.ndarray):
		self.position = position
		self.original = original
		self.result = None
		self.to_dos = []
		self.info = {}
	
	@staticmethod
	def from_info(position: int, frame_info: dict):
		new_frame = Fotograma(position, None)
		new_frame.set_info(frame_info)
		return new_frame

	def push_to_do(self, to_do: str):
		self.to_dos.append(to_do)
	
	def pop_to_do(self):
		if len(self.to_dos) < 1:
			return "FINISH"
		else:
			return self.to_dos.pop()

	def get_image_to_write(self):
		if self.result is not None:
			return self.result
		else:
			return self.original

	def get_info_to_store(self, serialize=False):
		info = {}
		for info_to_copy in INFO_TO_KEEP_LIST:
			if info_to_copy in self.info:
				if serialize is True:
					info[info_to_copy] = to_serializable(self.info[info_to_copy])
				else:
					info[info_to_copy] = self.info[info_to_copy]

		return info

	def set_info(self, info):
		for info_to_copy in INFO_TO_KEEP_LIST:
			if info_to_copy in info:
				self.info[info_to_copy] = info[info_to_copy]

	def validate(self, true_info):
		validation_info = {}
		previous_lost = False

		## IS_COURT ##
		if "IS_COURT" in true_info:
			validation_info["IS_COURT"] = {}
			if "IS_COURT" in self.info:
				validation_info["IS_COURT"]["confusion"] = VAL.ConfusionMatrix.validate(self.info["IS_COURT"], true_info["IS_COURT"])
			else:
				validation_info["IS_COURT"]["loss"] = True
				previous_lost = True

		## COURT_POINTS ##
		if previous_lost is False:
			for court_type in ["COURT_POINTS", "COURT_POINTS_DOUBLES"]:
				if court_type in true_info:
					validation_info[court_type] = {}
					if court_type in self.info:
						iou = VAL.IoU.validate_from_polygon(self.info[court_type], true_info[court_type])
						validation_info[court_type]["confusion"] = VAL.ConfusionMatrix.validate(iou > COURT_VAL_UMBRAL, True)
						validation_info[court_type]["iou"] = iou
					else:
						validation_info[court_type]["loss"] = True
						previous_lost = True

		## PLAYERS_DETECTION ##
		if previous_lost is False:
			for player_type in ["top_player", "bottom_player"]:
				if player_type in true_info:
					validation_info[player_type] = {}
					if player_type in self.info:
						dist = VAL.EuclideanDistance.validate(self.info[player_type], true_info[player_type])
						validation_info[player_type]["confusion"] = VAL.ConfusionMatrix.validate(dist < PLAYER_MAX_DIST, True)
						validation_info[player_type]["dist"] = dist
					else:
						validation_info[player_type]["loss"] = True

		## PLAYERS_DETECTION boxes ##
		if previous_lost is False:
			for player_box_type in ["top_player_box", "bottom_player_box"]:
				if player_box_type in true_info:
					validation_info[player_box_type] = {}
					if player_box_type in self.info:
						iou = VAL.IoU.validate_from_boxes(self.info[player_box_type], true_info[player_box_type])
						validation_info[player_box_type]["confusion"] = VAL.ConfusionMatrix.validate(iou > PLAYER_VAL_UMBRAL, True)
						validation_info[player_box_type]["iou"] = iou
					else:
						validation_info[player_box_type]["loss"] = True

		## BALL ##
		if previous_lost is False:
			if "ball" in true_info:
				validation_info["ball"] = {}
				if "ball" in self.info:
					#ecm = VAL.ECM.validate(self.info["ball"], true_info["ball"])
					dist = VAL.EuclideanDistance.validate(self.info["ball"], true_info["ball"])
					validation_info["ball"]["confusion"] = VAL.ConfusionMatrix.validate(dist < BALL_MAX_DIST, True)
					validation_info["ball"]["dist"] = dist
				else:
					validation_info["ball"]["loss"] = True

		## PROCESS TIME ##
		validation_info["process_time"] = self.info["process_time"]

		self.info["validation"] = validation_info
		return validation_info


def copy_frame_to_mem(frame: Fotograma):
	FRM_COPY = Fotograma(-1, frame.original.copy())
	FRM_COPY.set_info(frame.get_info_to_store())
	FRM_COPY.info["to_topview"] = frame.info.get("to_topview", None)
	FRM_COPY.info["to_cameraview"] = frame.info.get("to_cameraview", None)
	return FRM_COPY


class Buffer():
	def __init__(self, tam: int = TAM_BUFFER):
		self._buffer = []
		self.actual = -1
		self.max = tam

	def push(self, frame):
		if len(self._buffer) >= self.max:
			self._buffer.pop(0)
		self._buffer.append(frame)
	
	def get(self, pos):
		if isinstance(pos, list):
			to_return = []
			for i in pos:
				value = self.get(i)
				if value is None: # ¿?
					return []
				else:
					to_return.append(value)
			return to_return
		elif isinstance(pos, int):
			if len(self._buffer) > 0 and pos < len(self._buffer) and pos >= -len(self._buffer):
				try:
					return self._buffer[pos]
				except IndexError as e:
					print(pos)
					print(self._buffer)
					raise e
	
	def len(self):
		return len(self._buffer)

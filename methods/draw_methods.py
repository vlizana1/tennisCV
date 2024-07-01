import cv2
import sys
import numpy as np

import fotograma as FTGM
#import math_methods as MTH


# BGR
BLUE = (255,0,0)
GREEN = (0,255,0)
RED = (0,0,255)
YELLOW = (15,196,241)

FADE_COLORS = {
	"r": {"color": [1/2, 1/2, 3/2], "not": [2,   2,   2/3]},
	"y": {"color": [1/2, 3/2, 3/2], "not": [2,   2/3, 2/3]},
	"g": {"color": [1/2, 3/2, 1/2], "not": [2,   2/3, 2]},
	"b": {"color": [3/2, 1/2, 1/2], "not": [2/3, 2,   2]}
}


def draw_points(fotograma: FTGM.Fotograma, pts, color, fill=True):
	if fotograma.result is None:
		fotograma.result = np.copy(fotograma.original)
	
	thickness = -1 if fill is True else 1
	for p in pts:
		if p is not None and len(p) == 2:
			cv2.circle(fotograma.result, (p[0], p[1]), 10, color, thickness)


def draw_lines(fotograma: FTGM.Fotograma, lines, color):
	if fotograma.result is None:
		fotograma.result = np.copy(fotograma.original)
	for line in lines:
		if line is not None and len(line) == 4:
			cv2.line(fotograma.result, (line[0], line[1]), (line[2], line[3]), color, 2)


def draw_squares(fotograma: FTGM.Fotograma, squares, color):
	if fotograma.result is None:
		fotograma.result = np.copy(fotograma.original)
	for square in squares:
		if square is not None and len(square) == 4:
			cv2.line(fotograma.result, (square[0], square[1]), (square[2], square[1]), color, 1)
			cv2.line(fotograma.result, (square[2], square[1]), (square[2], square[3]), color, 1)
			cv2.line(fotograma.result, (square[2], square[3]), (square[0], square[3]), color, 1)
			cv2.line(fotograma.result, (square[0], square[3]), (square[0], square[1]), color, 1)


def draw_track(fotograma: FTGM.Fotograma, track, color):
	track_lines = []
	for i in range(len(track) - 1):
		if track[i] is not None and track[i+1] is not None:
			track_lines.append([track[i][0], track[i][1], track[i+1][0], track[i+1][1]])
	draw_lines(fotograma, track_lines, color)


"""
def draw_circunference(fotograma: FTGM.Fotograma, pts, color):
	draw_lines(fotograma, MTH.get_area_lines(pts), color)
"""


def draw_area(fotograma: FTGM.Fotograma, pts, color):
	if fotograma.result is None:
		fotograma.result = np.copy(fotograma.original)

	reshaped_pts = np.array(pts, np.int32)
	reshaped_pts = reshaped_pts.reshape((-1, 1, 2))

	paint_mask = np.zeros(fotograma.result.shape[:2], dtype=np.uint8)
	cv2.fillPoly(paint_mask, [reshaped_pts], 255)

	mascara = np.zeros(fotograma.result.shape[:2], dtype=np.uint8)
	cv2.fillPoly(mascara, [reshaped_pts], 255)

	for i_RGB, variation in zip([0, 1, 2], color):
		fotograma.result[:, :, i_RGB] = np.where(mascara == 255, fotograma.result[:, :, i_RGB] * variation, fotograma.result[:, :, i_RGB])


def draw_box(fotograma: FTGM.Fotograma, box, color):
	area_from_box = [
		[box[0], box[1]],
		[box[2], box[1]],
		[box[2], box[3]],
		[box[0], box[3]],
	]

	draw_area(fotograma, area_from_box, color)


def write_text(fotograma: FTGM.Fotograma, text: str, position, background=False):
	if "\n" in text:
		return write_lines(fotograma, text.split("\n"), position, background)
	else:
		if fotograma.result is None:
			fotograma.result = np.copy(fotograma.original)

		(text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

		if background is True:
			margin = 5
			cv2.rectangle(fotograma.result,
						(position[0] - margin, position[1] - text_height - margin),
						(position[0] + text_width + margin, position[1] + margin),
						(0, 0, 0),
						cv2.FILLED)

		cv2.putText(fotograma.result, text, position,
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

		return text_width, text_height


def write_lines(fotograma: FTGM.Fotograma, lines: list, position, background=False):
	separation = 10

	aux_position = position
	for line in lines:
		_, text_height = write_text(fotograma, line, aux_position, background=background)
		aux_position[1] += text_height + separation
	
	return aux_position















































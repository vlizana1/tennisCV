import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import cv2

import fotograma as FTGM
import court_points as CRT
import methods.draw_methods as DRW

from keras.models import load_model


IMAGE_HEIGHT = 360

SENSITIVITY = 0.3

TD_IS_COURT = "IS_COURT"
TD_CORRECT_WITH_LAST = "CORRECT_WITH_LAST"
TD_CORRECT_APROXIMATING = "CORRECT_APROXIMATING"


def get_model():
	return load_model("models/isCourt.keras")


def __img_preprocess(img):
	img_prop = img.shape[0] / img.shape[1]
	img = cv2.resize(img, (int(IMAGE_HEIGHT / img_prop), IMAGE_HEIGHT))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype('float32')
	img /= 255
	return img


def is_court(fotograma: FTGM.Fotograma, sensitivity=SENSITIVITY, show_progress=False):
	img = __img_preprocess(fotograma.original)
	img = img.reshape(1, img.shape[0], img.shape[1], 1)
	ans = fotograma.info["models"]["is_court"].predict(img, batch_size=1, verbose=0)
	
	if  sensitivity < ans < 1 - sensitivity:
		fotograma.push_to_do(TD_CORRECT_WITH_LAST)
		fotograma.info["IS_COURT_ANS"] = ans
		if show_progress: DRW.draw_points(fotograma, [[20, 20]], DRW.BLUE)
	else:
		if ans < 0.5:
			fotograma.info["IS_COURT"] = True
			fotograma.push_to_do(CRT.TD_COURT_POINTS)
			if show_progress: DRW.draw_points(fotograma, [[20, 20]], DRW.GREEN)
		else:
			fotograma.info["IS_COURT"] = False
			if show_progress: DRW.draw_points(fotograma, [[20, 20]], DRW.RED)


def correct_with_last(fotograma: FTGM.Fotograma, last_frames, sensitivity=SENSITIVITY, show_progress=False):
	if len(last_frames) > 0 and "IS_COURT" in last_frames[0].info:
		if last_frames[0].info["IS_COURT"] is True \
		and fotograma.info["IS_COURT_ANS"] < 1 - sensitivity: #TODO: por que el and
			fotograma.info["IS_COURT"] = True
			if show_progress: DRW.draw_points(fotograma, [[30, 20]], DRW.GREEN)
		elif last_frames[0].info["IS_COURT"] is False \
		and fotograma.info["IS_COURT_ANS"] > sensitivity: #TODO: por que el and
			fotograma.info["IS_COURT"] = False
			if show_progress: DRW.draw_points(fotograma, [[30, 20]], DRW.RED)
	
	if "IS_COURT" not in fotograma.info:
		#fotograma.push_to_do(TD_CORRECT_APROXIMATING)
		fotograma.info["IS_COURT"] = False
	elif fotograma.info["IS_COURT"] is True:
		fotograma.push_to_do(CRT.TD_COURT_POINTS)


def correct_approximating(fotograma: FTGM.Fotograma):
	if fotograma.info.get("IS_COURT_ANS", 1) < 0.5:
		fotograma.info["IS_COURT"] = True
	else:
		fotograma.info["IS_COURT"] = False
	
	fotograma.info["IS_COURT_ANS"] = None
	
	if fotograma.info.get("IS_COURT", False) is True:
		fotograma.push_to_do(CRT.TD_COURT_POINTS)
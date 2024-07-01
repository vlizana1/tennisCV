import seaborn as sns
import matplotlib.pyplot as plt
import methods.math_methods as MTH
from sklearn.metrics import confusion_matrix


class ConfusionMatrix():
	True_Positive = 0
	False_Positive = 1
	False_Negative = 2
	True_Negative = 3
	
	@staticmethod
	def init():
		return [0]*4

	@staticmethod
	def acumulate(acumulation, new):
		if new < 4:
			acumulation[new] += 1
		return acumulation

	@staticmethod
	def validate(info, known_info):
		if info is True:
			if known_info is True:
				return ConfusionMatrix.True_Positive
			else:
				return ConfusionMatrix.False_Positive
		else:
			if known_info is True:
				return ConfusionMatrix.False_Negative
			else:
				return ConfusionMatrix.True_Negative

	@staticmethod
	def get_a_p_r(matrix):
		try:
			a = (matrix[ConfusionMatrix.True_Positive] + matrix[ConfusionMatrix.True_Negative]) / \
				(matrix[ConfusionMatrix.True_Positive] + matrix[ConfusionMatrix.True_Negative] + \
				matrix[ConfusionMatrix.False_Positive] + matrix[ConfusionMatrix.False_Negative])
		except ZeroDivisionError:
			a = 0
		
		try:
			p = matrix[ConfusionMatrix.True_Positive] / \
				(matrix[ConfusionMatrix.True_Positive] + matrix[ConfusionMatrix.False_Positive])
		except ZeroDivisionError:
			p = 0
		
		try:
			r = matrix[ConfusionMatrix.True_Positive] / \
				(matrix[ConfusionMatrix.True_Positive] + matrix[ConfusionMatrix.False_Negative])
		except ZeroDivisionError:
			r = 0
		
		return a, p, r

	@staticmethod
	def get_micro_a_p_r(matrix_list):
		general_a = sum([m[ConfusionMatrix.True_Positive] for m in matrix_list]) / \
					sum([sum(m) for m in matrix_list])

		micro_p = sum([m[ConfusionMatrix.True_Positive] for m in matrix_list]) / \
					sum([m[ConfusionMatrix.True_Positive] + m[ConfusionMatrix.False_Positive] for m in matrix_list])

		micro_r = sum([m[ConfusionMatrix.True_Positive] for m in matrix_list]) / \
					sum([m[ConfusionMatrix.True_Positive] + m[ConfusionMatrix.False_Negative] for m in matrix_list])

		return general_a, micro_p, micro_r


	@staticmethod
	def plot_matrix(matrix, title="Matriz de confusión", img_name="matriz_de_confusion"):
		plt.figure(figsize=(8, 6))
		sns.set_theme(font_scale=2)
		sns.heatmap([matrix[:2], matrix[2:]], annot=True, cmap="Blues", fmt="g", cbar=False, xticklabels=False, yticklabels=["True", "False"])
		plt.title(title)
		plt.savefig(img_name)



class IoU():
	@staticmethod
	def validate_from_polygon(pol_1, pol_2):
		for p in pol_1 + pol_2:
			if p is None \
			or len(p) != 2 \
			or p[0] is None \
			or p[1] is None:
				return -1

		# Area de la interseccion
		pol_inter = MTH.intersec_polygons(pol_1, pol_2)
		if len(pol_inter) >= 4:
			intersection = MTH.get_area_of_polygon(pol_inter)
		else:
			intersection = 0

		# Areas de los poligonos
		area1 = MTH.get_area_of_polygon(pol_1)
		area2 = MTH.get_area_of_polygon(pol_2)

		# Interseccion sobre Union
		return intersection / (area1 + area2 - intersection)
	

	@staticmethod
	def validate_from_boxes(box1, box2):
		# Box de interseccion
		x_1 = max(box1[0], box2[0])
		x_2 = min(box1[2], box2[2])
		y_1 = min(box1[1], box2[1])
		y_2 = max(box1[3], box2[3])

		# Area de la interseccion
		intersection = max(0, x_2 - x_1) * max(0, y_1 - y_2)

		# Areas de las boxes
		area1 = (box1[2] - box1[0]) * (box1[1] - box1[3])
		area2 = (box2[2] - box2[0]) * (box2[1] - box2[3])

		# Interseccion sobre Union
		return intersection / (area1 + area2 - intersection)

	@staticmethod
	def validate_from_ranges(range1, range2, over_truth=False):
		if range1[1] >= range2[1]:
			intersection = max(0, range2[1] - range1[0]) - max(0, range2[0] - range1[0])
		else:
			intersection = max(0, range1[1] - range2[0]) - max(0, range1[0] - range2[0])

		if over_truth is False:
			return intersection / ((range1[1] - range1[0]) + (range2[1] - range2[0]) - intersection)
		else:
			return intersection / (range1[1] - range1[0])

class EuclideanDistance():
	@staticmethod
	def validate(p1, p2):
		return pow(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2), 1/2)

class ECM():
	@staticmethod
	def validate(p1, p2):
		return (pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)) / 2
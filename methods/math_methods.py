import cv2
import math
import numpy as np
from shapely.geometry import Polygon


def get_line_slope(line):
	if line[0] == line[2]:
		return (line[1] - line[3]) / 1e-6
	else:
		return (line[1] - line[3]) / ((line[0] - line[2]))


def get_point_distance(p1, p2):
	return math.sqrt(pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1], 2))


def get_line_size(line):
	return get_point_distance([line[0], line[1]], [line[2], line[3]])


def get_line_distance(l1, l2):
	if l1[0] < l1[2]:
		ini1 = l1[:2]
		fin1 = l1[2:]
	else:
		fin1 = l1[:2]
		ini1 = l1[2:]
	if l2[0] < l2[2]:
		ini2 = l2[:2]
		fin2 = l2[2:]
	else:
		fin2 = l2[:2]
		ini2 = l2[2:]
	
	dist_ini = get_point_distance(ini1, ini2)
	dist_fin = get_point_distance(fin1, fin2)
	
	return (dist_ini + dist_fin) / 2


def get_cut_point(line1, line2):
	x1, y1, x2, y2 = line1
	x3, y3, x4, y4 = line2

	m1 = get_line_slope(line1)
	m2 = get_line_slope(line2)
	if m1 == m2:
		return None

	x_interseccion = (m1 * x1 - y1 - m2 * x3 + y3) / (m1 - m2)
	y_interseccion = round(m1 * (x_interseccion - x1) + y1)
	x_interseccion = round(x_interseccion)

	if (x1 <= x_interseccion <= x2 or x2 <= x_interseccion <= x1) and \
		(x3 <= x_interseccion <= x4 or x4 <= x_interseccion <= x3) and \
		(y1 <= y_interseccion <= y2 or y2 <= y_interseccion <= y1) and \
		(y3 <= y_interseccion <= y4 or y4 <= y_interseccion <= y3):
		return [x_interseccion, y_interseccion]
	else:
		return None


def resize_points(points, prop):
	resized = []
	for p in points:
		resized.append((int(p[0] * prop[0]),
						int(p[1] * prop[1])))
	return resized


def resize_lines(lines, prop):
	resized = []
	for l in lines:
		resized.append([int(l[0] * prop[0]), int(l[1] * prop[1]),
						int(l[2] * prop[0]), int(l[3] * prop[1])])
	return resized


def reshape_square(square):
	reshaped = [aux for aux in square]
	if reshaped[0] > reshaped[2]:
		reshaped = [square[2], square[1], square[0], square[3]]
	if reshaped[1] < reshaped[3]:
		reshaped = [square[0], square[3], square[2], square[1]]
	return reshaped


def get_square_size(square):
    return abs((square[2] - square[0]) * (square[3] - square[1]))


def get_center_of_square(square):
	x = int(square[0] + (square[2] - square[0]) / 2)
	y = int(square[1] + (square[3] - square[1]) / 2)
	return [x, y]


def get_foot_from_square(square):
	x = int(square[0] + (square[2] - square[0]) / 2)
	y = square[1]
	return [x, y]


def merge_lists(list_1, list_2, filter_nones=False):
	new_list = [p for p in list_1 if filter_nones is False or p is not None]
	new_list.extend([p for p in list_2 if filter_nones is False or p is not None])
	return new_list


def create_homography(points_src, points_dst):
	if isinstance(points_src, list):
		points_src = np.array(points_src, dtype=np.float32)
	if isinstance(points_dst, list):
		points_dst = np.array(points_dst, dtype=np.float32)
	h, _ = cv2.findHomography(points_src, points_dst)
	return h


def find_homography(homography, points, to_int=True):
	aux = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
	np_type = np.int32 if to_int is True else np.float32
	return np.array(cv2.perspectiveTransform(aux, homography)[:, 0], dtype=np_type) #TODO a lista


def get_circle_points(coord, radius, n_points):
	x, y = coord
	angulo = 2 * math.pi / n_points

	puntos = []
	for i in range(n_points):
		nuevo_x = x + radius * math.cos(i * angulo)
		nuevo_y = y + radius * math.sin(i * angulo)
		puntos.append([nuevo_x, nuevo_y])

	return puntos


def get_elipse_points(coord, a, b, n_points):
	x, y = coord
	angulo = 2 * math.pi / n_points

	puntos = []
	for i in range(n_points):
		nuevo_x = x + a * math.cos(i * angulo)
		nuevo_y = y + b * math.sin(i * angulo)
		puntos.append([nuevo_x, nuevo_y])

	return puntos


def get_area_lines(area):
	lines = []
	for i in range(len(area)):
		next_i = (i + 1) % len(area)
		line = [area[i][0], area[i][1], area[next_i][0], area[next_i][1]]
		lines.append(line)
	return lines


def is_points_inside_box(pt, box):
	if pt[0] > box[0] \
	or pt[0] < box[2] \
	or pt[1] < box[1] \
	or pt[1] > box[3]:
		return True
	else:
		return False


def is_point_inside(pt, area):
	aux_line = [0, 0, pt[0], pt[1]]

	area_lines = get_area_lines(area)
	n_cuts = 0

	for line in area_lines:
		c_point = get_cut_point(aux_line, line)
		#if c_point is not None and not (c_point[0] == line[0] and c_point[1] == line[1]):
		if c_point is not None:
			n_cuts += 1

	if n_cuts % 2 == 0:
		return False
	else:
		return True


def intersec_polygons(pol_1, pol_2):
	aux_1 = Polygon(pol_1)
	aux_2 = Polygon(pol_2)

	interseccion = aux_1.intersection(aux_2)

	if interseccion.is_empty:
		return []
	else:
		return [[int(x), int(y)] for x, y in interseccion.exterior.coords]


def intersec_polygons__ERRONEA(pol_1, pol_2):
	inter_pol = []

	#print("="*40)
	for i in range(len(pol_1)):
		#print("-"*40)

		line_1 = pol_1[i] + pol_1[(i+1) % len(pol_1)]
		line_2 = pol_2[i] + pol_2[(i+1) % len(pol_2)]

		last_1 = pol_1[(i-1) % len(pol_1)] + pol_1[i]
		last_2 = pol_2[(i-1) % len(pol_2)] + pol_2[i]

		#print("  " + str(line_1))
		#print("  " + str(line_2))

		if is_point_inside(line_1[:2], pol_2) is True:
			#print("    p1 IN: " + str(line_1[:2]))
			inter_pol.append(line_1[:2])
		elif is_point_inside(line_2[:2], pol_1) is True:
			#print("    p2 IN: " + str(line_2[:2]))
			inter_pol.append(line_2[:2])

		cut_p = get_cut_point(last_1, line_2)
		if cut_p is not None:
			#print("    l1 CT: " + str(cut_p))
			inter_pol.append(cut_p)
		else:
			cut_p = get_cut_point(last_2, line_1)
			if cut_p is not None:
				#print("    l2 CT: " + str(cut_p))
				inter_pol.append(cut_p)
		
		cut_p = get_cut_point(line_1, line_2)
		if cut_p is not None:
			#print("    12 CT: " + str(cut_p))
			inter_pol.append(cut_p)
	
	return inter_pol


def get_area_of_polygon(polygon):
	# Area de un poligono con teorema de Shoelace
	area = 0
	for i_p, p in enumerate(polygon):
		p_next = polygon[(i_p + 1) % len(polygon)]
		area += p[0] * p_next[1]
		area -= p_next[0] * p[1]
	return abs(area) / 2


def approximate_points_between(p_ini, p_fin, n_points):
	sep_x = (p_fin[0] - p_ini[0]) / (n_points + 1)
	sep_y = (p_fin[1] - p_ini[1]) / (n_points + 1)

	new_poitns = []
	for i_point in range(n_points):
		new_poitns.append([int(p_ini[0] + (sep_x * (i_point + 1))),
						   int(p_ini[1] + (sep_y* (i_point + 1)))])

	return new_poitns


def approximate_boxes_between(box_ini, box_fin, n_boxes):
	xy_ini_1 = box_ini[:2]
	xy_ini_2 = box_ini[2:]
	xy_fin_1 = box_fin[:2]
	xy_fin_2 = box_fin[2:]

	xy_news_1 = approximate_points_between(xy_ini_1, xy_fin_1, n_boxes)
	xy_news_2 = approximate_points_between(xy_ini_2, xy_fin_2, n_boxes)

	return [xy_new_1 + xy_new_2 for xy_new_1, xy_new_2 in zip(xy_news_1, xy_news_2)]


################ VALIDATION ################

class ConfusionMatrix():
	True_Positive = 0
	False_Positive = 1
	False_Negative = 2
	True_Negative = 3

	def __init__(self):
		self.matrix = [0]*4
		
	def add(self, result, thruth):
		if thruth is True:
			if result is True:
				self.matrix[self.True_Positive] += 1
				return True
			else:
				self.matrix[self.False_Negative] += 1
		else:
			if result is True:
				self.matrix[self.False_Positive] += 1
			else:
				self.matrix[self.True_Negative] += 1
				return True
		return False

	def get_accuracy(self):
		return (self.matrix[self.True_Positive] + self.matrix[self.True_Negative]) / \
			(self.matrix[self.True_Positive] + self.matrix[self.True_Negative] + \
			self.matrix[self.False_Positive] + self.matrix[self.False_Negative])

	def get_precision(self):
		try:
			return (self.matrix[self.True_Positive]) / \
				(self.matrix[self.True_Positive] + self.matrix[self.False_Positive])
		except ZeroDivisionError:
			return 1

	def get_recall(self):
		return (self.matrix[self.True_Positive]) / \
			(self.matrix[self.True_Positive] + self.matrix[self.False_Negative])

	def get_f1(self):
		p = self.get_precision()
		r = self.get_recall()
		return 2 * ((p * r) / (p + r))


class InterOverUnion():
	def __init__(self):
		self.record = []
	

	def add_boxes(self, box1, box2):
		iou = InterOverUnion.get_from_boxes(box1, box2)
		self.record.append(iou)
		return iou
	

	def add_polygon(self, pol1, pol2):
		iou = InterOverUnion.get_from_polygon(pol1, pol2)
		self.record.append(iou)
		return iou


	def get_avarage(self):
		return sum(self.record) / len(self.record)


	@staticmethod
	def get_from_boxes(box1, box2):
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
	def get_from_polygon(pol_1, pol_2):
		for p in pol_1 + pol_2:
			if p is None \
			or len(p) != 2 \
			or p[0] is None \
			or p[1] is None:
				#print("        algun NONE")
				return -1

		# Area de la interseccion
		pol_inter = intersec_polygons(pol_1, pol_2)
		print(pol_inter)
		if len(pol_inter) >= 4:
			intersection = get_area_of_polygon(pol_inter)
		else:
			intersection = 0

		# Areas de los poligonos
		area1 = get_area_of_polygon(pol_1)
		area2 = get_area_of_polygon(pol_2)

		# Interseccion sobre Union
		return intersection / (area1 + area2 - intersection)


class Crono():
	def __init__(self):
		self.record = []


	def add(self, measurement):
		self.record.append(measurement)
	

	def get_avarage(self):
		return sum(self.record) / len(self.record)


class ECM():
	def __init__(self):
		self.record = []


	def add(self, p1, p2):
		aux = pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)
		self.record.append(aux)
		#self.record.append(get_point_distance(p1, p2))
		return math.sqrt(aux)
	

	def get_ecm(self):
		return sum(self.record) / len(self.record)


if __name__ == "__main__":
	#print(InterOverUnion.get_from_boxes([0, 3, 3, 0], [4, 6, 6, 4]))
	#print(InterOverUnion.get_from_boxes([0, 4, 4, 0], [0, 2, 2, 0]))

	#polygon = [[0, 4], [4, 4], [4, 0], [0, 0]]
	#print(get_area_of_polygon(polygon))

	#pol_1 = [[0, 3], [2, 3], [2, 1], [0, 1]]
	#pol_2 = [[1, 2], [3, 2], [3, 0], [1, 0]]
	#pol_3 = [[4, 3], [6, 3], [6, 1], [4, 1]]

	#pol_1 = [[219, 817], [1715, 817], [1424, 292], [509, 293]]
	#pol_2 = [[221, 812], [1714, 811], [1428, 298], [508, 298]]

	#print(is_point_inside([3, 2], pol_1))

	#print(intersec_polygons(pol_1, pol_2))
	#print(intersec_polygons(pol_1, pol_3))

	truth = [[263, 566], [1014, 566], [823, 151], [442, 152]]
	pred = [[265, 571], [1017, 570], [823, 152], [442, 153]]
	umbral = [[170, 573], [940, 570], [823, 152], [442, 153]]

	print(InterOverUnion.get_from_polygon(truth, pred))
	print(InterOverUnion.get_from_polygon(truth, umbral))

	#box_1 = [1067, 323, 1157, 207]
	#box_2 = [1067, 323, 1157, 207]
	#print(InterOverUnion.get_from_boxes(box_1, box_2))

	#print(approximate_points_between([1, 1], [4, 5], 3))
	#print(approximate_points_between([1, 0], [5, 1], 3))

	#box_1 = [753, 855, 875, 616]
	#box_2 = [789, 850, 892, 615]
	#print(approximate_boxes_between(box_1, box_2, 5))
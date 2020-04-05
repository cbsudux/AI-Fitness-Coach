import math
import numpy as np
import cv2

kpt_names = {0: 'nose', 1: 'neck', 2: 'r_sho', 3: 'r_elb', 4: 'r_wri', 5: 'l_sho', 
			6: 'l_elb', 7: 'l_wri', 8: 'r_hip', 9: 'r_knee', 10: 'r_ank', 11: 'l_hip',
			 12: 'l_knee', 13: 'l_ank', 14: 'r_eye', 15: 'l_eye', 16: 'r_ear', 17: 'l_ear'}


# keep v2 default as horizontalk
def calculate_angle(v1, v0, v2 = None):
	"""
	Calculate angle from v1 to v0 to v2.
	returns -1 if zero check fails
	returns an angle if calculation if valid
	if v2 = None, angle is calculated wrt to the horizontal
	"""
	if v2 == None:
		v2 = [0,0]
		v2[0] = v1[0]
		v2[1] = v0[1]

	x1, x2 = v1[0] - v0[0], v2[0] - v0[0]
	y1, y2 = v1[1] - v0[1], v2[1] - v0[1]
	dot_product = x1 * x2 + y1 * y2
	norm_product = math.sqrt(((x1 * x1) + (y1 * y1)) * ((x2 * x2) + (y2 * y2)))
	
	if (norm_product == 0):
		return -1
	angle = (np.arccos(dot_product / norm_product) * 180 / 3.14)

	return int(angle)


def get_coordinates(pose, idx):
	"""
	Convenience method for getting (x, y) coordinates for
	a given body part id.
	returns tuple of (x, y) coordinates
	"""
	if idx in pose.keys():
		return (pose[idx]['x'], pose[idx]['y'])
	else:
		return False


def average_coordinates(pose, idx1, idx2):
	"""
	Given two body part ids, calculate the mid-point.
	Useful for finding the average height of symmetric 
	body parts (i.e. shoulders).
	If only one body part has been located, return those coordinates.
	returns a tuple of (x, y) coordinates
	"""

	if idx1 in pose.keys() and idx2 in pose.keys():
		return ((pose[idx1]['x'] + pose[idx2]['x'])/2, (pose[idx1]['y'] + pose[idx2]['y'])/2) 
	elif idx1 in pose.keys():
		return get_coordinates(pose, idx1)
	elif idx2 in pose.keys():
		return get_coordinates(pose, idx2)
	else: 
		return False


def draw(pose, img, names = False, coords = False, angles = False, lines = False):
	for keypoint, coord in pose.items():
		cv2.circle(img, (coord['x'], coord['y']), 3, (0,255,0), 3)

		if names:
			cv2.putText(img, kpt_names[keypoint], (coord['x'], coord['y']), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

		# if coords:
			# cv2.putText(img, str({%d, %d}.format(coord['x'], coord['y'],  cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

	key_pairs = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
					  [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]

	symmetric_keys = [[2, 5], [8, 11], [9, 12], [10, 13]]
	
	if lines:
		for key_pair in key_pairs: 
			try:
				x1,y1 = pose[key_pair[0]]['x'], pose[key_pair[0]]['y']
				x2,y2 = pose[key_pair[1]]['x'], pose[key_pair[1]]['y']
				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
			except:
				continue

	# angles wrt horizontal, where v2 = None
	if angles:
		if len(angles) == 0: # Empty list --> finding all angles between pairs
			for key_pair in key_pairs:
				try:
					x1,y1 = pose[key_pair[0]]['x'], pose[key_pair[0]]['y']
					x2,y2 = pose[key_pair[1]]['x'], pose[key_pair[1]]['y']
					cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
					angle = calculate_angle([x1,y1], [x2,y2])
					cv2.putText(img, str(int(angle)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
				except: # if key/key pair doesn't exist, then continue with the next pair
					continue

		if len(angles) == 2: # finding all angles between pairs
			for key_pair in key_pairs:
				if key_pair == angles: 
					try:
						x1,y1 = pose[key_pair[0]]['x'], pose[key_pair[0]]['y']
						x2,y2 = pose[key_pair[1]]['x'], pose[key_pair[1]]['y']
						cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
						angle = calculate_angle([x1,y1], [x2,y2])
						cv2.putText(img, str(int(angle)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
					except: # if key/key pair doesn't exist, then continue with the next pair
						continue

		elif len(angles) == 3: # angle between 3 joints
			try:
				x1,y1 = pose[angles[0]]['x'], pose[angles[0]]['y']
				x2,y2 = pose[angles[1]]['x'], pose[angles[1]]['y']
				x3,y3 = pose[angles[2]]['x'], pose[angles[2]]['y']
				cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
				cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)), (0,0,0), 2)
				angle = calculate_angle([x1,y1], [x2,y2], [x3,y3])
				cv2.putText(img, str(int(angle)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
			except: # if key/key pair doesn't exist, then continue with the next pair
				pass

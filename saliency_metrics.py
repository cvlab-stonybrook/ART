import numpy as np
import math

def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255


def cc(s_map,gt):
	epsilon = 1e-9
	s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + epsilon)
	gt_norm = (gt - np.mean(gt))/(np.std(gt) + epsilon)
	a = s_map_norm + epsilon
	b = gt_norm + epsilon
	r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
	return r

def nss(s_map,gt):
	if np.max(gt) == 255:
		gt = discretize_gt(gt)

	epsilon = 1e-9
	xy = np.where(gt==1)
	s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + epsilon)

	if np.sum(gt) == 0:
		return None
	return np.mean(s_map_norm[xy])

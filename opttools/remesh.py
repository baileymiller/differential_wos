import numpy as np
import gpytoolbox
import igl
import torch

def remesh(v, f, refine_factor = 1.0, remesh_iters = 5):
	h = np.mean(gpytoolbox.halfedge_lengths(v.astype(np.float64), f.astype(np.int32))) * refine_factor
	v, f = gpytoolbox.remesh_botsch(v.astype(np.float64),f.astype(np.int32), remesh_iters, h, True)
	return v, f

def clip(v, min_xyz, max_xyz, epsilon = 5e-3):
	return np.clip(v, min_xyz + epsilon, max_xyz - epsilon)

def remove_duplicates(v, f, small_area = 1e-4):
	v, _, _, f = gpytoolbox.remove_duplicate_vertices(v, epsilon=0.0, faces=f)
	f = igl.collapse_small_triangles(v, f, small_area)
	v, f, _, _ = igl.remove_unreferenced(v, f)
	return v, f
import numpy as np

class SquaredResidualSampler:
	def __init__(self, pMin, pMax, res, exploration_factor = 0.25, p_uniform = 0.5):
		self.pMin = pMin
		self.pMax = pMax
		self.extent = np.array(self.pMax - self.pMin)
		self.res = res
		self.n_voxels = self.res * self.res * self.res
		self.voxel_extent = self.extent / self.res
		self.voxel_volume = self.voxel_extent[0] * self.voxel_extent[1] * self.voxel_extent[2]

		# intitialize residual to be 1 everywhere, 
		self.avg_squared_residual = np.zeros((self.n_voxels))
		self.n_samples = np.zeros((self.n_voxels))

		self.exploration_factor = exploration_factor
		self.p_uniform = p_uniform

	def reset(self):
		self.avg_squared_residual = np.zeros((self.n_voxels))
		self.n_samples = np.zeros((self.n_voxels))

	def compute_weight(self):
		# set weight at unexplored voxels to 0.25 the avg weight
		total_avg_squared_residual = np.mean(self.avg_squared_residual)
		default_weight = total_avg_squared_residual if total_avg_squared_residual > 0 else 1

		# compute the squared residual weights
		weight = np.where(self.n_samples > 0, self.avg_squared_residual, default_weight * self.exploration_factor)

		# uniform distribution
		uniform_weight = np.ones(self.n_voxels) / float(self.n_voxels)

		# mixture of importance and uniform distribution
		return (1.0 - self.p_uniform) * weight / np.sum(weight) + self.p_uniform * uniform_weight

	def update_average(self, idx, val):
		self.avg_squared_residual[idx] *= self.n_samples[idx]
		self.avg_squared_residual[idx] += val
		self.n_samples[idx] += 1
		self.avg_squared_residual[idx] /= self.n_samples[idx]

	def generate_samples(self, n_samples):
		'''
		Generate n volume samples.
		'''
		weight = self.compute_weight()
		sample_voxel_idx = np.random.choice(self.n_voxels, n_samples, replace=True, p=weight)
		samples = self.idx_to_position(sample_voxel_idx)
		samples += np.random.random(samples.shape) * self.voxel_extent
		pdf = weight[sample_voxel_idx] * (1.0 / self.voxel_volume)
		return samples, pdf

	def position_to_idx(self, p):
		'''
		Convert position to a voxel index
		'''
		ijk = np.clip((p - self.pMin) / self.voxel_extent, 0, self.res - 1).astype(np.int32)
		return np.ravel_multi_index(ijk.T, dims=(self.res, self.res, self.res), mode='clip', order='C')

	def idx_to_position(self, idx):
		'''
		Convert voxel index into a 3D coordinate.
		'''
		ijk = np.array(np.unravel_index(idx, shape=(self.res, self.res, self.res),  order='C')).T
		return ijk * self.voxel_extent + self.pMin

	def update(self, position, squared_residual):
		'''
		Update the squared residual estimates.
		'''
		for voxel_idx, value in zip(self.position_to_idx(position), squared_residual):
			self.update_average(voxel_idx, value)

	def get_squared_residual(self, position):
		indices = self.position_to_idx(position)
		return self.avg_squared_residual[indices]

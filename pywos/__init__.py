import jax
from jax import jit
from jax import lax
import jax.numpy as jnp
from .walk import (
	estimate_differential,
	estimate_solution,
	jit_estimate_boundary_differential as estimate_boundary_differential
)
from .util import (
	filter_args_and_call, 
	bbox_distance, 
	bbox_normal,
	jit_eval,
	jit_grad_eval
)

class HarmonicImplicitSurface:
	def __init__(self, 
				 bbox, alpha, n_blobs, epsilon, 
				 level_set = 0, offset_factor=5, margin=0.2, 
				 blob_mode = 'grid', rng_key = 0):
		self.bbox = bbox
		self.alpha = alpha
		self.epsilon = epsilon
		self.level_set = level_set
		self.offset_factor = offset_factor

		if blob_mode == 'random':
			x_min, y_min, x_max, y_max = self.bbox
			x_range = x_max - x_min
			y_range = y_max - y_min

			x_min += margin * x_range
			x_max -= margin * x_range
			y_min += margin * y_range
			y_max -= margin * y_range

			rng = jax.random.PRNGKey(rng_key)
			x_rng, y_rng = jax.random.split(rng)
			X = jax.random.uniform(x_rng, (n_blobs,), minval=x_min, maxval=x_max)
			Y = jax.random.uniform(y_rng, (n_blobs,), minval=y_min, maxval=y_max)

		elif blob_mode == 'box':
			x_min, y_min, x_max, y_max = self.bbox
			x_range = x_max - x_min
			y_range = y_max - y_min
			x_min += margin * x_range
			x_max -= margin * x_range
			y_min += margin * y_range
			y_max -= margin * y_range

			n_side = int(jnp.ceil(n_blobs / 4))
			X = jnp.zeros((4 * n_side, ))
			X = X.at[0*n_side:1*n_side].set(jnp.linspace(x_min, x_max, n_side, endpoint=False))
			X = X.at[1*n_side:2*n_side].set(x_max)
			X = X.at[2*n_side:3*n_side].set(jnp.linspace(x_max, x_min, n_side, endpoint=False))
			X = X.at[3*n_side:4*n_side].set(x_min)
		
			Y = jnp.zeros((4 * n_side, ))
			Y = Y.at[0*n_side:1*n_side].set(y_max)
			Y = Y.at[1*n_side:2*n_side].set(jnp.linspace(y_max, y_min, n_side, endpoint=False))
			Y = Y.at[2*n_side:3*n_side].set(y_min)
			Y = Y.at[3*n_side:4*n_side].set(jnp.linspace(y_min, y_max, n_side, endpoint=False))

		elif blob_mode == 'grid':
			n_side = int(jnp.ceil(jnp.sqrt(n_blobs)))
			n_blobs = n_side * n_side
			x = self.bbox[0] + jnp.linspace(margin, (1.0 - margin), n_side) * (self.bbox[2] - self.bbox[0])
			y = self.bbox[1] + jnp.linspace(margin, (1.0 - margin), n_side) * (self.bbox[3] - self.bbox[1])
			X, Y =  jnp.meshgrid(x, y)	

		else:
			X = jnp.zeros((n_blobs,))
			Y = jnp.zeros((n_blobs,))
	
		self.p = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
		self.a = jnp.ones(n_blobs)

	def eval(self, x):
		return jit_eval(self.p, self.a, self.bbox, x)
	
	def grad_eval(self, x):
		return jit_grad_eval(self.p, self.a, self.bbox, x)
		
	def normal_eval(self, x):
		return jit_normal(self.p, self.a, self.bbox, self.epsilon, x)

	def differential_eval(self, x):
		return jit_differential_eval(self.p, self.a, self.bbox, self.epsilon, x)
	
	def normal_velocity_eval(self, x):
		return jit_normal_velocity_eval(self.p, self.a, self.bbox, self.epsilon, x)

	def empty_sphere(self, x):
		return jit_empty_sphere(self.p, self.a, self.bbox, self.alpha, self.level_set, self.epsilon, x)

	def step(self, x, step_rng):
		return jit_step(self.p, self.a, self.bbox, self.alpha, self.level_set, self.epsilon, x, step_rng)

	def should_terminate(self, x):
		return jit_should_terminate(self.p, self.a, self.bbox, self.level_set, self.epsilon, x)

	def estimate_solution(self, x, rng, n_walks, max_steps = -1):
		return estimate_solution(self.p, self.a, self.bbox, self.alpha, self.level_set, self.epsilon, x, rng, n_walks, max_steps)

	def estimate_differential(self, x, rng, n_walks, n_rwalks, max_steps = -1):
		return estimate_differential(self.p, self.a, self.bbox, self.alpha, self.level_set, self.epsilon, self.offset_factor, x, rng, n_walks, n_rwalks, max_steps)
	
	def estimate_boundary_differential(self, x, epsilon):	
		return estimate_boundary_differential(self.p, self.a, self.bbox, self.level_set, epsilon, x)

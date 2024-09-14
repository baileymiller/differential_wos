import inspect
import jax
from jax import jit
from jax import lax
import jax.numpy as jnp

def filter_args_and_call(func, args):
	sig = inspect.signature(func)
	param_names = set(sig.parameters.keys())
	filtered_args = {key: value for key, value in args.items() if key in param_names}
	return func(**filtered_args)

def bbox_distance(bbox, p):
	p = jnp.asarray(p)
	bbox = jnp.asarray(bbox)
	bbox_min_x = bbox[0]
	bbox_min_y = bbox[1]
	bbox_max_x = bbox[2]
	bbox_max_y = bbox[3]
	bbox_center = jnp.array([bbox_max_x + bbox_min_x, bbox_max_y + bbox_min_y]) / 2.0
	bbox_extents = jnp.array([bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y]) / 2.0
	p = p - bbox_center
	d = jnp.abs(p) - bbox_extents
	return jnp.linalg.norm(jnp.maximum(d, 0.0), axis=1) + jnp.minimum(jnp.max(d, axis=1), 0.0)

def bbox_normal(bbox, p):
	'''
	BBOX normal computed by comparing to diagonal lines
	 
		*   (0,1)   * pos diagonal
		  *       *
		    *   *
  (-1, 0)	  *        (1,0)
			*    * 
		  *		   *
		*	(0,-1)   * neg diagonal
	'''
	p = jnp.asarray(p)
	bbox = jnp.asarray(bbox)
	bbox_min_x = bbox[0]
	bbox_min_y = bbox[1]
	bbox_max_x = bbox[2]
	bbox_max_y = bbox[3]
	slope = (bbox_max_y - bbox_min_y) / (bbox_max_x - bbox_min_x)
	bbox_center = jnp.array([bbox_max_x + bbox_min_x, bbox_max_y + bbox_min_y]) / 2.0
	p = p - bbox_center
	y_bound = slope * p[:, 0]
	above_pos_diagonal = jnp.greater(p[:, 1], y_bound)
	above_neg_diagonal = jnp.greater(p[:, 1], -y_bound)
	return jnp.where(
		above_pos_diagonal[:, jnp.newaxis],
		jnp.where(above_neg_diagonal[:, jnp.newaxis], jnp.array([0, 1]), jnp.array([-1, 0])),
		jnp.where(above_neg_diagonal[:, jnp.newaxis], jnp.array([0, -1]), jnp.array([1, 0]))
	)

def eval(p, a, bbox, x):
	r = jnp.linalg.norm(x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :], axis=-1)
	val = jnp.sum(a[jnp.newaxis, :] / r, axis=1)
	return jnp.where(bbox_distance(bbox, x) < 0, val, 0)

def grad_eval(p, a, bbox, x):
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r2 = jnp.power(jnp.linalg.norm(x_sub_p, axis=-1), 2)
	return jnp.where(bbox_distance(bbox, x)[:, jnp.newaxis] < 0, 
		jnp.sum(-a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r2[:, :, jnp.newaxis], axis=1),
		0
	)

def normal_eval(p, a, bbox, epsilon, x):
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r3 = jnp.power(jnp.linalg.norm(x_sub_p, axis=-1), 3)
	grad =  jnp.sum(-a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r2[:, :, jnp.newaxis], axis=1)
	return jnp.where(
		bbox_distance(bbox, x)[:, jnp.newaxis] < -epsilon,
		grad / jnp.linalg.norm(grad, axis=-1, keepdims=True),
		bbox_normal(bbox, x)
	)

def differential_eval(p, a, bbox, epsilon, x):
	differential = jnp.ones((x.shape[0], p.shape[0], 3))
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r = jnp.linalg.norm(x_sub_p, axis=-1)
	r3 = jnp.power(r, 3)
	differential = differential.at[:, :, 0].set(1.0 / r)
	differential = differential.at[:, :, 1:].set(a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis])
	return jnp.where(bbox_distance(bbox, x)[:, jnp.newaxis, jnp.newaxis] > -epsilon, 0, differential)

def normal_velocity_eval(p, a, bbox, epsilon, x):
	dudt = differential_eval(p, a, bbox, epsilon, x)
	grad_norm = jnp.linalg.norm(grad_eval(p, a, bbox, x), axis=-1)
	return jnp.where(dudt == 0, 0, -dudt / grad_norm[:, jnp.newaxis, jnp.newaxis])

def boundary_offset(p, a, bbox, epsilon, offset_factor, x):
	normal = normal_eval(p, a, bbox, epsilon, x)
	return x - (offset_factor * epsilon) * normal

def empty_sphere(p, a, bbox, alpha, level_set, epsilon, x):
	r = jnp.linalg.norm(x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :], axis=-1)
	R = alpha * jnp.min(r, axis=1)
	val = eval(p, a, bbox, x)
	outside_surface = val < level_set
	harnack_a = val / level_set
	harnack_bound = R / 2.0 * jnp.abs(harnack_a + 2.0 - jnp.sqrt(harnack_a * harnack_a + 8.0 * harnack_a))
	bbox_bound = jnp.abs(bbox_distance(bbox, x))
	return jnp.where(jnp.logical_and(outside_surface, val > 0), jnp.minimum(harnack_bound, bbox_bound), 0)

def step(p, a, bbox, alpha, level_set, epsilon, x, step_rng):
	R = empty_sphere(p, a, bbox, alpha, level_set, epsilon, x)
	theta = 2.0 * jnp.pi * jax.random.uniform(step_rng, shape=(x.shape[0],))
	x_step = jnp.stack([R * jnp.cos(theta), R * jnp.sin(theta)], axis=-1)
	u_step = R * R
	return x_step, u_step

def should_terminate(p, a, bbox, level_set, epsilon, x):
	val = eval(p, a, bbox, x)
	return jnp.logical_or(
		jnp.logical_or(
			val > level_set,
			bbox_distance(bbox, x) > -epsilon
		),
		jnp.abs(val - level_set) / jnp.linalg.norm(grad_eval(p, a, bbox, x), axis=-1) < epsilon
	)

@jit
def jit_eval(p, a, bbox, x):
	return eval(p, a, bbox, x)

@jit
def jit_grad_eval(p, a, bbox, x):
	return grad_eval(p, a, bbox, x)

@jit
def jit_normal_eval(p, a, bbox, epsilon, x):
	return normal(p, a, bbox, epsilon, x)

@jit
def jit_differential_eval(p, a, bbox, epsilon, x):
	return differential_eval(p, a, bbox, epsilon, x)

@jit 
def jit_normal_velocity_eval(p, a, bbox, epsilon, x):
	return normal_velocity_eval(p, a, bbox, epsilon, x)

@jit
def jit_empty_sphere(p, a, bbox, alpha, level_set, epsilon, x):
	return empty_sphere(p, a, bbox, alpha, level_set, epsilon, x)

@jit
def jit_step(p, a, bbox, alpha, level_set, epsilon, x, step_rng):
	return step(p, a, bbox, alpha, level_set, epsilon, x, step_rng)

@jit
def jit_should_terminate(p, a, bbox, level_set, epsilon, x):
	return should_terminate(p, a, bbox, level_set, epsilon, x)
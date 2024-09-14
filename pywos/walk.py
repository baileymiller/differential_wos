import inspect
import jax
from jax import jit
from jax import lax
import jax.numpy as jnp
from .util import (
	filter_args_and_call,
	bbox_distance,
	bbox_normal,
	jit_should_terminate
)

def solution_step(p, a, bbox, alpha, level_set, epsilon, x, step_rng):
	# compute some shared quantities
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r = jnp.linalg.norm(x_sub_p, axis=-1)
	r3 = jnp.power(r, 3)
	bbox_dist = bbox_distance(bbox, x)
	bbox_n = bbox_normal(bbox, x)
	val = jnp.where(bbox_dist < 0, jnp.sum(a[jnp.newaxis, :] / r, axis=1), 0)
	grad = jnp.sum(-a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis], axis=1)

	# compute empty sphere	
	max_R = alpha * jnp.min(r, axis=1)
	outside_surface = val < level_set
	harnack_a = val / level_set
	harnack_bound = max_R / 2.0 * jnp.abs(harnack_a + 2.0 - jnp.sqrt(harnack_a * harnack_a + 8.0 * harnack_a))
	R = jnp.maximum(jnp.where(jnp.logical_and(outside_surface, val > 0), jnp.minimum(harnack_bound, jnp.abs(bbox_dist)), 0), epsilon)

	# compute step and contribution u
	theta = 2.0 * jnp.pi * jax.random.uniform(step_rng, shape=(x.shape[0],))
	step_dir = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
	x_step = R[:, jnp.newaxis] * step_dir
	u_step = R * R

	# compute stopping condition
	reached_boundary = jnp.logical_or(
		jnp.logical_or(val > level_set, bbox_dist > -epsilon),
		jnp.abs(val - level_set) / jnp.linalg.norm(grad, axis=-1) < epsilon)

	return x_step, u_step, 2.0 * step_dir / (R + epsilon)[:, jnp.newaxis], reached_boundary

def differential_step(p, a, bbox, alpha, level_set, epsilon, offset_factor, x, step_rng):
	# compute some shared quantities
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r = jnp.linalg.norm(x_sub_p, axis=-1)
	r3 = jnp.power(r, 3)
	bbox_dist = bbox_distance(bbox, x)
	bbox_n = bbox_normal(bbox, x)
	val = jnp.where(bbox_dist < 0, jnp.sum(a[jnp.newaxis, :] / r, axis=1), 0)
	grad = jnp.sum(-a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis], axis=1)

	# compute empty sphere	
	max_R = alpha * jnp.min(r, axis=1)
	outside_surface = val < level_set
	harnack_a = val / level_set
	harnack_bound = max_R / 2.0 * jnp.abs(harnack_a + 2.0 - jnp.sqrt(harnack_a * harnack_a + 8.0 * harnack_a))
	R = jnp.maximum(jnp.where(jnp.logical_and(outside_surface, val > 0), jnp.minimum(harnack_bound, jnp.abs(bbox_dist)), 0), epsilon)

	# compute step and contribution u
	theta = 2.0 * jnp.pi * jax.random.uniform(step_rng, shape=(x.shape[0],))
	x_step = jnp.stack([R * jnp.cos(theta), R * jnp.sin(theta)], axis=-1)
	u_step = R * R

	# compute offset position
	grad_norm = jnp.linalg.norm(grad, axis=-1)
	implicit_n = grad / grad_norm[:, jnp.newaxis]
	normal = jnp.where(bbox_dist[:, jnp.newaxis] < -epsilon, implicit_n, bbox_n)
	recursive_x0 = x - (offset_factor * epsilon) * normal

	# compute boundary velocity
	differential = jnp.ones((x.shape[0], p.shape[0], 3))
	differential = differential.at[:, :, 0].set(1.0 / r)
	differential = differential.at[:, :, 1:].set(a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis])
	vn = jnp.where(bbox_dist[:, jnp.newaxis, jnp.newaxis] < -epsilon, -differential / grad_norm[:, jnp.newaxis, jnp.newaxis], 0)

	# compute stopping condition
	reached_boundary = jnp.logical_or(
		jnp.logical_or(val > level_set, bbox_dist > -epsilon),
		jnp.abs(val - level_set) / jnp.linalg.norm(grad, axis=-1) < epsilon)

	return x_step, u_step, recursive_x0, vn, reached_boundary

def estimate_solution(p, a, bbox, alpha, level_set, epsilon, x, rng, n_walks, max_steps):
	n_pts = x.shape[0]
	n_blobs = p.shape[0]
	max_iters = n_walks * max_steps
	state = dict(
		p=p, a=a, bbox=bbox, alpha=alpha, 
		level_set=level_set, epsilon=epsilon,
		n_walks=n_walks, curr_n_walks=jnp.zeros((n_pts,)),
		rng=rng, step_rng=rng,
		x0=jnp.copy(x), x=x,
		u=jnp.zeros((n_pts,)), dudx=jnp.zeros((n_pts, 2)),
		curr_u=jnp.zeros((n_pts,)), first_dir=jnp.zeros((n_pts,2)),
		has_walks_remaining=jnp.full((n_pts, ), True), 
		iters=0
	)
	def cond(state):
		return jnp.logical_and(jnp.any(state['has_walks_remaining']), 
							   max_iters < 0 or state['iters'] < max_iters)

	def body(state):
		state['has_walks_remaining'] = jnp.less(state['curr_n_walks'], state['n_walks'])
		state['rng'], state['step_rng'] = jax.random.split(state['rng'])
		x_step, u_step, step_dir, reached_boundary = filter_args_and_call(solution_step, state)

		state['first_dir'] = jnp.where(state['first_dir'] == jnp.array([0, 0]), step_dir, state['first_dir'])

		active = jnp.logical_and(jnp.logical_not(reached_boundary), state['has_walks_remaining'])
		state['x'] += jnp.where(active[:, jnp.newaxis], x_step, 0)
		state['curr_u'] += u_step

		absorbed = jnp.logical_and(reached_boundary, state['has_walks_remaining'])
		state['x'] = jnp.where(absorbed[:,  jnp.newaxis], state['x0'], state['x'])

		# update gradient with control variate
		state['dudx'] += jnp.where(absorbed[:, jnp.newaxis], (state['curr_u'] - state['u'] / jnp.maximum(1, state['curr_n_walks']))[:, jnp.newaxis] * state['first_dir'], 0)
		state['first_dir'] = jnp.where(absorbed[:, jnp.newaxis], 0, state['first_dir'])

		# update solution estimate
		state['curr_n_walks'] += jnp.where(absorbed, 1, 0)
		state['u'] += jnp.where(absorbed, state['curr_u'], 0)
		state['curr_u'] = jnp.where(absorbed, 0, state['curr_u'])

		state['iters'] += 1
		return state

	state = lax.while_loop(cond, body, state)
	return state['u'] / n_walks, state['dudx'] / n_walks, state['rng']

def estimate_differential(p, a, bbox, alpha, level_set, epsilon, offset_factor, x, rng, n_walks, n_rwalks, max_steps = 1e4):	
	n_pts = x.shape[0]
	n_blobs = p.shape[0]
	max_iters = n_walks * n_rwalks * max_steps
	if n_rwalks <= 0:
		return estimate_solution(p, a, b, bbox, alpha, level_set, epsilon, x, rng, n_walks), jnp.zeros((n_pts, n_blobs, 3))

	state = dict(
		p=p, a=a, bbox=bbox, alpha=alpha, 
		level_set=level_set, epsilon=epsilon, offset_factor=offset_factor,
		n_walks=n_walks, curr_n_walks=jnp.zeros((n_pts,)).astype(jnp.int32),
		n_rwalks=n_rwalks, curr_n_rwalks=jnp.zeros((n_pts,)).astype(jnp.int32),
		rng=rng, step_rng=rng,
		is_primary=jnp.full(n_pts, True),
		x0=jnp.copy(x), x=x, recursive_x0=jnp.zeros_like(x), recursive_vn=jnp.zeros((n_pts, n_blobs, 3)),
		u=jnp.zeros((n_pts,)), dudn=jnp.zeros((n_pts,)), dudt=jnp.zeros((n_pts, n_blobs, 3)),
		has_walks_remaining=jnp.full((n_pts,), True),
		iters=0
	)
	state['curr_n_walks'] = jnp.where(filter_args_and_call(jit_should_terminate, state), state['n_walks'], state['curr_n_walks'])

	def cond(state):
		return jnp.logical_and(jnp.any(state['has_walks_remaining']),
							   max_iters < 0 or state['iters'] < max_iters)

	def body(state):
		# take step
		state['has_walks_remaining'] = jnp.less(state['curr_n_walks'], state['n_walks'])
		state['rng'], state['step_rng'] = jax.random.split(state['rng'])
		x_step, u_step, recursive_x0, recursive_vn, reached_boundary = filter_args_and_call(differential_step, state)

		# update x, u, dudn
		active = jnp.logical_and(jnp.logical_not(reached_boundary), state['has_walks_remaining'])
		active_primary = jnp.logical_and(active, state['is_primary'])
		active_recursive = jnp.logical_and(active, jnp.logical_not(state['is_primary']))
		state['x'] += jnp.where(active[:, jnp.newaxis], x_step, 0)
		state['u'] += jnp.where(active_primary, u_step, 0)
		state['dudn'] += jnp.where(active_recursive, -u_step / (state['offset_factor'] * state['epsilon']), 0)

		# update walk counts
		absorbed = jnp.logical_and(reached_boundary, state['has_walks_remaining'])
		primary_absorbed = jnp.logical_and(absorbed, state['is_primary'])
		recursive_absorbed = jnp.logical_and(absorbed, jnp.logical_not(state['is_primary']))
		state['curr_n_rwalks'] += jnp.where(recursive_absorbed, 1, 0)
		recursive_walks_finished = jnp.greater(state['curr_n_rwalks'], state['n_rwalks'])
		state['curr_n_walks'] += jnp.where(recursive_walks_finished, 1, 0)
		state['curr_n_rwalks'] = jnp.where(recursive_walks_finished, 0, state['curr_n_rwalks'])
		state['is_primary'] = jnp.where(primary_absorbed, False, jnp.where(recursive_walks_finished, True, state['is_primary']))

		# primary absorbed, offset to start recursive walk
		state['recursive_x0'] = jnp.where(primary_absorbed[:, jnp.newaxis], recursive_x0, state['recursive_x0'])
		state['recursive_vn'] = jnp.where(primary_absorbed[:, jnp.newaxis, jnp.newaxis], recursive_vn, state['recursive_vn'])
		state['x'] = jnp.where(absorbed[:, jnp.newaxis], state['recursive_x0'], state['x'])

		# recursive walks finished, compute dudt, reset dudn, and return to start of primary walk
		state['dudt'] += jnp.where(recursive_walks_finished[:, jnp.newaxis, jnp.newaxis], state['recursive_vn'] * (-state['dudn'] / state['n_rwalks'])[:, jnp.newaxis, jnp.newaxis], 0)
		state['dudn'] = jnp.where(recursive_walks_finished, 0, state['dudn'])
		state['x'] = jnp.where(recursive_walks_finished[:, jnp.newaxis], state['x0'], state['x'])

		# increment
		state['iters'] += 1

		return state

	state = lax.while_loop(cond, body, state)
	return state['u'] / n_walks, state['dudt'] / n_walks, state['rng']

def estimate_boundary_differential(p, a, bbox, level_set, epsilon, x):	
	# compute some shared quantities
	x_sub_p = x[:, jnp.newaxis, :] - p[jnp.newaxis, :, :]
	r = jnp.linalg.norm(x_sub_p, axis=-1)
	r3 = jnp.power(r, 3)
	bbox_dist = bbox_distance(bbox, x)
	bbox_n = bbox_normal(bbox, x)
	val = jnp.where(bbox_dist < 0, jnp.sum(a[jnp.newaxis, :] / r, axis=1), 0)
	grad = jnp.sum(-a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis], axis=1)
	grad_norm = jnp.linalg.norm(grad, axis=-1)

	# determine which points are inside of the mask (i.e. within epsilon of the boundary)
	boundary_mask = jnp.logical_and(
		bbox_dist < -epsilon,
		jnp.abs(val - level_set) / jnp.linalg.norm(grad, axis=-1) < epsilon
	)

	# compute boundary velocity
	differential = jnp.ones((x.shape[0], p.shape[0], 3))
	differential = differential.at[:, :, 0].set(1.0 / r)
	differential = differential.at[:, :, 1:].set(a[jnp.newaxis, :, jnp.newaxis] * x_sub_p / r3[:, :, jnp.newaxis])
	vn = -differential / grad_norm[:, jnp.newaxis, jnp.newaxis]
	vn = vn * jnp.where(val < level_set, 1, -1)[:, jnp.newaxis, jnp.newaxis]
	return vn / (2.0 * epsilon), jnp.where(boundary_mask, 1, 0)

@jit
def jit_estimate_boundary_differential(p, a, bbox, level_set, epsilon, x):
	return estimate_boundary_differential(p, a, bbox, level_set, epsilon, x)

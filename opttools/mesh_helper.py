import torch
from cholespy import CholeskySolverF, MatrixType
from .largesteps import compute_matrix

class MeshHelper:
	def __init__(self, alpha = None, lambda_ = None):
		self.M = None
		self.M_solver = None
		self.alpha = alpha
		self.lambda_ = lambda_
		assert self.lambda_ or self.alpha, 'Either alpha or lambda_ must be defined'

	def register_geometry(self, verts, faces, cotan = False):
		assert self.lambda_ or self.alpha, 'Either alpha or lambda_ must be defined'
		verts = torch.from_numpy(verts)
		faces = torch.from_numpy(faces)
		self.M = compute_matrix(verts, faces, self.lambda_, self.alpha, cotan)
		self.M_solver = CholeskySolverF(self.M.shape[0], self.M.indices()[0], self.M.indices()[1], self.M.values(), MatrixType.COO)

	def precondition_gradient(self, grad):
		assert self.M_solver, 'No geometry registered'
		precond_grad = torch.zeros_like(grad)
		self.M_solver.solve(grad.detach(), precond_grad)
		return precond_grad

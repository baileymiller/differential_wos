'''

                                    ORIGINAL LICENSE                                 
---------------------------------------------------------------------------------------
 Copyright (c) 2021 Baptiste Nicolet <baptiste.nicolet@epfl.ch>, All rights reserved.
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
 
1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to the author of this software, without
imposing a separate written license agreement for such Enhancements, then you
hereby grant the following license: a non-exclusive, royalty-free perpetual
license to install, use, modify, prepare derivative works, incorporate into
other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.
---------------------------------------------------------------------------------------

code adapted from: https://github.com/rgl-epfl/large-steps-pytorch

'''

import torch

def laplacian_cot(verts, faces, device):
	"""
	Compute the cotangent laplacian

	Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html

	Parameters
	----------
	verts : torch.Tensor
		Vertex positions.
	faces : torch.Tensor
		array of triangle faces.
	"""

	# V = sum(V_n), F = sum(F_n)
	V, F = verts.shape[0], faces.shape[0]

	face_verts = verts[faces]
	v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

	# Side lengths of each triangle, of shape (sum(F_n),)
	# A is the side opposite v1, B is opposite v2, and C is opposite v3
	A = (v1 - v2).norm(dim=1)
	B = (v0 - v2).norm(dim=1)
	C = (v0 - v1).norm(dim=1)

	# Area of each triangle (with Heron's formula); shape is (sum(F_n),)
	s = 0.5 * (A + B + C)
	# note that the area can be negative (close to 0) causing nans after sqrt()
	# we clip it to a small positive value
	area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

	# Compute cotangents of angles, of shape (sum(F_n), 3)
	A2, B2, C2 = A * A, B * B, C * C
	cota = (B2 + C2 - A2) / area
	cotb = (A2 + C2 - B2) / area
	cotc = (A2 + B2 - C2) / area
	cot = torch.stack([cota, cotb, cotc], dim=1)
	cot /= 4.0

	# Construct a sparse matrix by basically doing:
	# L[v1, v2] = cota
	# L[v2, v0] = cotb
	# L[v0, v1] = cotc
	ii = faces[:, [1, 2, 0]]
	jj = faces[:, [2, 0, 1]]
	idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
	L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

	# Make it symmetric; this means we are also setting
	# L[v2, v1] = cota
	# L[v0, v2] = cotb
	# L[v1, v0] = cotc
	L += L.t()

	# Add the diagonal indices
	vals = torch.sparse.sum(L, dim=0).to_dense()
	indices = torch.arange(V, device=device)
	idx = torch.stack([indices, indices], dim=0)
	L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
	return L

def laplacian_uniform(verts, faces, device):
	"""
	Compute the uniform laplacian

	Parameters
	----------
	verts : torch.Tensor
		Vertex positions.
	faces : torch.Tensor
		array of triangle faces.
	"""
	V = verts.shape[0]
	F = faces.shape[0]

	# Neighbor indices
	if faces.shape[1] == 3:
		ii = faces[:, [1, 2, 0]].flatten()
		jj = faces[:, [2, 0, 1]].flatten()
	else:
		ii = faces[:, [0, 1]].flatten()
		jj = faces[:, [1, 0]].flatten()
	
	adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
	adj_values = torch.ones(adj.shape[1], device=device, dtype=torch.float)

	# Diagonal indices
	diag_idx = adj[0]

	# Build the sparse matrix
	idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
	values = torch.cat((-adj_values, adj_values))

	# The coalesce operation sums the duplicate indices, resulting in the
	# correct diagonal
	return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def compute_matrix(verts, faces, lambda_, alpha=None, cotan=False, device='cpu'):
	"""
	Build the parameterization matrix.

	If alpha is defined, then we compute it as (1-alpha)*I + alpha*L otherwise
	as I + lambda*L as in the paper. The first definition can be slightly more
	convenient as it the scale of the resulting matrix doesn't change much
	depending on alpha.

	Parameters
	----------
	verts : torch.Tensor
		Vertex positions
	faces : torch.Tensor
		Triangle faces
	lambda_ : float
		Hyperparameter lambda of our method, used to compute the
		parameterization matrix as (I + lambda_ * L)
	alpha : float in [0, 1[
		Alternative hyperparameter, used to compute the parameterization matrix
		as ((1-alpha) * I + alpha * L)
	cotan : bool
		Compute the cotangent laplacian. Otherwise, compute the combinatorial one
	"""
	if cotan:
		assert faces.shape[1] == 3, 'Cotan not supported for line segments'
		L = laplacian_cot(verts, faces, device)
	else:
		L = laplacian_uniform(verts, faces, device)

	idx = torch.arange(verts.shape[0], dtype=torch.long, device=device)
	eye = torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), torch.ones(verts.shape[0], dtype=torch.float, device=device), (verts.shape[0], verts.shape[0]))
	if alpha is None:
		M = torch.add(eye, lambda_*L) # M = I + lambda_ * L
	else:
		if alpha < 0.0 or alpha >= 1.0:
			raise ValueError(f"Invalid value for alpha: {alpha} : it should take values between 0 (included) and 1 (excluded)")
		M = torch.add((1-alpha)*eye, alpha*L) # M = (1-alpha) * I + alpha * L
	return M.coalesce()

import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import imageio

import sys
sys.path.append('../')
sys.path.append('../zombie/build/pyzombie')

import pyzombie as pz
import opttools

# setings + data
LEARNING_RATE = 5e-2
N_ITERS = 100
DIFFERENTIAL_WPP = 32
PRIMARY_WPP = 8
EPSILON_SHELL = 1e-3
BOUNDARY_FD_OFFSET = 1e-2
PRECONDITIONER_ALPHA = 0.95
MAX_WALK_LENGTH = 1024

DATA = np.load('data.npz')
REFERENCE_SOLUTION = DATA['ref_u']
REFERENCE_POINTS = DATA['pts']

# setup PDE
dirichlet_boundary_condition_grid = pz.Grid2D([-1.5, -1.5],[3, 3], "", False)
dirichlet_boundary_condition_grid.setFromFunction(np.array([32, 32]), lambda x: 1)
pde = pz.PDE2D()
pz.setZeroBoundaryConditions2D(pde, 0.0)
pz.setDirichletGridBoundaryCondition2D(pde, dirichlet_boundary_condition_grid)
pde.absorption = 10.0

# setup initial polyline boundary
curr_v, curr_l = opttools.create_circle_mesh([0, 0], 1, 100)
geometry = pz.MeshGeometry2D(
    bbox=pz.BBox2D.create(np.array([-2, -2]), np.array([2, 2])), 
    bboxIsDirichlet=True, 
    domainIsWatertight=False,
    buildBVH=True,
    computeWeightedNormals=True, 
    computeSilhouettes=False, 
    id="d"
)
geometry.setDirichlet(curr_v, curr_l)
geometry.setDirichletDisplacement(pz.types.MeshDisplacementType.VertexTranslation)

# setup optimization
x = torch.from_numpy(curr_v).type(torch.float)
x.requires_grad = True
optimizer = opttools.UniformAdam([x], lr=LEARNING_RATE)
mesh_helper = opttools.MeshHelper(PRECONDITIONER_ALPHA) 
mesh_helper.register_geometry(curr_v, curr_l, cotan=False)

walk_settings = pz.WalkSettings(0, EPSILON_SHELL, EPSILON_SHELL, MAX_WALK_LENGTH, False)
walk_settings.solutionWeightedDifferentialBatchSize = DIFFERENTIAL_WPP
walk_settings.ignoreShapeDifferential = False
walk_settings.boundaryGradientOffset = BOUNDARY_FD_OFFSET
walk_settings.stepsBeforeApplyingTikhonov = 0

# estimation data
pdfs = np.ones((REFERENCE_POINTS.shape[0])) / geometry.bbox.volume()
sample_ests = pz.createSampleEstimationData2D(DIFFERENTIAL_WPP, PRIMARY_WPP, pz.types.EstimationQuantity.Solution, np.zeros_like(REFERENCE_POINTS))

# visualization 
os.makedirs('./visual', exist_ok=True)
colormap = plt.cm.turbo
normalized  = pltc.Normalize(vmin=0, vmax=1.0)
REFERENCE_RGB  = colormap(normalized(REFERENCE_SOLUTION).reshape((64, 64)))[:, :, :3]
REFERENCE_RGB = (REFERENCE_RGB.reshape(64, 64, 3) * 255).astype(np.uint8)

# save geometry before optimization
opttools.save_segment_soup('initial.obj', np.hstack((curr_v, np.zeros((curr_v.shape[0], 1)))), curr_l)

# run optimization
for curr_iter in tqdm(range(N_ITERS)):

	# estimate derivative with differential WoS
    evaluation_pts = pz.createSamplePoints2D(geometry.queries, REFERENCE_POINTS, pdfs, 0.0, True)
    pz.solve2D(geometry.queries, pde, walk_settings, sample_ests,  evaluation_pts, False)
    L = pz.l2Loss2D(evaluation_pts, REFERENCE_SOLUTION)

        
    # optimization step using estimate of derivative
    def get_param_idx(key):
        primitive_id, local_vertex_id, coord = map(int, key.split(".")[1:])
        vertex_id = curr_l[primitive_id, local_vertex_id]
        return 2 * vertex_id + coord
    
    optimizer.zero_grad()
    grad = L.differential.filter("d").dense(curr_v.shape[0] * curr_v.shape[1], get_param_idx)
    grad = grad.reshape(curr_v.shape)
    x.grad = torch.from_numpy(grad)
    x.grad = mesh_helper.precondition_gradient(x.grad)
    optimizer.step()

	# rebuild BVH and update preconditioner
    curr_v = x.detach().numpy()
    geometry.setDirichlet(curr_v, curr_l)
    geometry.setDirichletDisplacement(pz.types.MeshDisplacementType.VertexTranslation)
    
    # save optimized solution profile (progressively visualize geometry)
    solution = np.array(pz.getSolution2D(evaluation_pts))
    solution_rgb = colormap(normalized(solution).reshape((64, 64)))[:, :, :3]
    solution_rgb = (solution_rgb.reshape(64, 64, 3) * 255).astype(np.uint8)
    imageio.imwrite('./visual/{:03}.png'.format(curr_iter), np.concatenate((solution_rgb, REFERENCE_RGB), axis=1))

# save geometry after optimization
opttools.save_segment_soup('optimized.obj', np.hstack((curr_v, np.zeros((curr_v.shape[0], 1)))), curr_l)

del mesh_helper

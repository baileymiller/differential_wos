import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator

def get_grid(dims, bbox, res, indexing = 'xy'):
    '''
    Construct slice plane where (dim[0], dim[1], dim[2]) are slice plane extent.
    '''
    X = np.linspace(bbox.pMin[dims[0]], bbox.pMax[dims[0]], res)
    Y = np.linspace(bbox.pMin[dims[1]], bbox.pMax[dims[1]], res)
    Z = np.linspace(bbox.pMin[dims[2]], bbox.pMax[dims[2]], res)
    XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing=indexing)
    return np.stack((XX, YY, ZZ), axis=-1)

def get_interpolator(values, dims, bbox, res, indexing = 'xy'):
    X = np.linspace(bbox.pMin[dims[0]], bbox.pMax[dims[0]], res)
    Y = np.linspace(bbox.pMin[dims[1]], bbox.pMax[dims[1]], res)
    Z = np.linspace(bbox.pMin[dims[2]], bbox.pMax[dims[2]], res)
    if indexing == 'xy':
        values = values.transpose(1, 0, 2)
        return RegularGridInterpolator((Y, X, Z), values)
    else:
        return RegularGridInterpolator((X, Y, Z), values)

def generate_stratified_samples(dims, bbox, n_samples, indexing = 'xy'):
    res = int(max(np.ceil(np.power(n_samples, 1.0 / 3.0)), 1))
    
    # generate samples at corner of voxels
    X = np.linspace(bbox.pMin[dims[0]], bbox.pMax[dims[0]], res, endpoint = False)
    Y = np.linspace(bbox.pMin[dims[1]], bbox.pMax[dims[1]], res, endpoint = False)
    Z = np.linspace(bbox.pMin[dims[2]], bbox.pMax[dims[2]], res, endpoint = False)
    XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing=indexing)
    samples = np.stack((XX, YY, ZZ), axis=-1).reshape(-1, 3)

    # offset points in voxel
    voxel_length_x = (bbox.pMax[dims[0]] - bbox.pMin[dims[0]]) / res
    voxel_length_y = (bbox.pMax[dims[1]] - bbox.pMin[dims[1]]) / res
    voxel_length_z = (bbox.pMax[dims[2]] - bbox.pMin[dims[2]]) / res
    samples += np.random.random(samples.shape) * np.array([[voxel_length_x, voxel_length_y, voxel_length_z]])
    pdf = np.ones(len(samples)) / bbox.volume()
    return samples, pdf

def get_slice_plane(dims, Z, bbox, res, indexing='xy'):
    '''
    Construct slice plane where (dim[0], dim[1]) are slice plane extent and 
    slice is along the dim[2] axis at coordinate z
    '''
    X = np.linspace(bbox.pMin[dims[0]], bbox.pMax[dims[0]], res)
    Y = np.linspace(bbox.pMin[dims[1]], bbox.pMax[dims[1]], res)
    XX, YY = np.meshgrid(X, Y, indexing = indexing)
    plane = np.zeros((res, res, 3))
    plane[:, :, dims[0]] = XX
    plane[:, :, dims[1]] = YY
    plane[:, :, dims[2]] = Z * np.ones_like(XX)
    return plane

def apply_transform(T, v):
    v_hom = np.concatenate([v, np.ones_like(v[:, :1])], axis=-1)
    v_transformed = (v_hom @ T.transpose())
    return v_transformed[:, :3] / v_transformed[:, 3][:, None]

def get_slice_area(dims, bbox):
    '''
    Return the 2D area of the slice plane of the bounding box (bbox) for the specified dimensions (dims)
    '''
    return (bbox.pMax[dims[1]] - bbox.pMin[dims[1]]) * (bbox.pMax[dims[0]] - bbox.pMin[dims[0]])

def clamp_2d(v, x_range, y_range):
    x_clamped = np.clip(v[0], x_range[0], x_range[1])
    y_clamped = np.clip(v[1], y_range[0], y_range[1])
    return np.array([x_clamped, y_clamped])

def interpolate_grid_values(dims, bbox, values, pts):
    '''
    Interpolate slice plane values (slice specified by dims and bbox) at a set of points
    '''
    res_x, res_y = values.shape
    return values[tuple(zip(*np.array([
        clamp_2d(
            ((np.array([pt[i, dims[0]], pt[i, dims[1]]]) - np.array([bbox.pMin[dims[0]], bbox.pMin[dims[1]]])) /
            (np.array([bbox.pMax[dims[0]], bbox.pMax[dims[1]]]) - np.array([bbox.pMin[dims[0]], bbox.pMin[dims[1]]])) * 
            np.array([res_x, res_y])),
            [0, res_x - 1],
            [0, res_y - 1]
        )
        for i in range(pts.shape[0])
    ]).astype(int)))]

def pose_to_params(T):
    '''
    Converts pose matrix to 7 parameters (abcd + translation)
    '''
    scale = np.linalg.norm(T[:3, :3], axis=-1)
    assert np.all(np.abs(scale - scale[0]) < 1e-4), 'all scale components must be equal'
    qi, qj, qk, qr = Rotation.as_quat(Rotation.from_matrix(T[:3, :3]))
    qi *= scale[0]
    qj *= scale[0]
    qk *= scale[0]
    qr *= scale[0]
    return np.array([qr, qi, qj, qk, T[0, 3], T[1, 3], T[2, 3]])

def params_to_pose(params):
    '''
    Converts following parameters to a pose matrix
        params[0]: a
        params[1]: b
        params[2]: c
        params[3]: d
        params[4]: x translation
        params[5]: y translation
        params[6]: z translation
    '''
    a, b, c, d = params[0], params[1], params[2], params[3]
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d
    tx, ty, tz = params[4], params[5], params[6]
    return np.array([
        [a2 + b2 - c2 - d2, 2 * (b * c - a * d), 2 * (b * d + a * c), tx],
        [2 * (b * c + a * d), a2 - b2 + c2 - d2, 2 * (c * d - a * b), ty],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a2 - b2 - c2 + d2, tz],
        [0, 0, 0, 1]
    ])

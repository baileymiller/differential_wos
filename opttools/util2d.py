import numpy as np

def load_obj(filename):
    vertices = []
    lines = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()

            # Ignore empty lines and comments
            if not parts or parts[0] == '#':
                continue

            # Extract vertex data
            if parts[0] == 'v':
                x, y = float(parts[1]), float(parts[2])
                vertices.append((x, y))

            # Extract line data
            if parts[0] == 'l':
                line_indices = list(map(int, parts[1:]))
                lines.append(line_indices)

    return np.array(vertices), np.array(lines) - 1

def scale_and_center(vertices):
    min_x = min(vertices, key=lambda t: t[0])[0]
    max_x = max(vertices, key=lambda t: t[0])[0]
    min_y = min(vertices, key=lambda t: t[1])[1]
    max_y = max(vertices, key=lambda t: t[1])[1]

    # Compute scaling factor
    current_width = max_x - min_x
    current_height = max_y - min_y
    scale_factor = 2.0 / max(current_width, current_height)

    # Scale and translate vertices
    scaled_vertices = []
    for x, y in vertices:
        scaled_x = (x - min_x) * scale_factor
        scaled_y = (y - min_y) * scale_factor

        # Translate to center at origin
        translated_x = scaled_x - (current_width * scale_factor / 2)
        translated_y = scaled_y - (current_height * scale_factor / 2)
        
        scaled_vertices.append((translated_x, translated_y))

    return scaled_vertices

def create_circle_mesh(center, radius, num_points):
    # Create evenly spaced angles
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)

    # Convert polar coordinates to Cartesian coordinates
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    vertices = list(zip(x, y))

    # Create lines connecting the vertices
    lines = [[i, i+1] for i in range(num_points - 1)]
    lines.append([num_points - 1, 0])  # Close the circle

    return np.array(vertices), np.array(lines)

def create_circle_bezier(center, radius, n, init_color):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.array([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)]).T

    point = np.zeros((n, 2))
    delta = np.zeros((n, 2))
    scale = np.zeros((n, 2))
    index = np.zeros((n, 2))
    color = np.zeros((n, 3))
    for i in range(n):
        normal = (points[i] - center)
        tangent = np.array([-normal[1], normal[0]])
        tangent /= np.linalg.norm(tangent)

        point[i] = points[i]
        delta[i] = tangent * radius / 3.0 / n
        scale[i] = [0, 0]
        index[i] = [i, (i + 1) % n]
        color[i] = init_color

    return point, delta, scale, index, color

def plot_polylines(ax, data_sets):
    for vertices, lines, color in data_sets:
        for line_indices in lines:
            for i in range(len(line_indices) - 1):
                start_vertex = vertices[line_indices[i]]  # .obj is 1-indexed
                end_vertex = vertices[line_indices[i + 1]]
                ax.plot([start_vertex[0], end_vertex[0]], 
                         [start_vertex[1], end_vertex[1]], color)

    ax.axhline(0, color='grey',linewidth=0.5)
    ax.axvline(0, color='grey',linewidth=0.5)
    ax.grid(True, which='both')
    ax.set_aspect('equal', adjustable='box')

def get_affine_transform(T, S, R):
    return np.array([
        [S[0] * np.cos(R), -S[0] * np.sin(R), T[0]],
        [S[1] * np.sin(R), S[1] * np.cos(R), T[1]],
        [0, 0, 1]
    ])


def apply_transform(A, p):
    assert A.shape == (3,3), 'transform must be 3x3'
    homogeneous_p = np.hstack((p, np.ones((len(p), 1)))).dot(A.T)
    return homogeneous_p[:, :2] / homogeneous_p[:, 2][:, np.newaxis]

def get_grid(dims, bbox, res, indexing = 'xy'):
    '''
    Construct slice plane where (dim[0], dim[1]) are slice plane extent.
    '''
    X = np.linspace(bbox.pMin[dims[0]], bbox.pMax[dims[0]], res)
    Y = np.linspace(bbox.pMin[dims[1]], bbox.pMax[dims[1]], res)
    XX, YY = np.meshgrid(X, Y, indexing=indexing)
    return np.stack((XX, YY), axis=-1)

def clamp_2d(v, x_range, y_range):
    x_clamped = np.clip(v[0], x_range[0], x_range[1])
    y_clamped = np.clip(v[1], y_range[0], y_range[1])
    return np.array([x_clamped, y_clamped])

def _interpolate_grid_values(dims, bbox, values, pts):
    res = np.array([values.shape[dims[0]], values.shape[dims[1]]])
    pMin = np.array([bbox.pMin[dims[0]], bbox.pMin[dims[1]]])
    pMax = np.array([bbox.pMax[dims[0]], bbox.pMax[dims[1]]])
    scale = (pMax - pMin)
    indices = np.zeros((pts.shape[0], 2))
    for i in range(pts.shape[0]):
        pt = np.array([pts[i, dims[0]], pts[i, dims[1]]])
        relative_pos = (pt - pMin) / scale
        indices[i, :] = np.clip(res * relative_pos, [0, 0], res - 1)
    indices = tuple(zip(*indices.astype(int)))
    return values[indices]

def interpolate_grid_values(dims, bbox, values, pz_pts):
    pts = np.array([pz_pt.pt for pz_pt in pz_pts])
    return _interpolate_grid_values(dims, bbox, values, pts) 

def interpolate_boundary_grid_values(dims, bbox, values, pz_boundary_pts, eps = 1e-3):
    pts = np.array([pz_pt.pt - pz_pt.n * eps for pz_pt in pz_boundary_pts])
    exterior_pts = np.array([pz_pt.pt + pz_pt.n * eps for pz_pt in pz_boundary_pts])
    return (
        _interpolate_grid_values(dims, bbox, values, pts),
        _interpolate_grid_values(dims, bbox, values, exterior_pts)
    )

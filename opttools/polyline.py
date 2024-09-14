import torch
import numpy as np
from scipy.interpolate import interp1d

def create_polyline(start, end, n_segments, constant_vt = 0):
    '''
    Create a 0 indexed polyline that interpolates between start and end with n segments.
    Set the texture coordinates for the vertices to all have the same value.
    '''
    interp = np.linspace(0, 1, num=(n_segments + 1), endpoint=True)
    v = start[None, :] * (1.0 - interp[:, None]) + end[None, :] * interp[:, None]
    vt = np.ones((v.shape[0], 2), dtype=np.int32) * constant_vt
    l = [np.array(range(0, n_segments))]
    return np.array(v, dtype=np.float32), vt, np.array(l, dtype=np.int32)

def create_polyloop(c, r, n_segments, constant_vt = 0):
    theta = np.linspace(0, 2 * np.pi, num=n_segments, endpoint=False)
    v = np.stack([np.sin(theta) * r, np.zeros_like(theta), np.cos(theta) * r], axis=-1)
    v += c[None, :]
    vt = np.ones((v.shape[0], 2), dtype=np.int32) * constant_vt
    l = [np.array(range(0, n_segments + 1)) % n_segments]
    return np.array(v, dtype=np.float32), vt, np.array(l, dtype=np.int32)

def combine_polyline_objs(polyline_objs):
    n_v = 0
    for v, vt, l in polyline_objs:
        n_v += v.shape[0]
    curr_v = 0
    combined_v = np.zeros((n_v, 3))
    combined_vt = np.zeros((n_v, 2))
    combined_l = []
    for v, vt, l in polyline_objs:        
        combined_v[curr_v:curr_v+len(v)] = v
        combined_vt[curr_v:curr_v+len(v)] = vt
        combined_l.extend(l + np.array(curr_v))
        curr_v += len(v)
    return np.array(combined_v), np.array(combined_vt), combined_l
    
def save_polyline(filename, v, vt, l):
    '''
    Save polylines into OBJ format. Converts 0 based indexing to 1 based indexing
    in the process.
    '''
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")
        for i in range(vt.shape[0]):
            file.write(f"vt {vt[i, 0]} {vt[i, 0]}\n")
        for polyline in l:
            line_str = 'l '
            for i in range(polyline.shape[0]):
                line_str += f'{str(polyline[i] + 1)} '
            line_str += '\n'
            file.write(line_str)

def save_polyline_with_data(filename, v, vt, l, data):
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")

        for i in range(vt.shape[0]):
            value = data[vt[i, 0]]
            file.write(f"vt {value} {value}\n")

        for polyline in l:
            line_str = 'l '
            for i in range(polyline.shape[0]):
                line_str += f'{str(polyline[i] + 1)} '
            line_str += '\n'
            file.write(line_str)

def save_polyline_with_normals(filename, v, vn, l):
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")

        for i in range(vn.shape[0]):
            file.write(f"vn {vn[i, 0]} {vn[i, 1]} {vn[i, 2]}\n")

        for polyline in l:
            line_str = 'l '
            for i in range(polyline.shape[0]):
                line_str += f'{str(polyline[i] + 1)} '
            line_str += '\n'
            file.write(line_str)

def save_segment_soup_with_normals(filename, v, vn, segments):
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")
        
        for i in range(vn.shape[0]):
            file.write(f"vn {vn[i, 0]} {vn[i, 1]} {vn[i, 2]}\n")

        for segment in segments:
            file.write(f'l {segment[0] + 1} {segment[1] + 1}\n')

def save_segment_soup_with_data(filename, v, vt, segments, data):
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")

        for i in range(vt.shape[0]):
            value = data[vt[i, 0]]
            file.write(f"vt {value} {value}\n")

        for segment in segments:
            file.write(f'l {segment[0] + 1} {segment[1] + 1}\n')

def save_segment_soup(filename, v, segments):
    with open(filename, 'w') as file:
        for i in range(v.shape[0]):
            file.write(f"v {v[i, 0]} {v[i, 1]} {v[i, 2]}\n")

        for segment in segments:
            file.write(f'l {segment[0] + 1} {segment[1] + 1}\n')

def load_polyline(filename):
    '''
    Load an OBJ with polylines, convert 1 based OBJ indexing to 0 based indexing.
    '''
    v, vt, l = [], [], []
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip('\n').split()
            if tokens[0] == 'v':
                v.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                
            elif tokens[0] == 'vt':
                vt.append([int(float(tokens[1])), int(float(tokens[2]))])
                
            elif tokens[0] == 'l':
                l.append(np.array(list(map(float, tokens[1:])), dtype=np.int32))
                
    return np.array(v), np.array(vt), np.array(l) - 1

def load_polyline_with_data(filename):
    v, vn, data, l = [], [], [], []
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.strip('\n').split()
            if tokens[0] == 'v':
                v.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
             
            elif tokens[0] == 'vn':
                vn.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])

            elif tokens[0] == 'vt':
                data.append(float(tokens[1]))
                
            elif tokens[0] == 'l':
                l.append(np.array(list(map(float, tokens[1:])), dtype=np.int32))

    return np.array(v), np.array(vn), np.array(data), np.array(l) - 1

def convert_lines_to_segments(l):
    '''
    Convert a list of polylines into a list of segments (i.e. soup)
    '''
    segments = []
    for line in l:
        for i in range(len(line) - 1):
            segments.append(np.array([line[i], line[i+1]]))
    return np.array(segments)

class PolylineRegularizer:
    def __init__(self, v, l):
        self.polylines = l

        # resting length of segments
        self.rest_length = []
        for pl in self.polylines:
            self.rest_length.append(
                np.array([
                    np.linalg.norm(v[pl[i]] - v[pl[i+1]]) 
                    for i in range(pl.shape[0] - 1)
                ]))

    def bending_energy(self, v):
        energy = 0.0
        for pl in self.polylines:
            for i in range(1, pl.shape[0]-1):
                v0 = v[pl[i-1], :]
                v1 = v[pl[i]]
                v2 = v[pl[i+1], :]
                e0 = (v1 - v0)
                e1 = (v2 - v1)
                e0crosse1 = torch.linalg.cross(e0, e1)
                e0norm = torch.linalg.norm(e0)
                e1norm = torch.linalg.norm(e1)
                e0dote1 = torch.sum(e0 * e1)
                kb = (2.0 * e0crosse1) / (e0norm * e1norm + e0dote1)
                energy += torch.sum(kb * kb) / (e0norm + e1norm)
        return energy

    def stretching_energy(self, v):
        energy = 0.0
        for pl, rl in zip(self.polylines, self.rest_length):
            for i in range(0, pl.shape[0]-1):
                segment_length = torch.linalg.norm(v[pl[i]] - v[pl[i+1]])
                energy += torch.pow(segment_length - rl[i], 2.0)
        return energy
    
    def length(self, v):
        energy = 0.0
        for pl in self.polylines:
            for i in range(0, pl.shape[0]-1):
                if torch.linalg.norm(v[pl[i]] - v[pl[i+1]]) > 1e-8:
                    # only calc energy if not a degenerate vertex, i.e. length 0
                    energy += torch.pow(torch.linalg.norm(v[pl[i]] - v[pl[i+1]]), 2.0)
        return energy

    def zero_endpoint_gradients(self, v_grad):
        for pl in self.polylines:
            v_grad[pl[0], :] = 0
            v_grad[pl[-1], :] = 0

    def interpolate_gradients(self, v, v_grad, mask, is_loop = True):
        for pl in self.polylines:
            nv = pl.shape[0]
            pl_dist = np.zeros(nv)
            pl_grad = np.zeros((nv, 3))
            pl_mask = np.zeros(nv, dtype=bool)
            for i in range(nv):
                pl_dist[i] = 0 if i == 0 else pl_dist[i-1] + np.linalg.norm(v[pl[i]] - v[pl[i-1]])
                pl_grad[i] = v_grad[pl[i]]
                pl_mask[i] = 1 if not is_loop and (i == 0 or i == nv - 1) else mask[pl[i]]
            pl_grad_fn = interp1d(pl_dist[pl_mask], pl_grad[pl_mask], kind='linear', axis=0)
            use_interp = np.logical_not(pl_mask)
            pl_grad[use_interp] = pl_grad_fn(pl_dist[use_interp])
            v_grad[pl[use_interp]] = pl_grad[use_interp]

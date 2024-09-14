# Differential walk on spheres

Differential WoS is built on top of the [Zombie library](https://github.com/rohan-sawhney/zombie) primarily via modifications to the [walk on stars algorithm](zombie/include/zombie/point_estimation/walk_on_stars.h) and a [sparse data structure](zombie/include/zombie/core/value.h) for handling differential quantities. As with the original Zombie library, the codebase is CPU-based and relies on a user to define lambdas for the boundary conditions and geometric queries. We extend this design pattern by accepting a normal velocity lambda, which means the core algorithm can be applied to any parameterized geometry for which a user has defined closest point queries (i.e. `GeometricQueries::computeDistToDirichlet`) and normal velocities (i.e. `GeometricQueries::computeDirichletDisplacement`) via the [geometric queries interface](zombie/include/zombie/core/geometric_queries.h).

> [!WARNING]  
> This is an experimental codebase from our paper [Differential Walk on Spheres](https://imaging.cs.cmu.edu/differential_walk_on_spheres/). In the coming months the features in this codebase will be added to the official [Zombie codebase](https://github.com/rohan-sawhney/zombie).

## Codebase overview
This supplemental includes the following code:

* `./zombie/include/zombie` the primary CPP library containing the core differential WoS algorithms
* `./zombie/pyzombie` Python bindings and implementation of normal displacements.
* `./zombie/pywos` A GPU implementation of differential WoS for our curve inflation experiment built on [Jax](https://github.com/google/jax)
* `./opttools` A collection of helper functions we use in our experiments.

## Where to look
For more details on our implementation of differential WoS or normal displacements, we include links to the relevant files below. The PyZombie API also provides a good overview of the functionality available in our codebase.

* [core differential WoS algorithm](zombie/include/zombie/point_estimation/walk_on_stars.h)
* [normal velocity for shape from diffusion](zombie/pyzombie/include/pyzombie/mesh_geometry.h)
* [normal velocity for pose estimation](zombie/pyzombie/include/pyzombie/mesh_geometry.h)
* [normal velocity for optimization-driven thermal design](zombie/pyzombie/include/pyzombie/toast_scene.h)
* [normal velocity for Bezier](zombie/pyzombie/include/pyzombie/bezier_scene.h)
* [normal velocity for implicit](pywos/walk.py)
* [Pyzombie API](zombie/pyzombie/api.cpp.h)

## Running example optimization
First create a virtual environment and install all requirements.
```
python3 -m venv ./diffwos
source ./diffwos/bin/activate
pip3 install -r requirements.txt
```

Next, compile the Zombie library and build the Python bindings. 

```
cd zombie && mkdir build
cd build && cmake ..
make -j4
```
Finally, you can run a 2D reconstruction example using the python bindings for Zombie
```
cd ../../zombie_example
python3 optimize.py
```
The example will create a directory with progressive views of the solution corresponding to the current optimized boundary and the reference solution (i.e. a horizontal stack of solutions). Once the optimization is complete, you can compare the `initial.obj` and `optimized.obj` polylines using a mesh viewing library such as [MeshLab](https://www.meshlab.net/).

Our current implementation relies on some syntactic sugar in Python to convert sparse dictionaries of derivatives into dense tensors for PyTorch. PyZombie relies on user defined maps from derivative keys (e.g. `key='d.23.0'`) to a derivative tensor index (e.g. `idx=48`) to densify derivatives. In the example optimization, for instance, we convert strings indicating the vertex index and coordinate offset into a 1D tensor as follows

```python
def get_param_idx(key):
    primitive_id, local_vertex_id, coord = map(int, key.split(".")[1:])
    vertex_id = curr_l[primitive_id, local_vertex_id]
    return 2 * vertex_id + coord
    
grad = L.differential.filter("d").dense(curr_v.shape[0] * curr_v.shape[1], get_param_idx)
```

### Troubleshooting:
The `optimize.py` file uses relative paths to the PyZombie library, so ensure that you're calling from within the `zombie_example` directory or you may get a `ModuleNotFoundError`
```
Traceback (most recent call last):
  File "/Users/user/supplement/zombie_example/optimize.py", line 14, in <module>
    import pyzombie as pz
ModuleNotFoundError: No module named 'pyzombie'
```
Note that whichever virtual environment is active during library compilation should also be active when running the code.

## Extra features
Only the features introduced in our paper have been tested, but our implementation also supports derivatives with respect to parameterized source terms and Neumann boundary conditions. The codebase also supports derivatives with respect to gradients, although this code has not been thoroughly tested.

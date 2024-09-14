#pragma once

#include "pyzombie/common.h"
namespace py = pybind11;

namespace pyzombie {

template<typename T>
zombie::Value<T, 2> interpolate2D(const Vector2 &pt, 
							      const Vector2 &scale,
							      const std::array<int, 2> &shape, 
							      const std::vector<T> &data,
								  T initVal,
								  std::string id,
							      bool differentialEnabled) {
	int res_x = shape[0];
    int res_y = shape[1];

	float x = pt.x() * (res_x - 1);
    float y = pt.y() * (res_y - 1);

	int x1 = static_cast<int>(floor(x));
	int x2 = static_cast<int>(ceil(x));
	int y1 = static_cast<int>(floor(y));
	int y2 = static_cast<int>(ceil(y));

	if (x1 < 0 || x2 >= res_x || y1 < 0 || y2 >= res_y) return zombie::Value<T, 2>(initVal);

	float dx = x - x1;
	float dy = y - y1;

	int idx0 = x1 + y1 * res_x;
	int idx1 = x2 + y1 * res_x;
	int idx2 = x1 + y2 * res_x;
	int idx3 = x2 + y2 * res_x;

	T v0 = data[idx0];
	T v1 = data[idx1];
	T v2 = data[idx2];
	T v3 = data[idx3];
	
	zombie::Value<T, 2> value(initVal);

	// value
    T c00 = v0 * (1 - dx) + v1 * dx;
    T c01 = v2 * (1 - dx) + v3 * dx;
	value.data = c00 * (1 - dy) + c01 * dy;

	// gradient
	std::array<T, 2> gradient = {
		T(((v1 - v0) * (1.0 - dy) + (v3 - v2) * dy) * (res_x - 1) / scale[0]),
		T(((v2 - v0) * (1.0 - dx) + (v3 - v1) * dx) * (res_y - 1) / scale[1])
	};
	value.gradient = zombie::SpatialGradient<T, 2>(gradient.data());

	// differential
	if (differentialEnabled) {
		value.differential.ref(id + "." + std::to_string(idx0)) = (1 - dx) * (1 - dy);
		value.differential.ref(id + "." + std::to_string(idx1)) = dx * (1 - dy);
		value.differential.ref(id + "." + std::to_string(idx2)) = (1 - dx) * dy;
		value.differential.ref(id + "." + std::to_string(idx3)) = dx * dy;
	}

	return value;
}
	
template<typename T>
zombie::Value<T, 3> interpolate3D(const Vector3 &pt, 
								  const Vector3 &scale,
							      const std::array<int, 3> &shape, 
							      const std::vector<T> &data,
								  T initVal,
								  std::string id,
							      bool differentialEnabled) {
	int res_x = shape[0];
    int res_y = shape[1];
	int res_z = shape[2];

	float x = pt.x() * (res_x - 1);
    float y = pt.y() * (res_y - 1);
	float z = pt.z() * (res_z - 1);

	int x1 = static_cast<int>(floor(x));
	int x2 = static_cast<int>(ceil(x));
	int y1 = static_cast<int>(floor(y));
	int y2 = static_cast<int>(ceil(y));
	int z1 = static_cast<int>(floor(z));
	int z2 = static_cast<int>(ceil(z));

	if (x1 < 0 || x2 >= res_x ||
		y1 < 0 || y2 >= res_y ||
		z1 < 0 || z2 >= res_y) return zombie::Value<T, 3>(initVal);

	float dx = x - x1;
	float dy = y - y1;
	float dz = z - z1;

	int idx0 = 	x1 + y1 * res_x + z1 * res_x * res_y;
	int idx1 = x2 + y1 * res_x + z1 * res_x * res_y;
	int idx2 = x1 + y1 * res_x + z2 * res_x * res_y;
	int idx3 = x2 + y1 * res_x + z2 * res_x * res_y;
	int idx4 = x1 + y2 * res_x + z1 * res_x * res_y;
	int idx5 = x2 + y2 * res_x + z1 * res_x * res_y;
	int idx6 = x1 + y2 * res_x + z2 * res_x * res_y;
	int idx7 = x2 + y2 * res_x + z2 * res_x * res_y;

	T v0 = data[idx0];
	T v1 = data[idx1];
	T v2 = data[idx2];
	T v3 = data[idx3];
	T v4 = data[idx4];
	T v5 = data[idx5];
	T v6 = data[idx6];
	T v7 = data[idx7];

	T c00 = v0 * (1 - dx) + v1 * dx;
    T c01 = v2 * (1 - dx) + v3 * dx;
    T c10 = v4 * (1 - dx) + v5 * dx;
    T c11 = v6 * (1 - dx) + v7 * dx;

	zombie::Value<T, 3> value(initVal);
	
	// value
	T c0 = c00 * (1 - dy) + c10 * dy;
    T c1 = c01 * (1 - dy) + c11 * dy;
	value.data = c0 * (1 - dz) + c1 * dz;	

	// gradient
	std::array<T, 3> gradient = {
		T(((v1 - v0) * (1.0 - dy) + (v5 - v4) * dy + 
		 (v3 - v2) * (1.0 - dz) + (v7 - v6) * dz) * (res_x - 1) / scale[0]),
		T(((v4 - v0) * (1.0 - dx) +(v5 - v1) * dx +
         (v6 - v2) * (1.0 - dz) + (v7 - v3) * dz) * (res_y - 1) / scale[1]),
		T(((v2 - v0) * (1.0 - dx) + (v3 - v1) * dx +
         (v6 - v4) * (1.0 - dy) + (v7 - v5) * dy) * (res_z - 1) / scale[2])
	};
	value.gradient = zombie::SpatialGradient<T, 3>(gradient.data());

	// differential
	if (differentialEnabled) {
		value.differential.ref(id + "." + std::to_string(idx0)) = (1 - dx) * (1 - dy) * (1 - dz);
		value.differential.ref(id + "." + std::to_string(idx1)) = dx * (1 - dy) * (1 - dz);
		value.differential.ref(id + "." + std::to_string(idx2)) = (1 - dx) * (1 - dy) * dz;
		value.differential.ref(id + "." + std::to_string(idx3)) = dx * (1 - dy) * (1 - dz);
		value.differential.ref(id + "." + std::to_string(idx4)) = (1 - dx) * dy * (1 - dz);
		value.differential.ref(id + "." + std::to_string(idx5)) = dx * dy * (1 - dz);
		value.differential.ref(id + "." + std::to_string(idx6)) = (1 - dx) * dy * dz;
		value.differential.ref(id + "." + std::to_string(idx7)) = dx * dy * dz;
	}

	return value;
};

template <typename T, int DIM>
struct Grid {
	std::array<int, DIM> shape;
	std::vector<T> data;
	Vector<DIM> origin = Vector<DIM>::Zero();
	Vector<DIM> scale = Vector<DIM>::Ones();
	bool differentialEnabled = true;
	std::string id; 

	Grid(Vector<DIM> origin, Vector<DIM> scale, 
		 std::string id, bool differentialEnabled): 
		 origin(origin), 
		 scale(scale), 
		 id(id),
		 differentialEnabled(differentialEnabled) {};

	virtual void setFromFunction(const VectorI<DIM> &dims, 
	 							 const std::function<T(Vector<DIM>)> &f) {
		throw std::runtime_error("setFromFunction not implemented.");
	}

	virtual void setFromArray(const py::array_t<float> &values) {
		throw std::runtime_error("setFromArray not implemented by default.");
	}

	virtual py::array_t<float> getArray() const {
		throw std::runtime_error("getArray not implemented.");
	}
	
	zombie::Value<T, DIM> operator()(const Vector<DIM> &x) const {
		return this->interpolate(x);
	};

	virtual zombie::Value<T, DIM> interpolate(const Vector<DIM> &x) const {
		return 0.0f;
	};

	virtual Vector<DIM> toLocal(const Vector<DIM> &x) const {
		return (x - origin).array() / scale.array();
	}
};

template<>
void Grid<float, 2>::setFromFunction(const VectorI<2> &dims, 
	 							     const std::function<float(Vector<2>)> &f) {
	shape[0] = dims[0];
	shape[1] = dims[1];
	data.resize(shape[0] * shape[1]);
	if (data.size() == 0) {
		return;
	}
	for (int i = 0; i < shape[0]; i++)
	for (int j = 0; j < shape[1]; j++) {
		Vector<2> p(i * scale[0] / shape[0] + origin[0],
			        j * scale[1] / shape[1] + origin[1]);
		data[i + j * shape[0]] = f(p);
	}
}

template<>
void Grid<float, 3>::setFromFunction(const VectorI<3> &dims, 
	 							     const std::function<float(Vector<3>)> &f) {
	int nGridPts = 1;
	for (int i = 0; i < 3; i++) {
		shape[i] = dims[i];
		nGridPts *= dims[i];
	}

	data.resize(nGridPts);
	if (data.size() == 0)
		return;

	for (int i = 0; i < shape[0]; i++)
	for (int j = 0; j < shape[1]; j++) 
	for (int k = 0; k < shape[2]; k++) {
		Vector<3> p(i * scale[0] / shape[0] + origin[0],
			        j * scale[1] / shape[1] + origin[1],
					k * scale[2] / shape[2] + origin[2]);
		data[i + j * shape[0] + k * shape[0] * shape[1]] = f(p);
	}
}

template <>
void Grid<float, 2>::setFromArray(const py::array_t<float> &values) {
	py::buffer_info valuesBufferInfo = values.request();

	if (valuesBufferInfo.ndim != 2) {
		throw std::runtime_error("Values should have dimensions N_i^2");
	}
	
	for (int i = 0; i < 2; i++)
		shape[i] = valuesBufferInfo.shape[i];	

	float* valuesBuffer = static_cast<float*>(valuesBufferInfo.ptr);
	data.resize(valuesBufferInfo.size);	
	for (int i = 0; i < data.size(); i++)
		data[i] = valuesBuffer[i];
}

template<>
void Grid<float, 3>::setFromArray(const py::array_t<float> &values) {
	py::buffer_info valuesBufferInfo = values.request();

	if (valuesBufferInfo.ndim != 3) {
		throw std::runtime_error("Values should have dimensions N_i^3");
	}
	
	for (int i = 0; i < 3; i++)
		shape[i] = valuesBufferInfo.shape[i];	

	float* valuesBuffer = static_cast<float*>(valuesBufferInfo.ptr);
	data.resize(valuesBufferInfo.size);	
	for (int i = 0; i < data.size(); i++)
		data[i] = valuesBuffer[i];
}

template<>
py::array_t<float> Grid<float, 2>::getArray() const {
	py::array_t<float> result(data.size(), (float*)data.data(), py::capsule(data.data(), [](void *data) {}));
	result.resize({shape[0], shape[1]});
	return result;
}

template<>
py::array_t<float> Grid<float, 3>::getArray() const {
	py::array_t<float> result(data.size(), (float*)data.data(), py::capsule(data.data(), [](void *data) {}));
	result.resize({shape[0], shape[1], shape[2]});
	return result;
}

template<>
zombie::Value<float, 2> Grid<float, 2>::interpolate(const Vector2 &globalPt) const {
	if (data.size() == 0) return 0.0f;
	Vector2 pt = toLocal(globalPt);
	return interpolate2D<float>(pt, scale, shape, data, 0.0f, id, differentialEnabled);
};

template<>
zombie::Value<float, 3> Grid<float, 3>::interpolate(const Vector3 &globalPt) const {
	if (data.size() == 0) return 0.0f;
	Vector3 pt = toLocal(globalPt);
	return interpolate3D<float>(pt, scale, shape, data, 0.0f, id, differentialEnabled);
}

}; //pyzombie
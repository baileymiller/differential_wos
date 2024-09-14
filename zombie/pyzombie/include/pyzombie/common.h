#pragma once

#include <stdexcept>
#include <sstream>
#include <exception>
#include <iostream>
#include <vector>
#include <array>
#include <vector>

#include <fcpw/fcpw.h>

#include <zombie/core/value.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <zombie/point_estimation/walk_on_stars.h>
#include <zombie/core/pde.h>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace pyzombie {

template<size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector1 = Vector<1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;
using Vector4 = Vector<4>;
using VectorXi = Eigen::Matrix<float, Eigen::Dynamic, 1>;

template<int DIM>
using Array = Eigen::Array<float, DIM, 1>;
using Array2 = Array<2>;
using RGB = Array<3>;

template<size_t DIM>
using VectorI = Eigen::Matrix<int, DIM, 1>;
using VectorI2 = VectorI<2>;
using VectorI3 = VectorI<3>;
using VectorI4 = VectorI<4>;

template<size_t DIM>
using Transform = Eigen::Matrix<float, DIM + 1, DIM + 1>;

template <typename T>
inline T lerp(float s, T A, T B) {
    return (1.0 - s) * A + s * B;
}

template<size_t DIM>
Vector<DIM + 1> homogenize(const Vector<DIM>& vec) {
    Vector<DIM + 1> homogenizedVec;
    for (size_t i = 0; i < DIM; ++i)
        homogenizedVec(i) = vec(i);
    homogenizedVec(DIM) = 1.0f;
    return homogenizedVec;
}

template<size_t DIM>
Vector<DIM> dehomogenize(const Vector<DIM + 1>& homogenizedVec) {
    Vector<DIM> vec;
    for (size_t i = 0; i < DIM ; ++i)
        vec(i) = homogenizedVec(i) / homogenizedVec(DIM);
    return vec;
}

template <typename T>
struct VectorWrapper {
    std::vector<T> data;

    VectorWrapper() {}
    
    void push_back(const T& value) { data.push_back(value); }

	int size() const { return data.size(); }

    const T& val(int i) const { return data[i]; }

    T& operator[](int i) { return data[i]; }
};

template <typename T, int DIM>
struct BoundarySamplePoint {
    Vector<DIM> pt;
    Vector<DIM> n;
    zombie::Differential<float> vn;
    T u, uExterior;
    float pdf;
};

template<int DIM>
Vector2 barycentricCoordinates(const Vector<DIM> &p,
							   const Vector<DIM> &pa,
							   const Vector<DIM> &pb,
							   const Vector<DIM> &pc) {
	Vector<DIM> v0 = pb - pa;
	Vector<DIM> v1 = pc - pa;
	Vector<DIM> v2 = p - pa;
	float d00 = v0.dot(v0);
	float d01 = v0.dot(v1);
	float d11 = v1.dot(v1);
	float d20 = v2.dot(v0);
	float d21 = v2.dot(v1);
	float denom = d00*d11 - d01*d01;
	float v = (d11*d20 - d01*d21) / denom;
	float w = (d00*d21 - d01*d20) / denom;
	float u = 1.0f - v - w;
	return Vector2(u, v);
}

}; // pyzombie

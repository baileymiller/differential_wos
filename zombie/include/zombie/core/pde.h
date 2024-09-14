#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <zombie/core/value.h>

namespace zombie {

template<int DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <typename T, int DIM>
struct PDE {
	// constructor
	PDE(): absorption(0.0f), dirichlet({}), neumann({}), source({}),
		   dirichletDoubleSided({}), neumannDoubleSided({}) {}

	// members
	float absorption;
	std::function<Value<T, DIM>(const Vector<DIM>&)> dirichlet;
	std::function<Value<T, DIM>(const Vector<DIM>&)> neumann;
	std::function<Value<T, DIM>(const Vector<DIM>&)> source;
	std::function<Value<T, DIM>(const Vector<DIM>&, bool)> dirichletDoubleSided;
	std::function<Value<T, DIM>(const Vector<DIM>&, bool)> neumannDoubleSided;
};

} // zombie

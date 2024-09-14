#pragma once

#include <cmath>
#include "pyzombie/common.h"

namespace pyzombie {

zombie::Value<float, 3> dirichletA(const Vector3& p) {
	zombie::Value<float, 3> value(0.0f);
	value.data = p[0] * p[1] * p[2];
	std::array<float, 3> gradient = { p[1] * p[2], p[0] * p[2], p[0] * p[1] };
	value.gradient = zombie::SpatialGradient<float, 3>(gradient.data());
	return value;
}

zombie::Value<float, 3> dirichletB(const Vector3& p) {
	zombie::Value<float, 3> value(0.0f);
	value.data = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
	std::array<float, 3> gradient = { 2 * p[0], 2 * p[1], 2 * p[2] };
	value.gradient = zombie::SpatialGradient<float, 3>(gradient.data());
	return value;
}

};

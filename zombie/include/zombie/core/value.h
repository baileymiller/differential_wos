#pragma once

#include <map>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

namespace zombie {

template<int DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <typename T, int DIM>
struct SpatialGradient {
	std::array<T, DIM> data;

	SpatialGradient() {}
	
	SpatialGradient(T value) {
		for (int i = 0; i < DIM; i++) 
			data[i] = value;
	}
	
	explicit SpatialGradient(const T* value) {
		for (int i = 0; i < DIM; i++) 
			data[i] = value[i];
	}
	
	explicit SpatialGradient(const T &magnitude, const Vector<DIM> &direction) {
		for (int i = 0; i < DIM; i++)
			data[i] = magnitude * direction[i];
	}

	explicit SpatialGradient(const std::array<T, DIM> &data): data(data) {}
	
	T dot(const SpatialGradient<T, DIM>& other) const {
		T result = data[0] * other.data[0];
		for (int i = 1; i < DIM; i++) 
			result += data[i] * other.data[i];
		return result;
	}
	
	T dot(const Vector<DIM>& vec) const {
		T result = data[0] * vec[0];
		for (int i = 1; i < DIM; i++) 
			result += data[i] * vec[i];
		return result;
	}

	T& operator[](int i) {
		return data[i];
	}
};

template<typename T>
struct Differential {
	/* 
		For efficiency we assume that initVal is a zero element
		
			val * initVal = initVal
			val + initVal = initVal
		
		This allow us to quickly add or subtract two differentials since we only have to 
		iterate over a single set of keys. Using any other initVal could lead to incorrect behavior. 
		
		We ensure that any modification to the differential cannot alter this initVal.
		As a result, we allow multiplication/division by a scalar (Differential<float> * float)
		but not addition/subtraction with a scalar (Differential<float> + float)
	*/
	T initVal;
	std::map<std::string, T> data;

	Differential() {}

	explicit Differential(T initVal): initVal(initVal) {}
	
	Differential(const Differential<T> &other): 
                 initVal(other.initVal),
                 data(other.data) {};
    
    Differential(const Differential<T> *other): 
                 initVal(other->initVal),
                 data(other->data) {};

	Differential<T> filter(std::string id) const {
		Differential<T> filteredDifferential(initVal);
		for (const auto&[key, value]: data) {
        	size_t dotPos = key.find('.');
        	std::string startStr = key.substr(0, dotPos);
        	if (startStr == id) {
            	filteredDifferential.ref(key) = value;
        	}
		}
		return filteredDifferential;
	}

	std::vector<T> dense(int size, std::function<int(std::string)> getIndex) const {
		std::vector<T> dense(size, initVal);
		for (const auto&[key, value]: data)
			dense[getIndex(key)] = value;
		return dense;
	}

	bool has(const std::string& param) {
		return data.find(param) != data.end();
	}

	T val(const std::string& param) const {
		const auto it = data.find(param);
        return it != data.end() ? it->second : initVal;
    }
	
	T& ref(const std::string& param) {
		auto [it, inserted] = data.insert(std::pair<std::string, T>(param, initVal));
    	return it->second;
    }
};

template <typename T, int DIM>
struct Value {
	T data;								// f(x, \theta)
	SpatialGradient<T, DIM> gradient; 	// \grad_x f(x, \theta) 	spatial gradient
	Differential<T> differential;		// d f(x,\theta)/d\theta 	differential with respect to arbitrary parameters	
	
	Value(T initVal): 
		  data(initVal),
		  gradient(initVal),
		  differential(initVal) {}

	Value(T data, 
	      SpatialGradient<T, DIM> gradient, 
		  Differential<T> differential):
		  data(data),
		  gradient(gradient),
		  differential(differential) {}

	Value(Value<T, DIM> *value): 
		  data(value->data),
		  gradient(value->gradient),
		  differential(value->differential) {}

	operator T() const {
		return data;
	}

	operator SpatialGradient<T, DIM>() const {
		return gradient;
	}

	operator Differential<T>() const {
		return differential;
	}
};

template<>
SpatialGradient<float, 2>::SpatialGradient() {
	for (int i = 0; i < 2; i++) 
		data[i] = 0.0f;
}

template<>
SpatialGradient<float, 3>::SpatialGradient() {
	for (int i = 0; i < 3; i++) 
		data[i] = 0.0f;
}

template <typename T, int DIM>
SpatialGradient<T, DIM>& operator+=(SpatialGradient<T, DIM>& lhs, const SpatialGradient<T, DIM> &rhs) {
	for (int i = 0; i < DIM; i++)
		lhs.data[i] += rhs.data[i];
	return lhs;
}

template <typename T, int DIM>
SpatialGradient<T, DIM>& operator-=(SpatialGradient<T, DIM>& lhs, const SpatialGradient<T, DIM> &rhs) {
	for (int i = 0; i < DIM; i++)
		lhs.data[i] -= rhs.data[i];
	return lhs;
}

template <typename T, int DIM>
SpatialGradient<T, DIM>& operator/=(SpatialGradient<T, DIM>& lhs, const float &rhs) {
	for (int i = 0; i < DIM; i++)
		lhs.data[i] -= rhs;
	return lhs;
}

template <typename T, int DIM>
SpatialGradient<T, DIM>& operator*=(SpatialGradient<T, DIM>& lhs, const float &rhs) {
	for (int i = 0; i < DIM; i++)
		lhs.data[i] *= rhs;
	return lhs;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator+(const SpatialGradient<T, DIM> &lhs, const SpatialGradient<T, DIM> &rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] + rhs.data[i];
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator-(const SpatialGradient<T, DIM> &lhs, const SpatialGradient<T, DIM> &rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] - rhs.data[i];
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator*(const SpatialGradient<T, DIM>& lhs, const float &rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] * rhs;
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator*(const float &lhs, const SpatialGradient<T, DIM>& rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs * rhs.data[i];
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator*(const SpatialGradient<T, DIM>& lhs, const Vector<DIM>& rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] * rhs[i];
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator*(const SpatialGradient<T, DIM>& lhs, const SpatialGradient<T, DIM>& rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] * rhs.data[i];
	return result;
}

template<typename T, int DIM>	
SpatialGradient<T, DIM> operator/(const SpatialGradient<T, DIM>& lhs, const float &rhs) {
	SpatialGradient<T, DIM> result;
	for (int i = 0; i < DIM; i++)
		result.data[i] = lhs.data[i] / rhs;
	return result;
}

template<>
Differential<float>::Differential(): initVal(0.0f) {}

template<typename T>
Differential<T> operator+(const Differential<T>& lhs, const Differential<T>& rhs) {
	Differential<T> result(lhs);
	for (const auto& [key, value] : rhs.data)
		result.ref(key) += value;
	return result;
}

template<typename T>	
Differential<T>& operator+=(Differential<T> &lhs, const Differential<T> &rhs) {
	for (const auto& [key, value] : rhs.data)
		lhs.ref(key) += value;
	return lhs;
}

template<typename T>	
Differential<T>& operator-=(Differential<T> &lhs, const Differential<T> &rhs) {
	for (const auto& [key, value] : rhs.data)
		lhs.ref(key) -= value;
	return lhs;
}

template<typename T>
Differential<T> operator-(const Differential<T>& lhs, const Differential<T>& rhs) {
	Differential<T> result(lhs);
	for (const auto& [key, value] : rhs.data)
		result.ref(key) -= value;
	return result;
}

template<typename T>
Differential<T> operator/(const Differential<T>& lhs, const float& rhs) {
	Differential<T> result(lhs.initVal);
	for (const auto& [key, value] : lhs.data)
		result.ref(key) = value / rhs;
	return result;
}

template<typename T>
Differential<T>& operator/=(Differential<T> &lhs, const float& rhs) {
	for (auto& [key, value] : lhs.data)
		value /= rhs;
	return lhs;
}
template<typename T>
Differential<T>& operator*=(Differential<T> &lhs, const float& rhs) {
	for (auto& [key, value] : lhs.data)
		value *= rhs;
	return lhs;
}

template<typename T>
Differential<T> operator*(const Differential<T>& lhs, const float& rhs) {
	Differential<T> result(lhs.initVal);
	for (const auto& [key, value] : lhs.data)
		result.ref(key) = value * rhs;
	return result;
}
template<typename T>
Differential<T> operator*(const float& lhs, const Differential<T>& rhs) {
	Differential<T> result(rhs.initVal);
	for (const auto& [key, value] : rhs.data)
		result.ref(key) = value * lhs;
	return result;
}

template<typename T>
typename std::enable_if<!std::is_same<T, float>::value, Differential<T>>::type 
operator*(const Differential<float>& lhs, const T& rhs) {
	Differential<T> result(lhs.initVal * rhs);
	for (const auto& [key, value] : lhs.data)
		result.ref(key) = value * rhs;
	return result;
}
template<typename T>
typename std::enable_if<!std::is_same<T, float>::value, Differential<T>>::type 
operator*(const T& lhs, const Differential<float>& rhs) {
	Differential<T> result(lhs * rhs.initVal);
	for (const auto& [key, value] : rhs.data)
		result.ref(key) = value * lhs;
	return result;
}

template<typename T>
typename std::enable_if<!std::is_same<T, float>::value, Differential<T>>::type 
operator*(const Differential<T>& lhs, const T& rhs) {
	Differential<T> result(lhs.initVal);
	for (const auto& [key, value] : lhs.data)
		result.ref(key) = value * rhs;
	return result;
}
template<typename T>
typename std::enable_if<!std::is_same<T, float>::value, Differential<T>>::type 
operator*(const T& lhs, const Differential<T>& rhs) {
	Differential<T> result(rhs.initVal);
	for (const auto& [key, value] : rhs.data)
		result.ref(key) = value * lhs;
	return result;
}

template<typename T, int DIM>
typename std::enable_if<!std::is_same<T, float>::value, Differential<SpatialGradient<T, DIM>>>::type 
operator*(const Differential<T> &lhs, const Vector<DIM> &rhs) {
	Differential<SpatialGradient<T, DIM>> result(SpatialGradient<T, DIM>(lhs.initVal));
	for (const auto& [key, value] : lhs.data)
		result.ref(key) = SpatialGradient<T, DIM>(value) * rhs;
	return result;
}

template<int DIM>
Differential<float> dot(const Differential<Vector<DIM>>& differential, const Vector<DIM> &v) {
	Differential<float> result(0.0f);	
	for (const auto&[key, value]: differential.data) {
		result.ref(key) = value.dot(v);
	}
	return result;
}

template<typename T, int DIM>
Value<T, DIM> operator+(const Value<T, DIM>&lhs, const Value<T, DIM> &rhs) {
	return Value<T, DIM>(
		lhs.data + rhs.data,
		lhs.gradient + rhs.gradient,
		lhs.differential + rhs.differential
	);
}

template<typename T, int DIM>
Value<T, DIM> operator-(const Value<T, DIM>&lhs, const Value<T, DIM> &rhs) {
	return Value<T, DIM>(
		lhs.data - rhs.data,
		lhs.gradient - rhs.gradient,
		lhs.differential - rhs.differential
	);
}

template<typename T, int DIM>
Value<T, DIM> operator*(const Value<T, DIM>&lhs, const float &rhs) {
	return Value<T, DIM>(
		lhs.data * rhs,
		lhs.gradient * rhs,
		lhs.differential * rhs
	);
}

template<typename T, int DIM>
Value<SpatialGradient<T, DIM>, DIM> operator*(const Value<T, DIM>&lhs, const Vector<DIM> &rhs) {
	// new gradient data term
	SpatialGradient<T, DIM> data = SpatialGradient<T, DIM>(lhs.data) * rhs;

	// empty "gradient" of gradient (does not support higher order gradients)
	SpatialGradient<SpatialGradient<T, DIM>, DIM> gradient;

	// expand differential values to gradients
	SpatialGradient<T, DIM> initVal(lhs.differential.initVal);
	Differential<SpatialGradient<T, DIM>> differential(initVal);
	for (const auto&[key, value]: lhs.differential.data) {
		differential.ref(key) = SpatialGradient<T, DIM>(value) * rhs;
	}

	return Value<SpatialGradient<T, DIM>, DIM>(data, gradient, differential);
}

template<typename T, int DIM>
Value<T, DIM> operator*(const float &lhs, const Value<T, DIM>&rhs) {
	return Value<T, DIM>(
		lhs * rhs.data,
		lhs * rhs.gradient,
		lhs * rhs.differential
	);
}

template<typename T, int DIM>
Value<T, DIM> operator/(const Value<T, DIM>&lhs, const float &rhs) {
	return Value<T, DIM>(
		lhs.data / rhs,
		lhs.gradient / rhs,
		lhs.differential / rhs
	);
}

template<typename T, int DIM>
Value<T, DIM>& operator+=(Value<T, DIM> &lhs, const Value<T, DIM>& rhs) {
	lhs.data += rhs.data;
	lhs.gradient += rhs.gradient;
	lhs.differential += rhs.differential;
	return lhs;
}

template<typename T, int DIM>
Value<T, DIM>& operator*=(Value<T, DIM> &lhs, const float& rhs) {
	lhs.data *= rhs;
	lhs.gradient *= rhs;
	lhs.differential *= rhs;
	return lhs;
}

}; //zombie

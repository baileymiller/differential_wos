#pragma once

#include "pyzombie/common.h"
#include <queue>

#define EPSILON 1e-4
#define SQRT3 1.7320508075688772935274463415058723f

namespace pyzombie {

inline int computeQuadraticRoots(float a, float b, float c,
								 float *roots, bool *isValid)
{
	float disc = b*b - 4.0f*a*c;
	if (std::fabs(disc) < EPSILON) {
		// equation has a single real root
		roots[0] = -b/(2.0f*a);
		isValid[0] = true;
		isValid[1] = false;

		return 1;

	} else if (disc > 0) {
		// equation has two real roots
		float det = std::sqrt(disc);
		roots[0] = (-b + det)/(2.0f*a);
		roots[1] = (-b - det)/(2.0f*a);
		isValid[0] = true;
		isValid[1] = true;

		return 2;
	}

	// equation has no real roots
	isValid[0] = false;
	isValid[1] = false;

	return 0;
}

inline int computeCubicRoots(float a, float b, float c, float d,
							 float *roots, bool *isValid)
{
	// initialize
	for (int i = 0; i < 3; i++) isValid[i] = false;

	// check edge cases
	bool aIsZero = std::fabs(a) < EPSILON;
	bool bIsZero = std::fabs(b) < EPSILON;
	bool cIsZero = std::fabs(c) < EPSILON;

	if (aIsZero && bIsZero && cIsZero) {
		// equation is constant and has no roots
		return 0;

	} else if (aIsZero && bIsZero) {
		// equation is linear and has 1 real root
		roots[0] = -d/c;
		isValid[0] = true;

		return 1;

	} else if (aIsZero) {
		// equation is quadratic and has <= 2 real roots
		return computeQuadraticRoots(b, c, d, roots, isValid);
	}

	// equation is cubic and has <= 3 real roots
	b /= a;
	c /= a;
	d /= a;
	float p = -b/3.0f;
	float q = p*p*p + (b*c - 3.0f*d)/6.0f;
	float r = c/3.0f - p*p;
	float disc = q*q + r*r*r;

	if (disc >= 0.0f) {
		// equation has either 1 or 2 real roots
		float det = std::sqrt(disc);
		float qpPlus = q + det;
		float qpMinus = q - det;
		float s = (qpPlus > 0 ? 1.0f : -1.0f)*std::cbrt(std::fabs(qpPlus));
		float t = (qpMinus > 0 ? 1.0f : -1.0f)*std::cbrt(std::fabs(qpMinus));
		float im = std::fabs(SQRT3*(s - t)/2.0f);

		roots[0] = s + t + p;
		isValid[0] = true;

		if (im < EPSILON) {
			roots[1] = p - (s + t)/2.0f;
			isValid[1] = true;

			return 2;
		}

		return 1;
	}

	// equation has 3 real roots
	float sqrtR = std::sqrt(-r);
	float theta = std::acos(q/(sqrtR*sqrtR*sqrtR));
	float cosTheta3 = std::cos(theta/3.0f);
	float sinTheta3 = std::sin(theta/3.0f);

	roots[0] = 2.0f*sqrtR*cosTheta3 + p;
	roots[1] = -sqrtR*(cosTheta3 + SQRT3*sinTheta3) + p;
	roots[2] = -sqrtR*(cosTheta3 - SQRT3*sinTheta3) + p;
	isValid[0] = true;
	isValid[1] = true;
	isValid[2] = true;

	return 3;
}

template <int DEG>
class PolynomialRootFinder {
public:
	// constructor
	PolynomialRootFinder(float tau_): tau(tau_), nIntervals(0) {
		if (DEG <= 3) {
			throw std::runtime_error("PolynomialRootFinder(): roots can be computed analytically for degree: " + std::to_string(DEG));
		}
	}

	// find the roots of a polynomial
	void findRoots(float tStart, float tEnd, bool ignoreMin, bool ignoreMax,
				   float *roots, bool *isValid) {
		// find intervals containing a single root
		findRootIntervals(tStart, tEnd);

		// compute the roots
		computeRoots(ignoreMin, ignoreMax, roots, isValid);
	}

	// returns the requested coefficient of the polynomial
	float& getCoefficient(int i) {
		return sturmChain[0][i];
	}

private:
	// evaluates the polynomial
	float evaluatePolynomial(float t, int n=0) {
		// evaluate using Horner's method
		float v = 0.0f;
		for (int i = 0; i < DEG + 1 - n; i++) {
			v *= t;
			v += sturmChain[n][i];
		}

		return v;
	}

	// sets pc to the remainder of the division of pa by pb
	void computeRemainder(const float *pa, const float *pb, float *pc, int n) {
		float pb0 = pb[0];
		if (std::fabs(pb0) > EPSILON) {
			float T = pa[0]/pb0;
			float M = (pa[1] - T*pb[1])/pb0;

			for (int i = 0; i < n; i++) {
				pc[i] = M*pb[i + 1] - pa[i + 2];
				if (i + 1 < n) pc[i] += T*pb[i + 2];
			}

		} else {
			// compute remainder via long division
			int nPa = n + 2;
			int nPb = n + 1;
			int qDegree = 1;

			// count the degree of the quotient
			for (int i = 0; i < n + 1; i++) {
				if (std::fabs(pb[i]) < EPSILON) {
					qDegree++;
					nPb--;

				} else {
					break;
				}
			}

			if (qDegree == nPa) {
				// if pb contains only zeros, set the remainder to 0
				for (int i = 0; i < n; i++) pc[i] = 0.0f;

			} else {
				// copy pa
				float remainder[DEG + 1];
				for (int i = 0; i < nPa; i++) remainder[i] = pa[i];

				// compute remainder
				for (int i = 0; i <= qDegree; i++) {
					float qi = remainder[i]/pb[0];

					for (int j = 0; j < nPb; j++) {
						remainder[i + j] -= qi*pb[j];
					}
				}

				// set pc
				for (int i = 0; i < n; i++) pc[i] = -remainder[i + 2];
			}
		}
	}

	// fills sturm chain entries
	void updateSturmChain() {
		// set the 1st entry of the sturm chain to the derivative of the 0th entry
		for (int i = 0; i < DEG; i++) {
			sturmChain[1][i] = (DEG - i)*sturmChain[0][i];
		}

		// set the remaining entries of the sturm chain
		for (int i = 2; i <= DEG; i++) {
			computeRemainder(sturmChain[i - 2], sturmChain[i - 1], sturmChain[i], DEG + 1 - i);
		}
	}

	// counts the number of times the sign of consecutive polynomials in the
	// sturm chain is different
	int countSturmSignChanges(float t) {
		float p0 = evaluatePolynomial(t);
		bool isPositive = p0 > 0;
		int count = 0;

		for (int i = 1; i <= DEG; i++) {
			float pi = evaluatePolynomial(t, i);

			if (isPositive && pi < 0) {
				isPositive = false;
				count++;

			} else if (!isPositive && pi > 0) {
				isPositive = true;
				count++;
			}
		}

		return count;
	}

	// finds intervals containing a single root; TODO: optimize
	void findRootIntervals(float tStart, float tEnd) {
		// update the sturm chain using the values of the coefficients
		updateSturmChain();

		// find intervals
		nIntervals = 0;
		std::queue<Vector2> intervalQueue;
		intervalQueue.emplace(Vector2(tStart, tEnd));

		while (!intervalQueue.empty() && nIntervals != DEG) {
			const Vector2& interval = intervalQueue.front();
			float a = interval[0];
			float b = interval[1];
			intervalQueue.pop();

			int nRoots = countSturmSignChanges(a) - countSturmSignChanges(b);
			if (nRoots == 1 || b - a < tau) {
				intervals[nIntervals++] = Vector2(a, b);

			} else if (nRoots > 1) {
				float m = (a + b)/2.0f;
				intervalQueue.emplace(Vector2(a, m));
				intervalQueue.emplace(Vector2(m, b));
			}
		}
	}

	// computes the roots contained in the intervals
	void computeRoots(bool ignoreMin, bool ignoreMax, float *roots, bool *isValid) {
		for (int i = 0; i < nIntervals; i++) {
			// evaluate the polynomial at the interval endpoints
			float a = intervals[i][0];
			float b = intervals[i][1];
			float pa = evaluatePolynomial(a);
			float pb = evaluatePolynomial(b);

			// categorize the root
			bool isInflection = pa*pb > 0;
			bool isMin = pa < 0 && pb > 0;
			bool isMax = pa > 0 && pb < 0;
			bool prune = isInflection || (ignoreMin && isMin) || (ignoreMax && isMax);

			if (prune) {
				isValid[i] = false;

			} else {
				// perform binary search to isolate the root
				float sign = isMin ? 1.0f : -1.0f;
				float m = (a + b)/2.0f;
				int iter = 0;

				do {
					float pm = evaluatePolynomial(m);
					bool maxIterExceeded = iter > 1000;
					if (maxIterExceeded) {
						std::cout << "PolynomialRootFinder::findRoots(): Max iterations exceeded" << std::endl;
					}

					if (std::fabs(pm) < tau || b - a < tau || maxIterExceeded) break;
					else if (pm*sign < 0) a = m;
					else b = m;

					m = (a + b)/2.0f;
					iter++;
				} while (true);

				roots[i] = m;
				isValid[i] = true;
			}
		}
	}

	// members
	float tau;
	int nIntervals;
	Vector2 intervals[DEG];
	float sturmChain[DEG + 1][DEG + 1];
};

} // namespace pyzombie

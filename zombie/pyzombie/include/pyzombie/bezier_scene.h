#pragma once

#include "pyzombie/common.h"
#include "pyzombie/polynomial_root_finder.h"
#include "pcg32.h"
#include <cmath>

namespace py = pybind11;

#define BEZIER_RAY_OFFSET 1e-6f
#define BEZIER_EPSILON 1e-6f
#define BEZIER_GRADIENT_EPSILON 1e-12

// only valid for 2D
namespace pyzombie {

struct BezierInteraction {
	int bezierIndex;
	bool isBbox = false;
	Vector2 p;							// curve point
	Vector2 n;							// curve normal
	float d = fcpw::maxFloat;			// distance to curve point
	float sign = 1.0f;					// sign of distance (1 exterior / -1 interior)
	float s;							// texture coordinate (dist along bezier)

	BezierInteraction() {}
};

struct BezierPoint {
	Vector2 pt;
	Vector2 prev;
	Vector2 next;
	Vector2 delta;
	float prevScale, nextScale;

	BezierPoint(Vector2 pt, Vector2 delta, float prevScale, float nextScale):
			    pt(pt),
				prev(pt - std::exp(prevScale) * delta),
				next(pt + std::exp(nextScale) * delta),
				delta(delta),
				prevScale(prevScale),
				nextScale(nextScale) {}
};

struct Bezier {
	int degree = 3;
	float cpPolynomialDegree;
	const VectorI2 index;
	std::array<Vector2, 4> coefficients;
	const std::vector<BezierPoint> &bezierPoints;

	Bezier(const std::vector<BezierPoint> &bezierPoints, VectorI2 index):
		   bezierPoints(bezierPoints), index(index) {
		update();
	} 

	void update() {
		if (bezierPoints.size() == 0) {
			coefficients.fill(Vector2::Zero());
			cpPolynomialDegree = 0;
		} else {
			const Vector2 &pa = bezierPoints[index[0]].pt;	
			const Vector2 &pb = bezierPoints[index[0]].next;
			const Vector2 &pc = bezierPoints[index[1]].prev;	
			const Vector2 &pd = bezierPoints[index[1]].pt;	

			coefficients[0] = -pa + 3.0f * pb - 3.0f * pc + pd;
			coefficients[1] = 3.0f * pa - 6.0f * pb + 3.0 * pc;
			coefficients[2] = 3.0f * (pb - pa);
			coefficients[3] = pa;

			bool aIsZero = std::fabs(coefficients[0].squaredNorm()) < BEZIER_EPSILON;
			bool bIsZero = std::fabs(coefficients[1].squaredNorm()) < BEZIER_EPSILON;
			bool cIsZero = std::fabs(coefficients[2].squaredNorm()) < BEZIER_EPSILON;

			if (aIsZero && bIsZero && cIsZero) cpPolynomialDegree = 0;
			else if (aIsZero && bIsZero) cpPolynomialDegree = 1;
			else if (aIsZero) cpPolynomialDegree = 3;
			else cpPolynomialDegree = 5;
		}
	}
	
	void closestPoint(const Vector2& x, BezierInteraction& interaction, float cpPrecision, bool flipNormalOrientation) const {
		if (bezierPoints.size() == 0) {
			throw std::runtime_error("CPQ cannot be performed, no position data.");
		}

		const Vector2 &pa = bezierPoints[index[0]].pt;	
		const Vector2 &pb = bezierPoints[index[0]].next;	
		const Vector2 &pc = bezierPoints[index[1]].prev;	
		const Vector2 &pd = bezierPoints[index[1]].pt;	

		// finding the closest point to a cubic bezier requires finding the roots of a 5th
		// degree polynomial of the form (B(t) - x).dB(t) = 0; however, degenerate cases
		// can result in finding roots to 1st or 3rd degree polynomials as well
		float roots[7];
		roots[5] = 0.0f;
		roots[6] = 1.0f;
		bool isValid[7] = {false, false, false, false, false, true, true};

		Vector2 dBx = coefficients[3] - x;
		std::array<Vector2, 3> dCoefficients;
		dCoefficients[0] = 3.0f*coefficients[0];
		dCoefficients[1] = 2.0f*coefficients[1];
		dCoefficients[2] = coefficients[2];
		if (cpPolynomialDegree == 1) {
			// compute the coefficients of the 1st degree polynomial
			Vector2 ones = Vector2::Ones();
			roots[0] = -dBx.dot(ones) / coefficients[2].dot(ones);
			isValid[0] = true;
		} else if (cpPolynomialDegree == 3) {
			// compute the coefficients of the 3rd degree polynomial
			float a = coefficients[1].dot(dCoefficients[1]);
			float b = coefficients[1].dot(dCoefficients[2]) +
					  coefficients[2].dot(dCoefficients[1]);
			float c = coefficients[2].dot(dCoefficients[2]) +
					  dBx.dot(dCoefficients[1]);
			float d = dBx.dot(dCoefficients[2]);

			// find the roots
			computeCubicRoots(a, b, c, d, roots, isValid);

		} else if (cpPolynomialDegree == 5) {
			// compute the coefficients of the 5th degree polynomial
			PolynomialRootFinder<5> rootFinder(cpPrecision);
			rootFinder.getCoefficient(0) = coefficients[0].dot(dCoefficients[0]);
			rootFinder.getCoefficient(1) = coefficients[0].dot(dCoefficients[1]) +
										   coefficients[1].dot(dCoefficients[0]);
			rootFinder.getCoefficient(2) = coefficients[0].dot(dCoefficients[2]) +
										   coefficients[1].dot(dCoefficients[1]) +
										   coefficients[2].dot(dCoefficients[0]);
			rootFinder.getCoefficient(3) = coefficients[1].dot(dCoefficients[2]) +
										   coefficients[2].dot(dCoefficients[1]) +
										   dBx.dot(dCoefficients[0]);
			rootFinder.getCoefficient(4) = coefficients[2].dot(dCoefficients[2]) +
										   dBx.dot(dCoefficients[1]);
			rootFinder.getCoefficient(5) = dBx.dot(dCoefficients[2]);

			// find the roots
			rootFinder.findRoots(BEZIER_EPSILON, 1.0f - BEZIER_EPSILON, false, true, roots, isValid);
		}

		// find the closest point
		float t = fcpw::maxFloat;
		Vector2 p;
		float root = 0.0f;
		for (int i = 0; i < 7; i++) {
			if (isValid[i]) {
				Vector2 r = i == 5 ? pa : (i == 6 ? pd : evaluate(roots[i]));
				float u = (x - r).norm();
				if (u < t) {
					t = u;
					p = r;
					root = roots[i];
				}
			}
		}
		
		interaction.p = p;

		interaction.n = normal(std::clamp(root, 1e-3f, 1.0f - 1e-3f), true);
		interaction.n = interaction.n / interaction.n.norm();
		interaction.n = flipNormalOrientation ? -interaction.n : interaction.n;

		interaction.d = std::abs(t);
		interaction.sign = interaction.n.dot(interaction.p - x) > 0 ? -1.0 : 1.0;

		interaction.s = std::clamp(root, 0.0f, 1.0f);
		interaction.isBbox = false;
	}

	Vector2 derivative(float t, int n) const {
		Vector2 p = Vector2::Zero();
		for (int i = 0; i <= degree - n; i++) {
			float c = 1.0f;
			for (int j = 0; j < n; j++) c *= (degree - i - j);
			p *= t;
			p += c * coefficients[i];
		}

		return p;
	}

	Vector2 tangent(float t, bool normalized = true) const {
		Vector2 p = derivative(t, 1);
		if (normalized) {
			float pNorm = p.norm();
			if (pNorm > BEZIER_EPSILON) p /= pNorm;
		}
		return p;
	}

	Vector2 normal(float t, bool normalized = true) const {
		Vector2 s = tangent(t, normalized);
		return Vector2(s(1), -s(0));
	}

	Vector2 evaluate(float t) const {
		// evaluate using Horner's method
		Vector2 p = coefficients[0];
		for (int i = 1; i <= degree; i++) {
			p *= t;
			p += coefficients[i];
		}

		return p;
	}

	float curvature(float t) const {
		Vector2 N = normal(t, false);
		Vector2 dTds = derivative(t, 2);
		return N.dot(dTds) / std::pow(N.norm(), 3.0);
	}

	zombie::Differential<Vector2> computeSpatialDisplacement(std::string id, float s) const {
		/* coefficients 
			B(s) = (1-s)^3 pa + 3 (1-s)^2 s (pa + pb) + 3 (1-s) s^2 (pc + pd) + s^3 pd
		*/
		std::string P0 = std::to_string(index[0]);
		std::string P1 = std::to_string(index[1]);

		const BezierPoint &p0 = bezierPoints[index[0]];
		const BezierPoint &p1 = bezierPoints[index[1]];
		const Vector2 &d0 = p0.delta;
		const Vector2 &d1 = p1.delta;
		float scale0 = std::exp(p0.nextScale);
		float scale1 = -std::exp(p1.prevScale);

		float s2 = s * s;
		float t = (1.0 - s);
		float t2 = t * t;
		float dBda = t2 * t;
		float dBdb = 3 * t2 * s;
		float dBdc = 3 * t * s2;
		float dBdd = s2 * s;

		zombie::Differential<Vector2> v(Vector2::Zero());
		v.ref(id + "#point." + P0 + ".0") = (dBda + dBdb) * Vector2(1.0f, 0.0f);  	// x0
		v.ref(id + "#point." + P0 + ".1") = (dBda + dBdb) * Vector2(0.0f, 1.0f);  	// y0
		v.ref(id + "#delta." + P0 + ".0") = dBdb * scale0 * Vector2(1.0f, 0.0f); 	// dx0
		v.ref(id + "#delta." + P0 + ".0") = dBdb * scale0 * Vector2(0.0f,  1.0);	// dy0
		v.ref(id + "#scale." + P0 + ".1") = dBdb * scale0 * d0;						// scale0
		v.ref(id + "#point." + P1 + ".0") = (dBdd + dBdc) * Vector2(1.0f, 0.0f);  	// x1
		v.ref(id + "#point." + P1 + ".1") = (dBdd + dBdc) * Vector2(0.0f, 1.0f);  	// y1
		v.ref(id + "#delta." + P1 + ".0") = dBdc * scale1 * Vector2(1.0f, 0.0f); 	// dx1
		v.ref(id + "#delta." + P1 + ".1") = dBdc * scale1 * Vector2(0.0f, 1.0f); 	// dy1
		v.ref(id + "#scale." + P1 + ".0") = dBdc * scale1 * d1;  					// scale1
		return v;
	}

	zombie::Differential<float> computeTextureDisplacement(std::string id, Vector2 x, float t) const {
		std::string P0 = std::to_string(index[0]);
		std::string P1 = std::to_string(index[1]);

		const Array2 A = coefficients[0].array();
		const Array2 B = coefficients[1].array();
		const Array2 C = coefficients[2].array();
		const Array2 D = (coefficients[3].array() - x.array());

		float tn[6];
		tn[0] = 1.0f;
		for (int i = 1; i < 6; i++)
			tn[i] = t * tn[i-1];

		// let P = (B - x) * B'
		float dPdt = ((15 * A * A) * tn[4] +
			          (20 * A * B) * tn[3] +
			          (12 * A * C + 6 * B * B) * tn[2] + 
			          (6 * A * D + 6 * B * C) * tn[1] + 
			          (2 * D * B + C * C) * tn[0]).sum();

		Array2 dPdA = (6 * A) * tn[5] +
					  (5 * B) * tn[4] + 
					  (4 * C) * tn[3] + 
					  (3 * D) * tn[2];
		Array2 dPdB = (5 * A) * tn[4] +
		              (4 * B) * tn[3] + 
					  (3 * C) * tn[2] +
					  (2 * D) * tn[1];
		Array2 dPdC = (4 * A) * tn[3] +
		              (3 * B) * tn[2] +
					  (2 * C) * tn[1] +
					   D;
		Array2 dPdD = (3 * A) * tn[2] +
					  (2 * B) * tn[1] +
					  C;

		Array2 dtdA = -dPdA / dPdt;
		Array2 dtdB = -dPdB / dPdt;
		Array2 dtdC = -dPdC / dPdt;
		Array2 dtdD = -dPdD / dPdt;

		Array2 dtda = -dtdA + 3 * dtdB - 3 * dtdC + dtdD;
		Array2 dtdb = 3 * dtdA - 6 * dtdB + 3 * dtdC;
		Array2 dtdc = -3 * dtdA + 3 * dtdB;
		Array2 dtdd = dtdA;
		
		const BezierPoint &p0 = bezierPoints[index[0]];
		const BezierPoint &p1 = bezierPoints[index[1]];
		const Array2 &d0 = p0.delta.array();
		const Array2 &d1 = p1.delta.array();
		float scale0 = std::exp(p0.nextScale);
		float scale1 = -std::exp(p1.prevScale);

		// derivatives computed using chain rule dt/d(param) = dt/da.x da.x/(param) + dt/da.y da.y/(param) + ....
		zombie::Differential<float> v(0.0f);
		v.ref(id + "#point." + P0 + ".0") = (dtda[0] + dtdb[0]);	// x0
		v.ref(id + "#point." + P0 + ".1") = (dtda[1] + dtdb[1]);	// y0
		v.ref(id + "#delta." + P0 + ".0") = dtdb[0] * scale0;		// dx0
		v.ref(id + "#delta." + P0 + ".1") = dtdb[1] * scale0;		// dy0
		v.ref(id + "#scale." + P0 + ".1") = (dtdb * scale0).sum(); 	// scale0
		v.ref(id + "#point." + P1 + ".0") = (dtdd[0] + dtdc[0]);	// x1
		v.ref(id + "#point." + P1 + ".1") = (dtdd[1] + dtdc[1]);	// y1
		v.ref(id + "#delta." + P1 + ".0") = dtdc[0] * scale1;		// dx1
		v.ref(id + "#delta." + P1 + ".1") = dtdc[1] * scale1;		// dy1
		v.ref(id + "#scale." + P1 + ".0") = (dtdc * scale1).sum(); 	// scale1
		return v;
	}
};

struct SceneBoundingBox {
	Vector2 pMin, pMax;
	Vector2 extent;

	SceneBoundingBox(Vector2 pMin, Vector2 extent): 
		pMin(pMin), 
		pMax(pMin + extent), 
		extent(extent) {}

	void closestPoint(const Vector2& x, BezierInteraction& interaction, bool flipNormalOrientation) const {
    	Vector2 clampedPt(std::clamp(x[0], pMin[0], pMax[0]),
						  std::clamp(x[1], pMin[1], pMax[1]));
		interaction.isBbox = true;

		float clampedDist = (clampedPt - x).norm();
		if (clampedDist < BEZIER_EPSILON) {
			// distance to sides
			float dist[4] = {
				x[0] - pMin[0], // left
				pMax[0] - x[0], // right
				x[1] - pMin[1], // bottom
				pMax[1] - x[1] // top
			};

			// find box segment 
			float minDist = fcpw::maxFloat;	
			int segmentIdx = 0;
			for (int i = 0; i < 4; i++) {
				if (dist[i] < minDist) {
					segmentIdx = i;
					minDist = dist[i];
				}
			}

			// compute interaction point + normal
			if (segmentIdx == 0) {
				// left
				interaction.p = Vector2(pMin[0], x[1]);
				interaction.n = Vector2(-1, 0);
				interaction.d = minDist;	
			} else if (segmentIdx == 1) {
				// right
				interaction.p = Vector2(pMax[0], x[1]);
				interaction.n = Vector2(1, 0);
				interaction.d = minDist;
			} else if (segmentIdx == 2) {
				// bottom
				interaction.p = Vector2(x[0], pMin[1]);
				interaction.n = Vector2(0, -1);
				interaction.d = minDist;
			} else {
				// top
				interaction.p = Vector2(x[0], pMax[1]);
				interaction.n = Vector2(0, 1);
				interaction.d = minDist;
			}
			interaction.sign = -1.0f;
		} else {
			if (x[1] == clampedPt[1]) {
				// projected along x dim	
				if (x[0] < pMin[0] + extent[0] / 2.0) {
					// left
					interaction.p = Vector2(pMin[0], x[1]);
					interaction.n = Vector2(1, 0);
				} else {
					// right
					interaction.p = Vector2(pMax[0], x[1]);
					interaction.n = Vector2(-1, 0);
				}
			} else {
				// projected along y dim	
				if (x[1] < pMin[1] + extent[1] / 2.0) {
					// bottom
					interaction.p = Vector2(x[0], pMin[1]);
					interaction.n = Vector2(0, 1);
				} else {
					// top
					interaction.p = Vector2(x[0], pMax[1]);
					interaction.n = Vector2(0, -1);
				}
			}
			interaction.d = clampedDist;
			interaction.sign = 1.0;
		}

		if (flipNormalOrientation) {
			interaction.n *= -1.0f;
			interaction.sign *= -1.0f;
		}
	}

};

class BezierScene {
public:
	zombie::PDE<RGB, 2> pde;
	zombie::GeometricQueries<2> queries;
	std::string id = "d";

	bool ignoreTextureDerivatives = false;
	bool ignoreShapeDerivatives = false;

	bool flipBezierNormalOrientation = false;
	bool flipBoundingBoxNormalOrientation = false;

	float cpPrecision = 1e-4f;

	SceneBoundingBox bbox;
	RGB backgroundColor = RGB(1.0, 1.0, 1.0);
	std::vector<RGB> color;
	std::vector<RGB> colorExterior;
	std::vector<BezierPoint> bezierPoints;
	std::vector<Bezier> beziers;

	BezierScene(Vector2 pMin = Vector2(-1.0, -1.0), 
				Vector2 extent = Vector2(2, 2),
				bool domainIsWatertight = false): 
				queries(domainIsWatertight),
				bbox(pMin, extent) {
		populateGeometricQueries();
		populatePDE();
	}

	void setColor(const py::array_t<float> &color) {
		py::buffer_info colorBuffer = color.request();
		if (colorBuffer.ndim != 2 || colorBuffer.shape[1] != 3) {
			throw std::runtime_error("Error: color array should have dimensions Nx3");
		}
		this->color.clear();
		this->colorExterior.clear();
		auto col = color.unchecked<2>();
		for (int i = 0; i < col.shape(0); i++) {
			RGB rgb(col(i, 0), col(i, 1), col(i, 2));
			this->color.emplace_back(rgb);
			this->colorExterior.emplace_back(rgb);
		}
	}

	void setDoubleSidedColor(const py::array_t<float> &color) {
		py::buffer_info colorBuffer = color.request();
		if (colorBuffer.ndim != 3 || colorBuffer.shape[1] != 2 || colorBuffer.shape[2] != 3) {
			throw std::runtime_error("Error: color array should have dimensions Nx2x3");
		}
		this->color.clear();
		this->colorExterior.clear();
		auto col = color.unchecked<3>();
		for (int i = 0; i < col.shape(0); i++) {
			this->color.emplace_back(RGB(col(i, 0, 0), col(i, 0, 1), col(i, 0, 2)));
			this->colorExterior.emplace_back(RGB(col(i, 1, 0), col(i, 1, 1), col(i, 1, 2)));
		}
	}

	void setBezierPoints(const py::array_t<float> &bezierPoints, 
						 const py::array_t<float> &bezierDelta,
						 const py::array_t<float> &bezierScale)  {
		py::buffer_info bezierPointsBuffer= bezierPoints.request();
		if (bezierPointsBuffer.ndim != 2 || 
			bezierPointsBuffer.shape[1] != 2) {
			throw std::runtime_error("Error: Position array should have dimensions Nx2");
		}
		py::buffer_info bezierDeltaBuffer= bezierDelta.request();
		if (bezierDeltaBuffer.ndim != 2 || 
			bezierDeltaBuffer.shape[1] != 2) {
			throw std::runtime_error("Error: Delta array should have dimensions Nx2");
		}
		py::buffer_info bezierScaleBuffer= bezierScale.request();
		if (bezierScaleBuffer.ndim != 2 || 
			bezierScaleBuffer.shape[1] != 2) {
			throw std::runtime_error("Error: Scale array should have dimensions Nx2");
		}

		this->bezierPoints.clear();

		auto pointsData = bezierPoints.unchecked<2>();
		auto deltaData = bezierDelta.unchecked<2>();
		auto scaleData = bezierScale.unchecked<2>();

		if (pointsData.shape(0) != deltaData.shape(0) || pointsData.shape(0) != scaleData.shape(0)) {
			throw std::runtime_error("Error: position, delta, and scale should have same number of elements");
		}
		for (int i = 0; i < pointsData.shape(0); i++) {
			BezierPoint bp(Vector2(pointsData(i, 0), pointsData(i, 1)), 
						   Vector2(deltaData(i, 0), deltaData(i, 1)),
						   scaleData(i, 0), scaleData(i, 1));
			this->bezierPoints.emplace_back(bp);
		}

		for (Bezier& bezier : beziers){
			bezier.update();
		}
	}

	void setBeziers(const py::array_t<int> &index) {
		py::buffer_info indexBuffer = index.request();
		if (indexBuffer.ndim != 2 || indexBuffer.shape[1] != 2) {
			throw std::runtime_error("Error: Index array should have dimensions Nx2");
		}
		beziers.clear();
		auto data = index.unchecked<2>();
		for (int i = 0; i < data.shape(0); i++) {
			Bezier b(this->bezierPoints, {data(i, 0), data(i, 1)});
			beziers.emplace_back(b);
		}
	}

	void closestPoint(const Vector2& x, BezierInteraction& closestInteraction) const {
		bbox.closestPoint(x, closestInteraction, flipBoundingBoxNormalOrientation);
		for (int i = 0; i < beziers.size(); i++) {
			BezierInteraction interaction;
			beziers[i].closestPoint(x, interaction, cpPrecision, flipBezierNormalOrientation);
			if (interaction.d < closestInteraction.d) {
				closestInteraction = interaction;
				closestInteraction.bezierIndex = i;
			}
		}
	}

	std::vector<BoundarySamplePoint<RGB, 2>> samplePoints(int nPointsPerBezier) const {
		pcg32 sampler;
		float primarySpaceIntervalLength = 1.0f / float(nPointsPerBezier);
		std::vector<BoundarySamplePoint<RGB, 2>> samplePts(nPointsPerBezier * beziers.size());
		for (int i = 0; i < beziers.size(); i++) {
			Bezier bezier = beziers[i];
			for (int j = 0; j < nPointsPerBezier; j++) {
				float t = primarySpaceIntervalLength * (j + sampler.nextFloat());
				BoundarySamplePoint<RGB, 2>& samplePt = samplePts[i * nPointsPerBezier + j];
				samplePt.pt = bezier.evaluate(t);
				Vector2 n = bezier.normal(t, true);
				if (flipBezierNormalOrientation) n *= -1.0f;
				samplePt.pdf = (1.0 / beziers.size()) * (1.0f / bezier.derivative(t, 1).norm()); 
				samplePt.vn = zombie::dot<2>(bezier.computeSpatialDisplacement(id, t), n);
				RGB A, B;
				getColors(bezier.index, false, A, B);
				samplePt.u = lerp<RGB>(1.0, A, B);
				getColors(bezier.index, true, A, B);
				samplePt.uExterior = lerp<RGB>(1.0, A, B);
			}
		}
		return samplePts;
	}

	zombie::Value<float, 2> computeLength(int nPointsPerBezier) const {
		zombie::Value<float, 2> L(0.0f);
		pcg32 sampler;
		float primarySpaceIntervalLength = 1.0f / float(nPointsPerBezier);
		for (const Bezier& bezier: beziers)
		for (int i = 0; i < nPointsPerBezier; i++) {
			float t = primarySpaceIntervalLength * (i + sampler.nextFloat());
			
			Vector2 dBds = bezier.derivative(t, 1);
			Vector2 n = Vector2(-dBds[1], dBds[0]).normalized();
			zombie::Differential<float> vn = zombie::dot<2>(bezier.computeSpatialDisplacement(id, t), n);
			float kappa = bezier.curvature(t);

			L.data += primarySpaceIntervalLength * dBds.norm();
			L.differential += (vn * primarySpaceIntervalLength * kappa * dBds.norm()); 
		}
		return L;
	}
	
private:
	void getColors(const VectorI2 &index, bool exterior, RGB &A, RGB &B) const {
		bool useExterior = exterior && colorExterior.size() > 0;
		if (!useExterior && color.size() > 0) {
			A = color[index[0]];
			B = color[index[1]];
		} else if (useExterior){
			A = colorExterior[index[0]];
			B = colorExterior[index[1]];
		} else {
			A = RGB::Zero();
			B = RGB::Zero();
		}
	}

	void populatePDE() {
		pde.absorption = 0.0f;
		pde.dirichlet = [this](Vector2 x) -> zombie::Value<RGB, 2> {
			BezierInteraction interaction;
			this->closestPoint(x, interaction);
			if (interaction.isBbox) {
				zombie::Value<RGB, 2> g(RGB::Zero());
				g.data = backgroundColor;
				g.differential.ref(id + "#background-color") = 1.0f;
				return g;
			}

			const Bezier& bezier = beziers[interaction.bezierIndex];
			RGB A, B;
			getColors(bezier.index, false, A, B);

			float s = interaction.s;
			std::string P0 = std::to_string(bezier.index[0]);
			std::string P1 = std::to_string(bezier.index[1]);

			zombie::Value<RGB, 2> g(RGB::Zero());
			g.data = lerp<RGB>(s, A, B);
			g.differential.ref(id + "#color." + P0) = RGB((1.0 - s));
			g.differential.ref(id + "#color." + P1) = RGB(s);
			
			// avoid computing texture displacement if (B-A) = 0 or texture derivatives disabled
			RGB dgds = (B - A);
			if (dgds.matrix().norm() > 0 && !this->ignoreTextureDerivatives) {
				zombie::Differential<float> dsdt = bezier.computeTextureDisplacement(id, x, s);
				g.differential += dsdt * dgds;
			}
			return g;
		};
		pde.neumann = [](Vector2 x) -> zombie::Value<RGB, 2> { 
			return zombie::Value<RGB, 2>(RGB::Zero());
		};
		pde.source = [](Vector2 x) -> zombie::Value<RGB, 2> { 
			return zombie::Value<RGB, 2>(RGB::Zero());
		};
		pde.dirichletDoubleSided = [this](Vector2 x, bool exterior) -> zombie::Value<RGB, 2> { 
			BezierInteraction interaction;
			this->closestPoint(x, interaction);
			if (interaction.isBbox) {
				zombie::Value<RGB, 2> g(RGB::Zero());
				g.data = backgroundColor;
				g.differential.ref(id + "#background-color") = 1.0f;
				return g;
			}

			const Bezier& bezier = beziers[interaction.bezierIndex];
			
			RGB A, B;
			getColors(bezier.index, exterior, A, B);

			float s = interaction.s;
			std::string P0 = std::to_string(bezier.index[0]);
			std::string P1 = std::to_string(bezier.index[1]);
			std::string SIDE = std::to_string(exterior);

			zombie::Value<RGB, 2> g(RGB::Zero());
			g.data = lerp<RGB>(s, A, B);
			g.differential.ref(id + "#color." + P0 + "." + SIDE) = RGB((1.0 - s));
			g.differential.ref(id + "#color." + P1 + "." + SIDE) = RGB(s);
			
			// avoid computing texture displacement if (B-A) = 0 or texture derivatives disabled
			RGB dgds = (B - A);
			if (dgds.matrix().norm() > 0 && !this->ignoreTextureDerivatives) {
				zombie::Differential<float> dsdt = bezier.computeTextureDisplacement(id, x, s);
				g.differential += dsdt * dgds;
			}
			return g;
		};
		pde.neumannDoubleSided = [](Vector2 x, bool exterior) -> zombie::Value<RGB, 2> { 
			return zombie::Value<RGB, 2>(RGB::Zero());
		};
	}	

	void populateGeometricQueries() {
		queries.computeDistToDirichlet = [this](const Vector2& x, bool computeSignedDistance) -> float {
			BezierInteraction interaction;
			this->closestPoint(x, interaction);
			return computeSignedDistance ? interaction.sign * interaction.d : interaction.d;
		};
		queries.computeDistToNeumann = [](const Vector2& x, bool computeSignedDistance) -> float {
			return fcpw::maxFloat;
		};
		queries.computeDistToBoundary = [this](const Vector2& x,
											   bool computeSignedDistance,
											   bool& closerToDirichlet) -> float {
			closerToDirichlet = true;
			return this->queries.computeDistToDirichlet(x, computeSignedDistance);
		};
		queries.projectToDirichlet = [this](Vector2& x, Vector2& normal,
											float& distance, bool computeSignedDistance) -> bool {
			BezierInteraction interaction;
			this->closestPoint(x, interaction);

			x = interaction.p;
			distance = computeSignedDistance ? interaction.sign * interaction.d : interaction.d;
			normal = interaction.n;

			return interaction.d != fcpw::maxFloat;
		};
		queries.projectToNeumann = [](Vector2& x, Vector2& normal,
									  float& distance, bool computeSignedDistance) -> bool {
			return false;
		};
		queries.projectToBoundary = [this](Vector2& x, Vector2& normal, float& distance,
										   bool& projectToDirichlet, bool computeSignedDistance) -> bool {
			projectToDirichlet = true;
			return this->queries.projectToDirichlet(x, normal, distance, computeSignedDistance);
		};
		queries.offsetPointAlongDirection = [](const Vector2& x, const Vector2& dir) -> Vector2 {
			return x + BEZIER_RAY_OFFSET * dir;
		};
		queries.intersectWithDirichlet = [](const Vector2& origin, const Vector2& normal,
											const Vector2& dir, float tMax, bool onDirichletBoundary,
											zombie::IntersectionPoint<2>& intersectionPt) -> bool {
			return false;
		};
		queries.intersectWithNeumann = [](const Vector2& origin, const Vector2& normal,
										  const Vector2& dir, float tMax, bool onNeumannBoundary,
										  zombie::IntersectionPoint<2>& intersectionPt) -> bool {
			return false;
		};
		queries.intersectWithBoundary = [](const Vector2& origin, const Vector2& normal,
										   const Vector2& dir, float tMax, bool onDirichletBoundary,
										   bool onNeumannBoundary, zombie::IntersectionPoint<2>& intersectionPt,
										   bool& hitDirichlet) -> bool {
			return false;
		};
		queries.intersectWithBoundaryAllHits = [](const Vector2& origin, const Vector2& normal,
												  const Vector2& dir, float tMax,
												  bool onDirichletBoundary, bool onNeumannBoundary,
												  std::vector<zombie::IntersectionPoint<2>>& intersectionPts,
												  std::vector<bool>& hitDirichlet) -> int {
			return false;
		};
		queries.sampleNeumann = [](const Vector2& x, float radius, float *randNums,
								   zombie::BoundarySample<2>& neumannSample) -> bool {
			return false;
		};
		queries.computeStarRadius = [](const Vector2& x, float minRadius,
									   float maxRadius, float silhouettePrecision,
									   bool flipNormalOrientation) -> float {
			return maxRadius;
		};
		queries.insideDomain = [](const Vector2& x) -> bool {
			return true;
		};
		queries.outsideBoundingDomain = [](const Vector2& x) -> bool {
			return false;
		};	
		queries.computeDirichletDisplacement = [this](const Vector2& x) -> zombie::Differential<Vector<2>> {
			if (this->ignoreShapeDerivatives) return zombie::Differential<Vector2>(Vector2::Zero());
			BezierInteraction interaction;
			this->closestPoint(x, interaction);
			if (interaction.isBbox) return zombie::Differential<Vector2>(Vector2::Zero());
			return this->beziers[interaction.bezierIndex].computeSpatialDisplacement(id, interaction.s);	
		};
	}

};

}; // pyzombie

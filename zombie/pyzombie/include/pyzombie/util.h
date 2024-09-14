#include "pyzombie/common.h"

namespace pyzombie {

template<typename T, int DIM>
void setZeroBoundaryConditions(zombie::PDE<T, DIM> &pde, T initVal) {
	pde.absorption = initVal;
	pde.dirichlet = [initVal](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return initVal;
	};
	pde.neumann = [initVal](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return initVal;
	};
	pde.source = [initVal](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return initVal;
	};
	pde.dirichletDoubleSided = [initVal](const Vector<DIM>& x, bool _) -> zombie::Value<T, DIM> {
		return initVal;
	};
	pde.neumannDoubleSided = [initVal](const Vector<DIM>& x, bool _) -> zombie::Value<T, DIM> {
		return initVal;
	};
}

template<typename T, int DIM>
void setConstantDirichletBoundaryConditions(zombie::PDE<T, DIM> &pde, T g, T zero, fcpw::BoundingBox<DIM> &bbox) {
	pde.dirichlet = [g, zero, &bbox](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		zombie::Value<T, DIM> value(zero);
		if (bbox.contains(x)) {
			value.data = g;
		}
		return value;
	};
	pde.dirichletDoubleSided = [g, zero, &bbox](const Vector<DIM>& x, bool _) -> zombie::Value<T, DIM> {
		zombie::Value<T, DIM> value(zero);
		if (bbox.contains(x)) {
			value.data = g;
		}
		return value;
	};
}

template<typename T, int DIM>
void setConstantSource(zombie::PDE<T, DIM> &pde, T g, T zero) {
	pde.source = [g, zero](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		zombie::Value<T, DIM> value(zero);
		value.data = g;
		return g;
	};
}

template<typename T, int DIM>
void setDirichletGridBoundaryCondition(zombie::PDE<T, DIM> &pde, const Grid<T, DIM> &grid) {
	pde.dirichlet = [&grid](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return grid(x);
	};
	pde.dirichletDoubleSided = [&grid](const Vector<DIM>& x, bool _) -> zombie::Value<T, DIM> {
		return grid(x);
	};
}

template<typename T, int DIM>
void setNeumannGridBoundaryCondition(zombie::PDE<T, DIM> &pde, const Grid<T, DIM> &grid) {
	pde.neumann = [&grid](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return grid(x);
	};
	pde.neumannDoubleSided = [&grid](const Vector<DIM>& x, bool _) -> zombie::Value<T, DIM> {
		return grid(x);
	};
}

template<typename T, int DIM>
void setSourceGridCondition(zombie::PDE<T, DIM> &pde, const Grid<T, DIM> &grid) {
	pde.source = [&grid](const Vector<DIM> &x) -> zombie::Value<T, DIM> {
		return grid(x);
	};
}

template<typename T, int DIM>
zombie::SamplePoint<T, DIM> createSamplePoint(const zombie::GeometricQueries<DIM> &queries,
											  Vector<DIM> x, float pdf, T initVal,
											  bool inDomain = true, bool estimateBoundaryNormalAligned = false) {
	Vector<DIM> n;
	Vector<DIM> projX = x;
	bool projectToDirichlet;
	float dist, dirichletDist, neumannDist;

	// determine distance to boundary, normal, etc.
	if (!queries.projectToBoundary(projX, n, dist, projectToDirichlet, false)) {
		n = Vector<DIM>::Zero();
		dirichletDist = std::numeric_limits<float>::max();
		neumannDist = std::numeric_limits<float>::max();
	} else if (projectToDirichlet) {
		dirichletDist = std::abs(dist);
		neumannDist = std::abs(queries.computeDistToNeumann(x, false));
	} else {
		dirichletDist = std::abs(queries.computeDistToDirichlet(x, false));
		neumannDist = std::abs(dist);
	}

	// determine sample type, use projected x if its a boundary point
	zombie::SampleType sampleType;
	if (inDomain) {
		sampleType = zombie::SampleType::InDomain;
	} else if (dirichletDist < neumannDist) {
		sampleType = zombie::SampleType::OnDirichletBoundary;
	} else {
		sampleType = zombie::SampleType::OnNeumannBoundary;
	}
	zombie::SamplePoint<T, DIM> pt(x, n, sampleType, pdf, 
								   dirichletDist, neumannDist, initVal);
	pt.estimateBoundaryNormalAligned = estimateBoundaryNormalAligned;
	return pt;
}

template<typename T, int DIM>
VectorWrapper<zombie::SamplePoint<T, DIM>> createSamplePoints(const zombie::GeometricQueries<DIM> &queries,
											                  py::array_t<float> x, py::array_t<float> pdf, T initVal,
											                  bool inDomain = true, bool estimateBoundaryNormalAligned = false) {
	if (x.ndim() != 2 || x.shape(1) != DIM) {
        throw std::runtime_error("x array must be NxDIM");
    }
    if (pdf.ndim() != 1 || pdf.size() != x.shape(0)) {
        throw std::runtime_error("pdf array must be 1D and of the same length as x's first dimension");
    }
	
	auto x_buf = x.unchecked<2>(); // NxDIM array
    auto pdf_buf = pdf.unchecked<1>(); // 1D array

    VectorWrapper<zombie::SamplePoint<T, DIM>> samplePts;
	for (int i = 0; i < x.shape(0); i++) {
		Vector<DIM> samplePt;
		for (int j = 0; j < DIM; j++)
			samplePt[j] = x_buf(i, j);

		samplePts.push_back(createSamplePoint(queries, samplePt, pdf_buf(i), initVal, inDomain, estimateBoundaryNormalAligned));
	}
	return samplePts;
}

template <int DIM>
VectorWrapper<zombie::SampleEstimationData<DIM>> createSampleEstimationData(int nWalks, int nRecursiveWalks,
 																			 zombie::EstimationQuantity quantity, 
																			 py::array_t<float> n) {
	if (n.ndim() != 2 || n.shape(1) != DIM) {
        throw std::runtime_error("n array must be NxDIM");
    }
	auto n_buf = n.unchecked<2>(); // NxDIM array

    VectorWrapper<zombie::SampleEstimationData<DIM>> sampleEstimationData;
	for (int i = 0; i < n.shape(0); i++) {
		Vector<DIM> ptNormal;
		for (int j = 0; j < DIM; j++)
			ptNormal[j] = n_buf(i, j);
		sampleEstimationData.push_back(zombie::SampleEstimationData<DIM>(nWalks, nRecursiveWalks, quantity, ptNormal));
	}

	return sampleEstimationData;
}

template<typename T, int DIM>
void solve(const zombie::GeometricQueries<DIM> &queries,
		   const zombie::PDE<T, DIM> &pde,
		   const zombie::WalkSettings<T>& settings,
		   VectorWrapper<zombie::SampleEstimationData<DIM>>& sampleEstimationData,
		   VectorWrapper<zombie::SamplePoint<T, DIM>>& samplePoints,
		   bool runSingleThreaded = false) {
	zombie::WalkOnStars<T, DIM> wostSolver(queries);
	wostSolver.solve(pde, settings, sampleEstimationData.data, samplePoints.data, runSingleThreaded);
}

template <int DIM>
zombie::Value<float, DIM> computeL2Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
							            const py::array_t<float> &target) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);

	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		if (pts.val(i).statistics != nullptr) {
			float residual = pts.val(i).statistics->getEstimatedSolution() - targetData[i];
			J.data += 0.5 * residual * residual / pts.val(i).pdf;
			J.differential += (pts.val(i).statistics->getEstimatedSolutionWeightedDifferential() - 
							pts.val(i).statistics->getEstimatedDifferential() * targetData[i]) / pts.val(i).pdf;
		} else {
			float residual = targetData[i];
			J.data += 0.5 * residual * residual / pts.val(i).pdf;
		}
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<float, DIM> computeL1Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
										const py::array_t<float> &target) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);
	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		if (pts.val(i).statistics != nullptr) {
			float u = pts.val(i).statistics->getEstimatedSolution();
			float diff = targetData[i] - u;
			J.data += std::abs(diff) / pts.val(i).pdf;
			if (diff < 0) {
				J.differential += pts.val(i).statistics->getEstimatedDifferential() / pts.val(i).pdf;
			} else {
				J.differential -= pts.val(i).statistics->getEstimatedDifferential() / pts.val(i).pdf;
			}
		} else {
			J.data += std::abs(targetData[i]) / pts.val(i).pdf;
		}
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<float, DIM> computeMaskedL1Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
											  const VectorWrapper<zombie::SampleEstimationData<DIM>>& ests,
							            	  const py::array_t<float> &target) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);
	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		if (ests.val(i).estimationQuantity != zombie::EstimationQuantity::None)  {
			if (pts.val(i).statistics != nullptr) {
				float u = pts.val(i).statistics->getEstimatedSolution();
				float diff = targetData[i] - u;
				J.data += std::abs(diff) / pts.val(i).pdf;
				if (diff < 0) {
					J.differential += pts.val(i).statistics->getEstimatedDifferential() / pts.val(i).pdf;
				} else {
					J.differential -= pts.val(i).statistics->getEstimatedDifferential() / pts.val(i).pdf;
				}
			} else {
				J.data += std::abs(targetData[i]) / pts.val(i).pdf;
			}
		}
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<float, DIM> computeMaskedL2Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
											  const VectorWrapper<zombie::SampleEstimationData<DIM>>& ests,
							            	  const py::array_t<float> &target) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);
	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		if (ests.val(i).estimationQuantity != zombie::EstimationQuantity::None)  {
			if (pts.val(i).statistics != nullptr) {
				float residual = pts.val(i).statistics->getEstimatedSolution() - targetData[i];
				J.data += 0.5 * residual * residual / pts.val(i).pdf;
				J.differential += (pts.val(i).statistics->getEstimatedSolutionWeightedDifferential() - 
								pts.val(i).statistics->getEstimatedDifferential() * targetData[i]) / pts.val(i).pdf;
			} else {
				float residual = targetData[i];
				J.data += 0.5 * residual * residual / pts.val(i).pdf;
			}
		}
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<float, DIM> computeRelativeL2Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
							            		const py::array_t<float> &target, float epsilon) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);

	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		float estimate = pts.val(i).statistics->getEstimatedSolution();
		float weight = 1.0 / (targetData[i] * targetData[i] + epsilon);
		float residual = (estimate - targetData[i]);
		J.data += weight * 0.5 * residual * residual / pts.val(i).pdf;
		J.differential += weight * (pts.val(i).statistics->getEstimatedSolutionWeightedDifferential() - 
				 		   			pts.val(i).statistics->getEstimatedDifferential() * targetData[i]) / pts.val(i).pdf;
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<float, DIM> computeMaskedRelativeL2Loss(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts,
													  const VectorWrapper<zombie::SampleEstimationData<DIM>>& ests,
							            			  const py::array_t<float> &target, 
													  float epsilon) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 1 || targetBuffer.shape[0] != pts.size()) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	float* targetData = static_cast<float*>(targetBuffer.ptr);

	zombie::Value<float, DIM> J(0.0f);
	for (int i = 0; i < pts.size(); i++) {
		if (ests.val(i).estimationQuantity != zombie::EstimationQuantity::None)  {
			if (pts.val(i).statistics != nullptr) {
				float estimate = pts.val(i).statistics->getEstimatedSolution();
				float weight = 1.0 / (targetData[i] * targetData[i] + epsilon);
				float residual = (estimate - targetData[i]);
				J.data += weight * 0.5 * residual * residual / pts.val(i).pdf;
				J.differential += weight * (pts.val(i).statistics->getEstimatedSolutionWeightedDifferential() - 
											pts.val(i).statistics->getEstimatedDifferential() * targetData[i]) / pts.val(i).pdf;
			} else {
				float residual = targetData[i];
				J.data += 0.5 * residual * residual / pts.val(i).pdf;
			}
		}
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<RGB, DIM> computeL2LossRGB(const VectorWrapper<zombie::SamplePoint<RGB, DIM>>& pts,
							             const py::array_t<float> &target) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 2 || targetBuffer.shape[0] != pts.size() || targetBuffer.shape[1] != 3) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	auto targetMut = target.unchecked<2>();
	zombie::Value<RGB, DIM> J(RGB(0.0f));
	for (int i = 0; i < targetMut.shape(0); i++) {
		RGB targetVal;
		for (int j = 0; j < 3; j++) targetVal[j] = targetMut(i, j);
		RGB residual = (pts.val(i).statistics->getEstimatedSolution() - targetVal);
		J.data += 0.5 * (residual * residual) / pts.val(i).pdf;
		J.differential += (pts.val(i).statistics->getEstimatedSolutionWeightedDifferential() - 
						   pts.val(i).statistics->getEstimatedDifferential() * targetVal) / pts.val(i).pdf;
	}
	return J / float(pts.size());
}

template <int DIM>
zombie::Value<RGB, DIM> computeL2BoundaryLossRGB(const VectorWrapper<pyzombie::BoundarySamplePoint<RGB, DIM>>& pts,
							               		 const py::array_t<float> &target,
												 const py::array_t<float> &targetExterior) {
	py::buffer_info targetBuffer = target.request();
	if (targetBuffer.ndim != 2 || targetBuffer.shape[0] != pts.size() || targetBuffer.shape[1] != 3) {
		throw std::runtime_error("Error: target buffer should match position size");
	}
	
	py::buffer_info targetExteriorBuffer = targetExterior.request();
	if (targetExteriorBuffer.ndim != 2 || targetExteriorBuffer.shape[0] != pts.size() || targetExteriorBuffer.shape[1] != 3) {
		throw std::runtime_error("Error: target exterior buffer should match position size");
	}

	auto targetMut = target.unchecked<2>();
	auto targetExteriorMut = targetExterior.unchecked<2>();
	
	zombie::Value<RGB, DIM> J(RGB(0.0f));
	for (int i = 0; i < targetMut.shape(0); i++) {
		RGB targetVal, targetExteriorVal;
		for (int j = 0; j < 3; j++) {
			targetVal[j] = targetMut(i, j);
			targetExteriorVal[j] = targetExteriorMut(i, j);
		}
		RGB residual = (pts.val(i).u - targetVal);
		RGB residualExterior = (pts.val(i).uExterior - targetExteriorVal);
		RGB l2loss = 0.5 * ((residual * residual) - (residualExterior * residualExterior));
		J.differential += (pts.val(i).vn * l2loss) / pts.val(i).pdf;
	}
	return J / float(pts.size());
}

template <int DIM>
py::array_t<float> getSolution(const VectorWrapper<zombie::SamplePoint<float, DIM>>& pts) {
	py::array_t<float> u(pts.size());
    auto u_mut = u.mutable_unchecked<1>();
	for (int i = 0; i < pts.size(); i++) {
		if (pts.val(i).statistics != nullptr) {
			u_mut(i) = pts.val(i).statistics->getEstimatedSolution();
		} else {
			u_mut(i) = 0.0f;
		}
	}
	return u;
}

template <int DIM>
py::array_t<float> getSolutionRGB(const VectorWrapper<zombie::SamplePoint<RGB, DIM>>& pts) {
	py::array_t<float> u({pts.size(), 3});
    auto u_mut = u.mutable_unchecked<2>();
	for (int i = 0; i < pts.size(); i++) {
		RGB rgb = pts.val(i).statistics->getEstimatedSolution();
		for (int j = 0; j < 3; j++) {
			u_mut(i, j) = rgb[j];
		}
	}
	return u;
}

};

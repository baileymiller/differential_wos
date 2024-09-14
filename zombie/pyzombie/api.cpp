#include "pyzombie/analytic.h"
#include "pyzombie/common.h"
#include "pyzombie/grid.h"
#include "pyzombie/bezier_scene.h"
#include "pyzombie/mesh_geometry.h"
#include "pyzombie/toast_scene.h"
#include "pyzombie/util.h"

template<typename T>
py::array_t<float> toPyArray(const T &val) { }

template<>
py::array_t<float> toPyArray(const float &val) {
	py::array_t<float> result({1});
	auto result_mut = result.mutable_unchecked<1>();
	result_mut(0) = val;
	return result;
}

template<>
py::array_t<float> toPyArray(const pyzombie::RGB &val) {
	py::array_t<float> result({3});
	auto result_mut = result.mutable_unchecked<1>();
	for (int i = 0; i < 3; i++)
		result_mut(i) = val[i];
	return result;
}

// convert a length DIM array of type T into a numpy/python array
template<typename T,  int DIM>
py::array_t<float> arrToPyArray(const std::array<T, DIM> &data) {	}

template<>
py::array_t<float> arrToPyArray<float, 2>(const std::array<float, 2> &data) {	
	py::array_t<float> result(2);
	std::copy(data.begin(), data.end(), static_cast<float*>(result.request().ptr));
	return result;
}

// convert a vector of type T into a numpy/python array
template<typename T>
py::array_t<float> vecToPyArray(const std::vector<T> &data) {	}

template<>
py::array_t<float> vecToPyArray<float>(const std::vector<float> &data) {	
	py::array_t<float> result(data.size());
	std::copy(data.begin(), data.end(), static_cast<float*>(result.request().ptr));
	return result;
}

template<>
py::array_t<float> vecToPyArray<pyzombie::RGB>(const std::vector<pyzombie::RGB> &data) {	
	py::array_t<float> result({static_cast<long>(data.size()), static_cast<long>(3)});
	auto result_mut = result.mutable_unchecked<2>();
	for (int i = 0; i < data.size(); i++) {
		for (int j = 0; j < 3; j++) {
			result_mut(i, j) = data[i][j];
		}
	}
	return result;
}

template<typename T>
void bind_Vector(py::module_ &m, const char *typestr) {
	using Vector = pyzombie::VectorWrapper<T>;
	py::class_<Vector, std::shared_ptr<Vector>>(m, typestr)
		.def(py::init<>())
		.def("push_back", &Vector::push_back)
		.def("size", &Vector::size)
		.def("__getitem__", 
			 &Vector::operator[],
			 py::return_value_policy::reference)
		.def("__iter__", [](const Vector &vec) {
			return py::make_iterator(vec.data.begin(), vec.data.end());
		}, py::keep_alive<0, 1>());
};

template<typename T, int DIM>
void bind_SpatialGradient(py::module_ &m, const char *typestr) {
	using SpatialGradient = zombie::SpatialGradient<T, DIM>;
	py::class_<SpatialGradient>(m, typestr)
		.def("dot", [](const SpatialGradient& gradient, const pyzombie::Vector<DIM>& v) -> T {
			return gradient.dot(v);
		})
		.def("__getitem__", &SpatialGradient::operator[])
		.def("numpy", [](const SpatialGradient &grad) {
			return arrToPyArray<T, DIM>(grad.data);
		});
}

template<typename T>
void bind_Differential(py::module_ &m, const char *typestr) {
	using Differential = zombie::Differential<T>;
	py::class_<Differential>(m, typestr)
		.def(py::init<T>())
		.def("val", [](const Differential &self, std::string key) -> py::array_t<float> {
			return toPyArray<T>(self.val(key));
		})
		.def("ref", &Differential::ref)
		.def_readwrite("data", &Differential::data)
		.def("filter",
			 &Differential::filter,
			 py::arg("id"))
		.def("dense", [](const Differential &self, int size, std::function<int(std::string)> getIndex) -> py::array_t<float> {
			return vecToPyArray<T>(self.dense(size, getIndex));
		})
		.def("__sub__", [](const Differential &self, const Differential& other) {
			return self - other;
		})
		.def("__iadd__", [](Differential &self, const Differential& other) {
			self+=other;
			return &self;
		})
		.def("__mul__", [](const Differential &self, const float& other) {
			return self * other;
		})
		.def("__truediv__", [](const Differential &self, const float& other) {
			return self / other;
		})
		.def("__itruediv__", [](Differential &self, const float& other) {
			self /= other;
			return &self;
		});
};

template<typename T, int DIM>
void bind_GradientDifferential(py::module_ &m, const char *typestr) {
	using Differential = zombie::Differential<zombie::SpatialGradient<T, DIM>>;
	py::class_<Differential>(m, typestr)
		.def(py::init<T>())
		.def("val", &Differential::val)
		.def("ref", &Differential::ref)
		.def_readwrite("data", &Differential::data)
		.def("__sub__", [](const Differential &self, const Differential& other) -> Differential {
			return self - other;
		})
		.def("__iadd__", [](Differential &self, const Differential& other) -> Differential& {
			self+=other;
			return self;
		})
		.def("__mul__", [](const Differential &self, const float& other) -> Differential {
			return self * other;
		})
		.def("__truediv__", [](const Differential &self, const float& other) -> Differential {
			return self / other;
		})
		.def("__itruediv__", [](Differential &self, const float& other) -> Differential& {
			self /= other;
			return self;
		});
};

template<typename T, int DIM>
void bind_Value(py::module_ &m, const char *typestr) {
	using Value = zombie::Value<T, DIM>;
	py::class_<Value, std::shared_ptr<Value>>(m, typestr)
		.def(py::init<T>())
		.def_readwrite("data", &Value::data)
		.def_readwrite("gradient", &Value::gradient)
		.def_readwrite("differential", &Value::differential)
		.def("__iadd__", [](Value &self, const Value& other) -> Value& {
			self+=other;
			return self;
		})
		.def("__imul__", [](Value &self, const float& other) -> Value& {
			self*=other;
			return self;
		});
}

template<int DIM>
void bind_BBox(py::module_ &m, const char *typestr) {
	using BBox = fcpw::BoundingBox<DIM>;
	py::class_<BBox, std::shared_ptr<BBox>>(m, typestr)
		.def(py::init<>())
		.def("create", [](pyzombie::Vector<DIM> pMin, pyzombie::Vector<DIM> pMax) -> BBox {
			BBox bbox;
			bbox.pMin = pMin;
			bbox.pMax = pMax;
			return bbox;
		})
		.def_readwrite("pMin", &BBox::pMin)
		.def_readwrite("pMax", &BBox::pMax)
		.def("extent", &BBox::extent)
		.def("volume", &BBox::volume)
		.def("toLocal", [](const BBox &bbox, pyzombie::Vector<DIM> &p) -> pyzombie::Vector<DIM> {
			return pyzombie::Vector<DIM>((p - bbox.pMin).array() / bbox.extent().array());
		});
};

template<int DIM>
void bind_MeshGeometry(py::module_ &m, const char *typestr) {
    using MeshGeometry = pyzombie::MeshGeometry<DIM>;
    py::class_<MeshGeometry, std::shared_ptr<MeshGeometry>>(m, typestr)
        .def(py::init<fcpw::BoundingBox<DIM>, 
					  bool, 
					  bool, 
					  bool, 
					  bool,
					  bool,
					  bool,
					  std::string>(),
			 py::arg("bbox"),
			 py::arg("bboxIsDirichlet") = bool(false),
			 py::arg("domainIsWatertight") = bool(false),
			 py::arg("buildBVH") = bool(true),
			 py::arg("computeWeightedNormals") = bool(true),
			 py::arg("computeSilhouettes") = bool(true),
			 py::arg("flipOrientation") = bool(false),
			 py::arg("id") = "d")
		.def_readwrite("id", &MeshGeometry::id)
		.def_readwrite("queries", &MeshGeometry::queries)
		.def_readonly("bbox", &MeshGeometry::bbox)
		.def_readonly("dirichletBBox", &MeshGeometry::dirichletBBox)
		.def_readonly("neumannBBox", &MeshGeometry::neumannBBox)
		.def("setDirichletDisplacement",
			 &MeshGeometry::setDirichletDisplacement,
			 py::arg("type") = pyzombie::MeshDisplacementType::None,
			 py::arg("params") = pyzombie::VectorXi::Zero(0)
		)
		.def("setDirichlet", 
			 &MeshGeometry::setDirichlet,
			 py::arg("position"),
			 py::arg("index"))
		.def("setNeumann", 
			 &MeshGeometry::setNeumann,
			 py::arg("position"),
			 py::arg("index"))
		.def("getDirichletPositions", 
			 &MeshGeometry::getDirichletPositions)
		.def("getDirichletIndices", 
			 &MeshGeometry::getDirichletIndices)
		.def("getNeumannPositions", 
			 &MeshGeometry::getNeumannPositions)
		.def("getNeumannIndices", 
			 &MeshGeometry::getNeumannIndices)
		.def("getNeumannNormals",
			 &MeshGeometry::getNeumannNormals);
}

void bind_ToastScene(py::module_ &m, const char *typestr) {
    using ToastScene = pyzombie::ToastScene;
    py::class_<ToastScene, std::shared_ptr<ToastScene>>(m, typestr)
        .def(py::init<float, 
		      		  float,
					  bool,
					  bool,
					  std::string>(),
			py::arg("absorption") = bool(true),
			py::arg("dirichletLineThickness") = float(1e-2),
			py::arg("buildBVH") = bool(true),
			py::arg("flipOrientation") = bool(false),
			py::arg("id") = "d")
		.def_readwrite("id", &ToastScene::id)
		.def_readwrite("queries", &ToastScene::queries)
		.def_readwrite("pde", &ToastScene::pde)
		.def_readonly("neumannBBox", &ToastScene::neumannBBox)
		.def("setDirichlet", 
			 &ToastScene::setDirichlet,
			 py::arg("position"),
			 py::arg("index"))
		.def("setNeumann", 
			 &ToastScene::setNeumann,
			 py::arg("position"),
			 py::arg("index"))
		.def("setTemperature",
			 &ToastScene::setTemperature,
			 py::arg("temperatureId"),
			 py::arg("temperature"))
		.def("getDirichletVertexPosition",
			 &ToastScene::getDirichletVertexPosition,
			 py::arg("pIndex"),
			 py::arg("vLocalIndex"));
}

void bind_BezierInteraction(py::module_ &m, const char *typestr) {
	using BezierInteraction = pyzombie::BezierInteraction;
	py::class_<BezierInteraction>(m, typestr)
		.def(py::init<>())
		.def_readwrite("p", &BezierInteraction::p)
		.def_readwrite("n", &BezierInteraction::n)
		.def_readwrite("d", &BezierInteraction::d)
		.def_readwrite("sign", &BezierInteraction::sign)
		.def_readwrite("s", &BezierInteraction::s)
		.def_readwrite("isBbox", &BezierInteraction::isBbox);
}

void bind_BezierPoint(py::module_ &m, const char *typestr) {
	using BezierPoint = pyzombie::BezierPoint;
	py::class_<BezierPoint>(m, typestr)
		.def(py::init<pyzombie::Vector2, 
					  pyzombie::Vector2, 
					  float, 
					  float>())
		.def_readwrite("pt", &BezierPoint::pt)
		.def_readwrite("prev", &BezierPoint::prev)
		.def_readwrite("next", &BezierPoint::next);
}

void bind_Bezier(py::module_ &m, const char *typestr) {
	using Bezier = pyzombie::Bezier;
	py::class_<Bezier>(m, typestr)
		.def("create", [](const pyzombie::VectorWrapper<pyzombie::BezierPoint> & pts, pyzombie::VectorI2 index) -> pyzombie::Bezier {
			return Bezier(pts.data, index);
		})
		.def("closestPoint", 
			 &Bezier::closestPoint,
			 py::arg("x"),
			 py::arg("interaction"),
			 py::arg("cpPrecision"),
			 py::arg("flipNormalOrientation"))
		.def("computeTextureDisplacement",
			 &Bezier::computeTextureDisplacement,
			 py::arg("id"),
			 py::arg("x"),
			 py::arg("s"))
		.def("evaluate",
			 &Bezier::evaluate,
			 py::arg("t"))
		.def("derivative",
			 &Bezier::derivative,
			 py::arg("t"),
			 py::arg("n"));
}

void bind_BezierScene(py::module_ &m, const char *typestr) {
    using BezierScene = pyzombie::BezierScene;
    py::class_<BezierScene, std::shared_ptr<BezierScene>>(m, typestr)
        .def(py::init<pyzombie::Vector2, 
					  pyzombie::Vector2,
					  bool>())
		.def_readwrite("pde", &BezierScene::pde)
		.def_readwrite("queries", &BezierScene::queries)
		.def_readwrite("id", &BezierScene::id)
		.def_readwrite("ignoreShapeDerivatives", &BezierScene::ignoreShapeDerivatives)
		.def_readwrite("ignoreTextureDerivatives", &BezierScene::ignoreTextureDerivatives)
		.def_readwrite("flipBezierNormalOrientation", &BezierScene::flipBezierNormalOrientation)
		.def_readwrite("flipBoundingBoxNormalOrientation", &BezierScene::flipBoundingBoxNormalOrientation)
		.def_readwrite("cpPrecision", &BezierScene::cpPrecision)
		.def_readwrite("backgroundColor", &BezierScene::backgroundColor)
		.def("setColor",
			 &BezierScene::setColor,
			 py::arg("color"))
		.def("setDoubleSidedColor",
			 &BezierScene::setDoubleSidedColor,
			 py::arg("color"))
		.def("setBezierPoints",
			 &BezierScene::setBezierPoints,
			 py::arg("point"),
			 py::arg("delta"),
			 py::arg("scale"))
		.def("setBeziers",
			 &BezierScene::setBeziers,
			 py::arg("index"))
		.def("closestPoint",
			 &BezierScene::closestPoint,
			 py::arg("x"),
			 py::arg("interaction"))
		.def("samplePoints", 
			 [](const BezierScene& bezierScene, 
		   		int nPointsPerBezier) -> pyzombie::VectorWrapper<pyzombie::BoundarySamplePoint<pyzombie::RGB, 2>> {
				pyzombie::VectorWrapper<pyzombie::BoundarySamplePoint<pyzombie::RGB,2>> samplePts;
				samplePts.data = bezierScene.samplePoints(nPointsPerBezier);
				return samplePts;
			}
		)
		.def("computeLength", 
			 &BezierScene::computeLength,
			 py::arg("nPointsPerBezier"));
}

template <typename T, int DIM>
void bind_SamplePoint(py::module_ &m, const char *typestr) {
	using SamplePoint = zombie::SamplePoint<T, DIM>;
	py::class_<SamplePoint, std::shared_ptr<SamplePoint>>(m, typestr)
		.def(py::init<const pyzombie::Vector<DIM>&,
					  const pyzombie::Vector<DIM>&,
					  zombie::SampleType,
					  float,
					  float,
					  float,
					  T>())
		.def("reset",
			 &SamplePoint::reset,
			 py::arg("initVal"))
		.def("setRNG", 
			 &SamplePoint::setRNG,
			py::arg("seed")
		)
		.def("getSamplerState", &SamplePoint::getSamplerState)
		.def("getSamplerInc", &SamplePoint::getSamplerInc)
		.def_readwrite("pt", &SamplePoint::pt)
		.def_readwrite("normal", &SamplePoint::normal)
		.def_readwrite("type", &SamplePoint::type)
		.def_readwrite("pdf", &SamplePoint::pdf)
		.def_readwrite("dirichletDist", &SamplePoint::dirichletDist)
		.def_readwrite("neumannDist", &SamplePoint::neumannDist)
		.def_readwrite("firstSphereRadius", &SamplePoint::firstSphereRadius)
		.def_readwrite("estimateBoundaryNormalAligned", &SamplePoint::estimateBoundaryNormalAligned)
		.def_readwrite("statistics", &SamplePoint::statistics)
		.def_readwrite("solution", &SamplePoint::solution)
		.def_readwrite("normalDerivative", &SamplePoint::normalDerivative)
		.def_readwrite("source", &SamplePoint::source);
}

template <typename T, int DIM>
void bind_BoundarySamplePoint(py::module_ &m, const char *typestr) {
	using BoundarySamplePoint = pyzombie::BoundarySamplePoint<T, DIM>;
	py::class_<BoundarySamplePoint>(m, typestr)
		.def(py::init<>())
		.def_readwrite("pt", &BoundarySamplePoint::pt)
		.def_readwrite("n", &BoundarySamplePoint::n)
		.def_readwrite("vn", &BoundarySamplePoint::vn)
		.def_readwrite("u", &BoundarySamplePoint::u)
		.def_readwrite("uExterior", &BoundarySamplePoint::uExterior)
		.def_readwrite("pdf", &BoundarySamplePoint::pdf);
}

template<typename T, int DIM>
void bind_SampleStatistics(py::module_ &m, const char *typestr) {
	using SampleStatistics = zombie::SampleStatistics<T, DIM>;
	py::class_<SampleStatistics, std::shared_ptr<SampleStatistics>>(m, typestr)
		.def(py::init<T>())
		.def("reset", 
			 &SampleStatistics::reset,
			 py::arg("initVal"))
		.def("getEstimatedSolution",
			 &SampleStatistics::getEstimatedSolution)
		.def("getEstimatedDerivative",
			 &SampleStatistics::getEstimatedDerivative)	
		.def("getEstimatedGradient", 
			 &SampleStatistics::getEstimatedGradient)
		.def("getEstimatedDifferential",
			 &SampleStatistics::getEstimatedDifferential)
		.def("getEstimatedDerivativeDifferential",
			 &SampleStatistics::getEstimatedDerivative)
		.def("getEstimatedGradientDifferential",
			 &SampleStatistics::getEstimatedGradientDifferential)
		.def("getEstimatedSolutionWeightedDifferential",
			 &SampleStatistics::getEstimatedSolutionWeightedDifferential)
		.def("getMeanWalkLength",
			 &SampleStatistics::getMeanWalkLength);
}

template<int DIM>
void bind_SampleEstimationData(py::module_ &m, const char* typestr) {
	using SampleEstimationData = zombie::SampleEstimationData<DIM>;
	py::class_<SampleEstimationData, std::shared_ptr<SampleEstimationData>>(m, typestr)
		.def(py::init<>())
		.def(py::init<int, 
					  zombie::EstimationQuantity,
					  pyzombie::Vector<DIM>>())
		.def(py::init<int, 
					  int,
					  zombie::EstimationQuantity,
					  pyzombie::Vector<DIM>>())
		.def_readwrite("nWalks", &SampleEstimationData::nWalks)
		.def_readwrite("nRecursiveWalks", &SampleEstimationData::nWalks)
		.def_readwrite("estimationQuantity", &SampleEstimationData::estimationQuantity)
		.def_readwrite("directionForDerivative", &SampleEstimationData::directionForDerivative);
}

template <typename T>
void bind_WalkSettings(py::module_ &m, const char* typestr) {
	using WalkSettings = zombie::WalkSettings<T>;
	py::class_<WalkSettings, std::shared_ptr<WalkSettings>>(m, typestr)
		.def(py::init<T, float, float, int, bool>())
		.def_readwrite("initVal", &WalkSettings::initVal)
		.def_readwrite("epsilonShell", &WalkSettings::epsilonShell)
		.def_readwrite("minStarRadius", &WalkSettings::minStarRadius)
		.def_readwrite("silhouettePrecision", &WalkSettings::silhouettePrecision)
		.def_readwrite("russianRouletteThreshold", &WalkSettings::russianRouletteThreshold)
		.def_readwrite("boundaryGradientOffset", &WalkSettings::boundaryGradientOffset)
		.def_readwrite("maxWalkLength", &WalkSettings::maxWalkLength)
		.def_readwrite("solutionWeightedDifferentialBatchSize", &WalkSettings::solutionWeightedDifferentialBatchSize)
		.def_readwrite("stepsBeforeApplyingTikhonov", &WalkSettings::stepsBeforeApplyingTikhonov)
		.def_readwrite("solveDoubleSided", &WalkSettings::solveDoubleSided)
		.def_readwrite("useGradientControlVariates", &WalkSettings::useGradientControlVariates)
		.def_readwrite("useGradientAntitheticVariates", &WalkSettings::useGradientAntitheticVariates)
		.def_readwrite("useCosineSamplingForDerivatives", &WalkSettings::useCosineSamplingForDerivatives)
		.def_readwrite("useFiniteDifferences", &WalkSettings::useFiniteDifferences)
		.def_readwrite("ignoreDirichletContribution", &WalkSettings::ignoreDirichletContribution)
		.def_readwrite("ignoreNeumannContribution", &WalkSettings::ignoreNeumannContribution)
		.def_readwrite("ignoreSourceContribution", &WalkSettings::ignoreSourceContribution)
		.def_readwrite("ignoreShapeDifferential", &WalkSettings::ignoreShapeDifferential)
		.def_readwrite("printLogs", &WalkSettings::printLogs);
}

template<int DIM>
void bind_GeometricQueries(py::module_ &m, const char* typestr) {
	using Queries = zombie::GeometricQueries<DIM>;
	py::class_<Queries,  std::shared_ptr<Queries>>(m, typestr)
		.def_readwrite("computeDistToDirichlet", &Queries::computeDistToDirichlet)
		.def_readwrite("computeDistToNeumann", &Queries::computeDistToNeumann)
		.def_readwrite("computeDistToBoundary", &Queries::computeDistToBoundary)
		.def_readwrite("projectToDirichlet", &Queries::projectToDirichlet)
		.def_readwrite("projectToNeumann", &Queries::projectToNeumann)
		.def_readwrite("projectToBoundary", &Queries::projectToBoundary)
		.def_readwrite("insideDomain", &Queries::insideDomain)
		.def("computeDirichletDisplacement", [](const Queries &queries, const zombie::Vector<DIM> &x, const zombie::Vector<DIM> &n) {
			return zombie::dot<DIM>(queries.computeDirichletDisplacement(x),  n);
		});
}

template <typename T, int DIM>
void bind_PDE(py::module_ &m,  const char* typestr) {
	using PDE = zombie::PDE<T, DIM>;
	py::class_<PDE>(m, typestr)
		.def(py::init<>())
		.def_readwrite("absorption", &PDE::absorption)
		.def_readwrite("dirichlet", &PDE::dirichlet)
		.def_readwrite("neumann", &PDE::neumann)
		.def_readwrite("source", &PDE::source)
		.def_readwrite("dirichletDoubleSided", &PDE::dirichletDoubleSided)
		.def_readwrite("neumannDoubleSided", &PDE::neumannDoubleSided);
}

template <typename T, int DIM>
void bind_Grid(py::module_ &m, const char* typestr) {
	using Grid = pyzombie::Grid<T, DIM>;
	py::class_<Grid>(m, typestr)
		.def(py::init<pyzombie::Vector<DIM>,
					  pyzombie::Vector<DIM>,
					  std::string,
					  bool>())
		.def_readwrite("differentialEnabled", &Grid::differentialEnabled)
		.def_readwrite("id", &Grid::id)
		.def("setFromFunction",
			 &Grid::setFromFunction,
			 py::arg("dims"),
			 py::arg("f"))
		.def("setFromArray",
			 &Grid::setFromArray,
			 py::arg("values"))
		.def("getArray",
			 &Grid::getArray)
		.def("__call__",
			 &Grid::interpolate,
			 py::arg("x"))
		.def("interpolate",
			 &Grid::interpolate,
			 py::arg("x"));
}

template <typename T, int DIM>
void bind_UtilSetZeroBoundaryConditions(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setZeroBoundaryConditions<T, DIM>,
		  py::arg("pde"),
		  py::arg("initVal"));
}

template <typename T, int DIM>
void bind_UtilSetConstantDirichletBoundaryConditions(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setConstantDirichletBoundaryConditions<T, DIM>,
		  py::arg("pde"),
		  py::arg("g"),
		  py::arg("zero"),
		  py::arg("bbox"));
}

template <typename T, int DIM>
void bind_UtilSetConstantSource(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setConstantSource<T, DIM>,
		  py::arg("pde"),
		  py::arg("g"),
		  py::arg("zero"));
}

void bind_UtilSetAnalyticDirichletBoundaryCondition(py::module_ &m, const char* typestr) {
	m.def((typestr + std::string("A")).c_str(), [](zombie::PDE<float, 3> &pde, fcpw::BoundingBox<3> &bbox) -> void {
		pde.dirichlet = [&bbox](const pyzombie::Vector3 &p) -> zombie::Value<float, 3> {
			return bbox.contains(p) ? pyzombie::dirichletA(p) : zombie::Value<float, 3>(0.0f);
		};
		pde.dirichletDoubleSided = [&bbox](const pyzombie::Vector3 &p, bool _) -> zombie::Value<float, 3> {
			return bbox.contains(p) ?  pyzombie::dirichletA(p) : zombie::Value<float, 3>(0.0f);
		};
	});
	m.def((typestr + std::string("B")).c_str(), [](zombie::PDE<float, 3> &pde, fcpw::BoundingBox<3> &bbox) -> void {
		pde.dirichlet = [&bbox](const pyzombie::Vector3 &p) -> zombie::Value<float, 3> {
			return bbox.contains(p) ? pyzombie::dirichletB(p) : zombie::Value<float, 3>(0.0f);
		};
		pde.dirichletDoubleSided = [&bbox](const pyzombie::Vector3 &p, bool _) -> zombie::Value<float, 3> {
			return bbox.contains(p) ?  pyzombie::dirichletB(p) : zombie::Value<float, 3>(0.0f);
		};
	});
}

template <typename T, int DIM>
void bind_UtilSetDirichletGridBoundaryCondition(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setDirichletGridBoundaryCondition<T, DIM>,
		  py::arg("pde"),
		  py::arg("grid"));
}

template <typename T, int DIM>
void bind_UtilSetNeumannGridBoundaryCondition(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setNeumannGridBoundaryCondition<T, DIM>,
		  py::arg("pde"),
		  py::arg("grid"));
}

template <typename T, int DIM>
void bind_UtilSetSourceGridCondition(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::setSourceGridCondition<T, DIM>,
		  py::arg("pde"),
		  py::arg("grid"));
}

template <typename T, int DIM>
void bind_UtilCreateSamplePoint(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::createSamplePoint<T, DIM>,
		  py::arg("queries"),
		  py::arg("x"),
		  py::arg("pdf"),
		  py::arg("initVal"),
		  py::arg("inDomain") = true,
		  py::arg("estimateBoundaryNormalAligned") = false);
}

template <typename T, int DIM>
void bind_UtilCreateSamplePoints(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::createSamplePoints<T, DIM>,
		  py::arg("queries"),
		  py::arg("x"),
		  py::arg("pdf"),
		  py::arg("initVal"),
		  py::arg("inDomain") = true,
		  py::arg("estimateBoundaryNormalAligned") = false);
}

template <int DIM>
void bind_UtilCreateSampleEstimationData(py::module_ &m, const char* typestr) {
	m.def(typestr,
		  pyzombie::createSampleEstimationData<DIM>,
		  py::arg("nWalks"),
		  py::arg("nRecursiveWalks"),
		  py::arg("quantity"),
		  py::arg("n"));
}

template <typename T, int DIM>
void bind_UtilSolve(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::solve<T, DIM>,
		  py::arg("queries"),
		  py::arg("pde"),
		  py::arg("settings"),
		  py::arg("sampleEstimationData"),
		  py::arg("samplePoints"),
		  py::arg("runSingleThreaded"));
}

template <int DIM>
void bind_UtilComputeL2Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeL2Loss<DIM>,
		  py::arg("pts"),
		  py::arg("target"));
}

template <int DIM>
void bind_UtilComputeL1Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeL1Loss<DIM>,
		  py::arg("pts"),
		  py::arg("target"));
}

template <int DIM>
void bind_UtilComputeMaskedL1Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeMaskedL1Loss<DIM>,
		  py::arg("pts"),
		  py::arg("ests"),
		  py::arg("target"));
}

template <int DIM>
void bind_UtilComputeMaskedL2Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeMaskedL2Loss<DIM>,
		  py::arg("pts"),
		  py::arg("ests"),
		  py::arg("target"));
}

template <int DIM>
void bind_UtilComputeRelativeL2Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeRelativeL2Loss<DIM>,
		  py::arg("pts"),
		  py::arg("target"),
		  py::arg("epsilon"));
}

template <int DIM>
void bind_UtilComputeMaskedRelativeL2Loss(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeMaskedRelativeL2Loss<DIM>,
		  py::arg("pts"),
		  py::arg("ests"),
		  py::arg("target"),
		  py::arg("epsilon"));
}

template <int DIM>
void bind_UtilComputeL2LossRGB(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeL2LossRGB<DIM>,
		  py::arg("pts"),
		  py::arg("target"));
}
template <int DIM>
void bind_UtilComputeL2BoundaryLossRGB(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::computeL2BoundaryLossRGB<DIM>,
		  py::arg("pts"),
		  py::arg("target"),
		  py::arg("targetExterior"));
}

template <int DIM>
void bind_UtilGetSolution(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::getSolution<DIM>,
		  py::arg("pts"));
}

template <int DIM>
void bind_UtilGetSolutionRGB(py::module_ &m,  const char* typestr) {
	m.def(typestr, 
		  pyzombie::getSolutionRGB<DIM>,
		  py::arg("pts"));
}

// Note: any functions which aborts / terminates can cause code in jupyter notebooks
// to hand indefinitely, consider "handling unraisable exceptions"
namespace py = pybind11;
PYBIND11_MODULE(pyzombie, m) { 
	// =================== Native Zombie =================== 
	py::module_ typesSubmodule = m.def_submodule("types");
	py::enum_<zombie::EstimationQuantity>(typesSubmodule, "EstimationQuantity")
		.value("Solution", zombie::EstimationQuantity::Solution)
		.value("SolutionAndGradient", zombie::EstimationQuantity::SolutionAndGradient)
		.value("Nothing", zombie::EstimationQuantity::None)
		.export_values();
	
	py::enum_<zombie::SampleType>(typesSubmodule, "SampleType")
		.value("InDomain", zombie::SampleType::InDomain)
		.value("OnDirichletBoundary", zombie::SampleType::OnDirichletBoundary)
		.value("OnNeumannBoundary", zombie::SampleType::OnNeumannBoundary)
		.export_values();

	py::enum_<pyzombie::MeshDisplacementType>(typesSubmodule, "MeshDisplacementType")
		.value("VertexTranslation", pyzombie::MeshDisplacementType::VertexTranslation)
		.value("Translation", pyzombie::MeshDisplacementType::Translation)
		.value("Affine", pyzombie::MeshDisplacementType::Affine)
		.value("EulerRodrigues", pyzombie::MeshDisplacementType::EulerRodrigues)
		.value("Nothing", pyzombie::MeshDisplacementType::None)
		.export_values();

	bind_Differential<float>(m, "Differential");
	bind_Differential<pyzombie::RGB>(m, "DifferentialRGB");

	bind_WalkSettings<float>(m, "WalkSettings");
	bind_WalkSettings<pyzombie::RGB>(m, "WalkSettingsRGB");

	bind_BBox<2>(m, "BBox2D");
	bind_BBox<3>(m, "BBox");

	bind_GeometricQueries<2>(m, "GeometricQueries2D");
	bind_GeometricQueries<3>(m, "GeometricQueries");
	
	bind_PDE<float, 2>(m, "PDE2D");
	bind_PDE<float, 3>(m, "PDE");
	bind_PDE<pyzombie::RGB, 2>(m, "PDE2DRGB");
	bind_PDE<pyzombie::RGB, 3>(m, "PDERGB");

	bind_SpatialGradient<float, 2>(m, "SpatialGradient2D");
	bind_SpatialGradient<float, 3>(m, "SpatialGradient");
	bind_SpatialGradient<pyzombie::RGB, 2>(m, "SpatialGradient2DRGB");
	bind_SpatialGradient<pyzombie::RGB, 3>(m, "SpatialGradientRGB");

	bind_GradientDifferential<float, 2>(m, "GradientDifferential2D");
	bind_GradientDifferential<float, 3>(m, "GradientDifferential");
	bind_GradientDifferential<pyzombie::RGB, 2>(m, "GradientDifferential2DRGB");
	bind_GradientDifferential<pyzombie::RGB, 3>(m, "GradientDifferentialRGB");

	bind_SamplePoint<float, 2>(m, "SamplePoint2D");
	bind_SamplePoint<float, 3>(m, "SamplePoint");
	bind_SamplePoint<pyzombie::RGB, 2>(m, "SamplePoint2DRGB");
	bind_SamplePoint<pyzombie::RGB, 3>(m, "SamplePointRGB");
	bind_BoundarySamplePoint<pyzombie::RGB, 2>(m, "BoundarySamplePoint2DRGB");

	bind_SampleStatistics<float, 2>(m, "SampleStatistics2D");	
	bind_SampleStatistics<float, 3>(m, "SampleStatistics");	
	bind_SampleStatistics<pyzombie::RGB, 2>(m, "SampleStatistics2DRGB");	
	bind_SampleStatistics<pyzombie::RGB, 3>(m, "SampleStatisticsRGB");	

	bind_SampleEstimationData<2>(m, "SampleEstimationData2D");
	bind_SampleEstimationData<3>(m, "SampleEstimationData");
	
	bind_Value<float, 2>(m, "Value2D");
	bind_Value<float, 3>(m, "Value");
	bind_Value<pyzombie::RGB, 2>(m, "Value2DRGB");
	bind_Value<pyzombie::RGB, 3>(m, "ValueRGB");

	// =================== PyZombie Core =================== 
	bind_Grid<float, 2>(m, "Grid2D");
	bind_Grid<float, 3>(m, "Grid");
		
	bind_Vector<zombie::SamplePoint<float, 2>>(m, "VectorSamplePoint2D");
	bind_Vector<zombie::SamplePoint<float, 3>>(m, "VectorSamplePoint");
	bind_Vector<zombie::SamplePoint<pyzombie::RGB, 2>>(m, "VectorSamplePoint2DRGB");
	bind_Vector<zombie::SamplePoint<pyzombie::RGB, 3>>(m, "VectorSamplePointRGB");
	bind_Vector<pyzombie::BoundarySamplePoint<pyzombie::RGB, 2>>(m, "VectorBoundarySamplePoint2DRGB");

	bind_Vector<zombie::SampleEstimationData<2>>(m, "VectorSampleEstimationData2D");
	bind_Vector<zombie::SampleEstimationData<3>>(m, "VectorSampleEstimationData");
	
	bind_MeshGeometry<2>(m, "MeshGeometry2D");
	bind_MeshGeometry<3>(m, "MeshGeometry");

	bind_Vector<pyzombie::BezierPoint>(m, "VectorBezierPoint2D");
	bind_BezierInteraction(m, "BezierInteraction2D");
	bind_BezierPoint(m, "BezierPoint2D");
	bind_Bezier(m, "Bezier2D");
	bind_BezierScene(m, "BezierScene2D");
	bind_ToastScene(m, "ToastScene");

	// =================== PyZombie Utilities =================== 
	bind_UtilSetZeroBoundaryConditions<float, 2>(m, "setZeroBoundaryConditions2D");
	bind_UtilSetZeroBoundaryConditions<float, 3>(m, "setZeroBoundaryConditions");
	
	bind_UtilSetConstantDirichletBoundaryConditions<float, 2>(m, "setConstantDirichletBoundaryConditions2D");
	bind_UtilSetConstantDirichletBoundaryConditions<float, 3>(m, "setConstantDirichletBoundaryConditions");

	bind_UtilSetConstantSource<float, 2>(m, "setConstantSource2D");
	bind_UtilSetConstantSource<float, 3>(m, "setConstantSource");

	bind_UtilSetDirichletGridBoundaryCondition<float, 2>(m, "setDirichletGridBoundaryCondition2D");
	bind_UtilSetDirichletGridBoundaryCondition<float, 3>(m, "setDirichletGridBoundaryCondition");

	bind_UtilSetNeumannGridBoundaryCondition<float, 2>(m, "setNeumannGridBoundaryCondition2D");
	bind_UtilSetNeumannGridBoundaryCondition<float, 3>(m, "setNeumannGridBoundaryCondition");

	bind_UtilSetSourceGridCondition<float, 2>(m, "setSourceGridCondition2D");
	bind_UtilSetSourceGridCondition<float, 3>(m, "setSourceGridCondition");

	bind_UtilSetAnalyticDirichletBoundaryCondition(m, "setAnalyticDirichletBoundaryCondition");

	bind_UtilCreateSamplePoint<float, 2>(m, "createSamplePoint2D");
	bind_UtilCreateSamplePoint<float, 3>(m, "createSamplePoint");
	bind_UtilCreateSamplePoint<pyzombie::RGB, 2>(m, "createSamplePoint2DRGB");
	bind_UtilCreateSamplePoint<pyzombie::RGB, 3>(m, "createSamplePointRGB");

	bind_UtilCreateSamplePoints<float, 2>(m, "createSamplePoints2D");
	bind_UtilCreateSamplePoints<float, 3>(m, "createSamplePoints");
	bind_UtilCreateSamplePoints<pyzombie::RGB, 2>(m, "createSamplePoints2DRGB");
	bind_UtilCreateSamplePoints<pyzombie::RGB, 3>(m, "createSamplePointsRGB");

	bind_UtilCreateSampleEstimationData<2>(m, "createSampleEstimationData2D");
	bind_UtilCreateSampleEstimationData<3>(m, "createSampleEstimationData");
	
	bind_UtilSolve<float, 2>(m, "solve2D");
	bind_UtilSolve<float, 3>(m, "solve");
	bind_UtilSolve<pyzombie::RGB, 2>(m, "solve2DRGB");
	bind_UtilSolve<pyzombie::RGB, 3>(m, "solveRGB");
	
	bind_UtilComputeL2Loss<2>(m, "l2Loss2D");
	bind_UtilComputeL2Loss<3>(m, "l2Loss");
	bind_UtilComputeL1Loss<3>(m, "l1Loss");
	bind_UtilComputeMaskedL1Loss<3>(m, "maskedL1Loss");
	bind_UtilComputeMaskedL2Loss<3>(m, "maskedL2Loss");
	bind_UtilComputeMaskedRelativeL2Loss<3>(m, "maskedRelativeL2Loss");

	bind_UtilComputeRelativeL2Loss<2>(m, "relativeL2Loss2D");
	bind_UtilComputeRelativeL2Loss<3>(m, "relativeL2Loss");
	bind_UtilComputeL2LossRGB<2>(m, "l2Loss2DRGB");
	bind_UtilComputeL2LossRGB<3>(m, "l2LossRGB");
	bind_UtilComputeL2BoundaryLossRGB<2>(m, "l2BoundaryLoss2DRGB");

	bind_UtilGetSolution<2>(m, "getSolution2D");
	bind_UtilGetSolution<3>(m, "getSolution");
	bind_UtilGetSolutionRGB<2>(m, "getSolution2DRGB");
	bind_UtilGetSolutionRGB<3>(m, "getSolutionRGB");
	m.def("generateSeed", zombie::generateSeed);
};

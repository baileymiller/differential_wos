#pragma once

#include "pyzombie/common.h"
#include "pyzombie/grid.h"

namespace py = pybind11;

#define GEOMETRY_BBOX_SCALE_FACTOR 1.001

// bbox for fcpw queries vs actual bbox mesh.
namespace pyzombie {

class ToastScene {
public:
	std::vector<int> temperatureId;
	std::vector<float> temperature;
	zombie::PDE<float, 3> pde;
	zombie::GeometricQueries<3> queries;
	std::string id = "d";
	fcpw::BoundingBox<3> neumannBBox;
	
	ToastScene(float absorption = 5.0,
               float dirichletLineThickness = 1e-2,
	           bool buildBVH = true,
		   	   bool flipOrientation = false,
		   	   std::string id = "d"): 
	        absorption(absorption),
		   	dirichletLineThickness(dirichletLineThickness),
		   	buildBVH(buildBVH),
		   	flipOrientation(flipOrientation),
		   	id(id),
		   	queries(false) {
		 		bbox.pMin = Vector3::Constant(fcpw::minFloat);
				bbox.pMax = Vector3::Constant(fcpw::maxFloat);
				populatePDE();
		   };

	void setDirichlet(const py::array_t<float> &position, const py::array_t<int> &index) {
		setPolylineGeometry(dirichlet, position, index);
		registerDirichletUpdate();
	}

	void setNeumann(const py::array_t<float> &position, const py::array_t<int> &index) {
		neumannBBox = fcpw::BoundingBox<3>();
		setTriangleMeshGeometry(neumann, position, index, neumannBBox);
		registerNeumannUpdate();
	}

	void setTemperature(const py::array_t<int> &temperatureId, const py::array_t<float> &temperature) {
		// WARNING: only well defined if temperature is constant for connected polylines.
		py::buffer_info temperatureBuffer = temperature.request();
		if (temperatureBuffer.ndim != 1) {
			throw std::runtime_error("Error: temperature buffer should have dimension N");
		}
		this->temperature.clear();
		auto temp = temperature.unchecked<1>();
		for (int i = 0; i < temp.shape(0); i++) {
			this->temperature.emplace_back(temp(i));
		}

		py::buffer_info temperatureIdBuffer = temperatureId.request();
		if (temperatureIdBuffer.ndim != 1) {
			throw std::runtime_error("Error: temperatureId buffer should have dimension N");
		}
		this->temperatureId.clear();
		auto tempId = temperatureId.unchecked<1>();
		for (int i = 0; i < tempId.shape(0); i++) {
			this->temperatureId.emplace_back(tempId(i));
		}
	}

	const Vector3& getDirichletVertexPosition(int pIndex, int vLocalIndex) {
		fcpw::SceneData<3>* sceneData = this->dirichlet.getSceneData();
		if (sceneData->soups.size() == 0) {
			throw std::runtime_error("Scene::getDirichletVertexPosition No soups found");
		}
		int vIndex = sceneData->soups[0].indices[2 * pIndex + vLocalIndex];
		return sceneData->soups[0].positions[vIndex];
	}

private:
	bool buildBVH;
	bool flipOrientation;
	float absorption;
	float dirichletLineThickness;
	fcpw::BoundingBox<3> bbox;
	fcpw::Scene<3> dirichlet;
	fcpw::Scene<3> neumann;
	std::function<float(float)> neumannSamplingTraversalWeight = [](float x) -> float { return 1.0f; };

	const fcpw::Aggregate<3>* getAggregate(fcpw::Scene<3> &geometry) {
		fcpw::SceneData<3> *sceneData = geometry.getSceneData();
		return sceneData->soups.size() > 0 ? sceneData->aggregateInstancePtrs[0] : nullptr;
	}		

	void registerDirichletUpdate() {
		fcpw::AggregateType aggregateType = buildBVH ?
											fcpw::AggregateType::Bvh_SurfaceArea :
											fcpw::AggregateType::Baseline;
		dirichlet.computeObjectNormals(0, false);
		dirichlet.build(aggregateType, true, false, false);
		populateGeometricQueries(queries, bbox,
								 getAggregate(dirichlet), getAggregate(neumann),
								 neumannSamplingTraversalWeight, 
								 true, dirichletLineThickness);
		setDirichletDisplacement();
	}

	void registerNeumannUpdate(bool computeWeightedNormals = false) {
		fcpw::AggregateType aggregateType = buildBVH ?
											fcpw::AggregateType::Bvh_SurfaceArea :
											fcpw::AggregateType::Baseline;
		neumann.computeObjectNormals(0, true);
		neumann.computeSilhouettes({});
		neumann.build(aggregateType, true, false, false);
		populateGeometricQueries(queries, bbox,
								 getAggregate(dirichlet), getAggregate(neumann),
								 neumannSamplingTraversalWeight,
								 true, dirichletLineThickness);
		setDirichletDisplacement();
	}

	void populateGeometricQueries(zombie::GeometricQueries<3>& geometricQueries,
							      const fcpw::BoundingBox<3>& boundingBox,
							      const fcpw::Aggregate<3> *dirichletAggregate,
							      const fcpw::Aggregate<3> *neumannAggregate,
							      const std::function<float(float)>& neumannSamplingTraversalWeight,
							      bool useDirichletLineBoundary = false,
							      float dirichletBoundaryThickness = 1e-3f)	 {
		// first populate with standard zombie queries
		zombie::populateGeometricQueries<3>(queries, bbox,
											dirichletAggregate, neumannAggregate,
											neumannSamplingTraversalWeight);

		// override dirichlet distance / projection queries to suppport wires with thickness
		geometricQueries.computeDistToDirichlet = [dirichletAggregate, &boundingBox,
	                                               useDirichletLineBoundary, dirichletBoundaryThickness](
											       const Vector3& x, bool computeSignedDistance) -> float {
			if (dirichletAggregate != nullptr) {
				Vector3 queryPt = Vector3::Zero();
				queryPt = x;

				fcpw::Interaction<3> interaction;
				fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
				dirichletAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);
				return std::max(interaction.d - dirichletBoundaryThickness, 0.0f);
			}

			float d2Min, d2Max;
			boundingBox.computeSquaredDistance(x, d2Min, d2Max);
			return std::sqrt(d2Max);
		};

		geometricQueries.projectToDirichlet = [dirichletAggregate, useDirichletLineBoundary, 
	                                      	   dirichletBoundaryThickness](Vector<3>& x, Vector<3>& normal,
															   	      	   float& distance, bool computeSignedDistance) -> bool {
			if (dirichletAggregate != nullptr) {
				Vector3 queryPt = Vector3::Zero();
				queryPt = x;

				fcpw::Interaction<3> interaction;
				fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
				dirichletAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);
				normal = (Vector3(x - interaction.p)).normalized();
				x = interaction.p + normal * dirichletBoundaryThickness;
				distance = std::max(interaction.d - dirichletBoundaryThickness, 0.0f);
				return true;
			}

			distance = 0.0f;
			return false;
		};
		geometricQueries.projectToBoundary = [&geometricQueries, 
											  dirichletAggregate, neumannAggregate, 
											  dirichletBoundaryThickness](Vector3& x, 
											  							  Vector3& normal, 
																		  float& distance,
										      							  bool& projectToDirichlet, 
																		  bool computeSignedDistance) -> bool {
			if (dirichletAggregate != nullptr && neumannAggregate != nullptr) {
				Vector3 queryPt = Vector3::Zero();
				queryPt = x;

				fcpw::Interaction<3> interactionDirichlet;
				fcpw::BoundingSphere<3> sphereDirichlet(queryPt, fcpw::maxFloat);
				dirichletAggregate->findClosestPoint(sphereDirichlet, interactionDirichlet, computeSignedDistance);
				interactionDirichlet.d = std::max(interactionDirichlet.d - dirichletBoundaryThickness, 0.0f);
				interactionDirichlet.n = (x - interactionDirichlet.p).normalized();
				interactionDirichlet.p = interactionDirichlet.p + interactionDirichlet.n * dirichletBoundaryThickness;

				fcpw::Interaction<3> interactionNeumann;
				fcpw::BoundingSphere<3> sphereNeumann(queryPt, fcpw::maxFloat);
				neumannAggregate->findClosestPoint(sphereNeumann, interactionNeumann, computeSignedDistance);

				if (interactionDirichlet.d < interactionNeumann.d) {
					x = interactionDirichlet.p;
					normal = interactionDirichlet.n;
					distance = interactionDirichlet.d;
					projectToDirichlet = true;
				} else {
					x = interactionNeumann.p;
					normal = interactionNeumann.n;
					distance = computeSignedDistance ? interactionNeumann.signedDistance(queryPt) : interactionNeumann.d;
					projectToDirichlet = false;
				}

				return true;
			}

			if (geometricQueries.projectToDirichlet(x, normal, distance, computeSignedDistance)) {
				projectToDirichlet = true;
				return true;

			} else if (geometricQueries.projectToNeumann(x, normal, distance, computeSignedDistance)) {
				projectToDirichlet = false;
				return true;
			}

			return false;
		};
	}

	void populatePDE() {	
		pde.absorption = absorption;
		pde.dirichlet = [this](Vector3 x) -> zombie::Value<float, 3> {
			const fcpw::Aggregate<3>* dirichletAggregate = this->getAggregate(this->dirichlet);
			zombie::Value<float, 3> g(0.0f);
			if (dirichletAggregate == nullptr) return g;
			
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(3) = x;
			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			dirichletAggregate->findClosestPoint(sphere, interaction, true);

			int pIndex = interaction.primitiveIndex;
			if (pIndex >= temperatureId.size()) {
				throw std::runtime_error("primitive out of range: " + std::to_string(pIndex) + " > " + std::to_string(temperatureId.size()));
			}

			int temperatureIndex = this->temperatureId[pIndex];
			if (temperatureIndex >= temperature.size()) {
				throw std::runtime_error("temp index out of range: " + std::to_string(temperatureIndex) + " > " + std::to_string(temperature.size()));
			}

			g.data = this->temperature[temperatureIndex];
			g.differential.ref(id + "#temperature." + std::to_string(temperatureIndex)) = 1.0f;
			return g;
		};
		pde.dirichletDoubleSided = [this](const Vector3& x, bool boundaryNormalAligned) -> zombie::Value<float, 3> {
			return this->pde.dirichlet(x);
		};
		pde.neumann = [](const Vector3& x) -> zombie::Value<float, 3> {
			return zombie::Value<float, 3>(0.0f);
		};
		pde.neumannDoubleSided = [this](const Vector3& x, bool boundaryNormalAligned) -> zombie::Value<float, 3> {
			return this->pde.neumann(x);
		};
		pde.source = [](const Vector3 &x) -> zombie::Value<float, 3> {
			return 0.0f;
		};
	}

	void setDirichletDisplacement() {	
		queries.computeDirichletDisplacement = [this](const Vector3& x) -> zombie::Differential<Vector3> {
			zombie::Differential<Vector3> v(Vector3::Zero());
			const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
			if (dirichletAggregate == nullptr) return v;

			Vector3 queryPt = Vector3::Zero();
			queryPt.head(3) = x;
		
			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			dirichletAggregate->findClosestPoint(sphere, interaction, true);
			Vector3 projX = interaction.p;
			int pIndex = interaction.primitiveIndex;
			const Vector3& v0 = this->getDirichletVertexPosition(pIndex, 0);
			const Vector3& v1 = this->getDirichletVertexPosition(pIndex, 1);

			float length = (v1 - v0).norm();
			if (length < 1e-8) return v; 	// degenerate vertex, no derivative
			float t1 = std::clamp((projX - v0).norm() / length, 0.0f, 1.0f);
			float t0 = 1.0 - t1;
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".0.0") = { t0, 0, 0};
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".0.1") = { 0, t0, 0};
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".0.2") = { 0, 0, t0};
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".1.0") = { t1, 0, 0 };
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".1.1") = { 0, t1, 0 };
			v.ref(this->id + "#geometry." + std::to_string(pIndex) + ".1.2") = { 0, 0, t1 };
			return v;
		};
	}

	void setPolylineGeometry(fcpw::Scene<3> &geometry, 
							 const py::array_t<float> &position, 
							 const py::array_t<int> &index) {
		py::buffer_info positionBuffer = position.request();
		if (positionBuffer.ndim != 2 || positionBuffer.shape[1] != 3) {
			throw std::runtime_error("Error: Position array should have dimensions Nx3");
		}

		py::buffer_info indexBuffer = index.request();
		if (indexBuffer.ndim != 2 || indexBuffer.shape[1] != 2) {
			throw std::runtime_error("Error: Indices array should have dimensions Nx2");
		}

		std::vector<std::vector<fcpw::PrimitiveType>> objectTypes(1, std::vector<fcpw::PrimitiveType>{fcpw::PrimitiveType::LineSegment});
		geometry.setObjectTypes(objectTypes);

		int V = positionBuffer.shape[0];
		geometry.setObjectVertexCount(V, 0);
		
		int L = indexBuffer.shape[0];
		geometry.setObjectLineSegmentCount(L, 0);

		float* positionData = static_cast<float*>(positionBuffer.ptr);
		for (int i = 0; i < V; ++i) {
			Vector3 p = Vector3::Zero();
			p[0] = float(positionData[i * 3 + 0]);
			p[1] = float(positionData[i * 3 + 1]);
			p[2] = float(positionData[i * 3 + 2]);
			geometry.setObjectVertex(p, i, 0);
		}
		
		int* indexData = static_cast<int*>(indexBuffer.ptr);
		for (int i = 0; i < L; ++i) {
			int lineSegment[2] = {(int)indexData[i * 2 + 0], (int)indexData[i * 2 + 1]};
			if (flipOrientation) std::swap(lineSegment[0], lineSegment[1]);
			if (lineSegment[0] >= V || lineSegment[0] < 0 || lineSegment[1] >= V || lineSegment[1] < 0) {
				std::stringstream errMsg;
				errMsg << "Line segment " << i;
				errMsg << " has an index out of range. Indices = ";
				errMsg << lineSegment[0] << ", " << lineSegment[1] << ") and ";
				errMsg << " num verticess = " << V << std::endl;;
				throw std::runtime_error(errMsg.str());
			}
			geometry.setObjectLineSegment(lineSegment, i, 0);
		}
	}

	void setTriangleMeshGeometry(fcpw::Scene<3> &geometry, 
							     const py::array_t<float> &position, 
							     const py::array_t<int> &index, 
							     fcpw::BoundingBox<3> &geometryBBox) {
		py::buffer_info positionBuffer = position.request();
		if (positionBuffer.ndim != 2 || positionBuffer.shape[1] != 3) {
			throw std::runtime_error("Error: Position array should have dimensions Nx3");
		}

		py::buffer_info indexBuffer = index.request();
		if (indexBuffer.ndim != 2 || indexBuffer.shape[1] != 3) {
			throw std::runtime_error("Error: Indices array should have dimensions Nx3");
		}

		std::vector<std::vector<fcpw::PrimitiveType>> objectTypes(1, std::vector<fcpw::PrimitiveType>{fcpw::PrimitiveType::Triangle});
		geometry.setObjectTypes(objectTypes);

		int V = positionBuffer.shape[0];
		geometry.setObjectVertexCount(V, 0);
		
		int T = indexBuffer.shape[0];
		geometry.setObjectTriangleCount(T, 0);

		float* positionData = static_cast<float*>(positionBuffer.ptr);
		for (int i = 0; i < V; ++i) {
			Vector3 p = Vector3::Zero();
			p[0] = float(positionData[i * 3]);
			p[1] = float(positionData[i * 3 + 1]);
			p[2] = float(positionData[i * 3 + 2]);
			geometryBBox.expandToInclude(p);
			geometry.setObjectVertex(p, i, 0);
		}

		Vector3 geometryBBoxCentroid = geometryBBox.centroid();
		geometryBBox.pMin = (geometryBBox.pMin - geometryBBoxCentroid) *  GEOMETRY_BBOX_SCALE_FACTOR + geometryBBoxCentroid;
		geometryBBox.pMax = (geometryBBox.pMax - geometryBBoxCentroid) *  GEOMETRY_BBOX_SCALE_FACTOR + geometryBBoxCentroid;
		
		int* indexData = static_cast<int*>(indexBuffer.ptr);
		for (int i = 0; i < T; ++i) {
			int triangle[3] = {
				(int)indexData[i * 3], 
				(int)indexData[i * 3 + 1], 
				(int)indexData[i * 3 + 2]
			};
			if (flipOrientation) std::swap(triangle[0], triangle[1]);
			if (triangle[0] >= V || triangle[0] < 0 || 
				triangle[1] >= V || triangle[1] < 0 ||
				triangle[2] >= V || triangle[2] < 0) {
				std::stringstream errMsg;
				errMsg << "Triangle " << i;
				errMsg << " has an index out of range. Indices = ";
				errMsg << triangle[0] << ", " << triangle[1] << ", " << triangle[2] << ") and ";
				errMsg << " num verticess = " << V << std::endl;;
				throw std::runtime_error(errMsg.str());
			}
			geometry.setObjectTriangle(triangle, i, 0);
		}
	}
};

}; // pyzombie

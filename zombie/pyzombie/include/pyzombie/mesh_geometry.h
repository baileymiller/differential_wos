#pragma once

#include "pyzombie/common.h"
#include "pyzombie/grid.h"

namespace py = pybind11;

#define GEOMETRY_BBOX_SCALE_FACTOR 1.001
#define BBOX_SCALE_FACTOR 0.999

// bbox for fcpw queries vs actual bbox mesh.
namespace pyzombie {

enum class MeshDisplacementType {
	VertexTranslation,
	Translation,
	Affine,
	EulerRodrigues,
	None
};

template <int DIM>
class MeshGeometry {
public:
	std::string id = "d";
	fcpw::BoundingBox<DIM> bbox;
	fcpw::BoundingBox<DIM> dirichletBBox;
	fcpw::BoundingBox<DIM> neumannBBox;
	zombie::GeometricQueries<DIM> queries;

	MeshGeometry(fcpw::BoundingBox<DIM> bbox,
		         bool bboxIsDirichlet = true,
		         bool domainIsWatertight = false,
		         bool buildBVH = true,
		         bool computeWeightedNormals = false,
		         bool computeSilhouettes = true,
				 bool flipOrientation = false,
				 std::string id = "d"): 
		         bbox(bbox),
		         bboxIsDirichlet(bboxIsDirichlet),
		         queries(domainIsWatertight),
		         buildBVH(buildBVH),
		         computeWeightedNormals(computeWeightedNormals),
		         computeSilhouettes(computeSilhouettes),
				 flipOrientation(flipOrientation),
				 id(id) {};

	void setDirichletDisplacement(const MeshDisplacementType &type = MeshDisplacementType::None,
								  const VectorXi &params = {}) {
		throw std::runtime_error("setDirichletDisplacement() not implemented for DIM=" + std::to_string(DIM));
	}

	void setDirichlet(const py::array_t<float> &position, const py::array_t<int> &index) {
		dirichletBBox = fcpw::BoundingBox<DIM>();
		setGeometry(dirichlet, position, index, dirichletBBox, bboxIsDirichlet);
		registerDirichletUpdate();
	}

	void setNeumann(const py::array_t<float> &position, const py::array_t<int> &index) {
		neumannBBox = fcpw::BoundingBox<DIM>();
		setGeometry(neumann, position, index, neumannBBox, !bboxIsDirichlet);
		registerNeumannUpdate();
	}

	py::array_t<float> getDirichletPositions() {
		return getPositions(dirichlet.getSceneData());
	}
	
	py::array_t<int> getDirichletIndices() {
		return getIndices(dirichlet.getSceneData());
	}

	py::array_t<float> getNeumannPositions() {
		return getPositions(neumann.getSceneData());
	}
	
	py::array_t<int> getNeumannIndices() {
		return getIndices(neumann.getSceneData());
	}

	py::array_t<float> getNeumannNormals() {
		return getNormals(neumann.getSceneData());
	}
	
private:
	bool bboxIsDirichlet;
	bool buildBVH;
	bool computeWeightedNormals;
	bool computeSilhouettes;
	bool flipOrientation;

	fcpw::Scene<3> dirichlet;
	fcpw::Scene<3> neumann;
	std::function<float(float)> neumannSamplingTraversalWeight = [](float x) -> float { return 1.0f; };

	void setGeometry(fcpw::Scene<3> &scene, const py::array_t<float> &position, const py::array_t<int> &index, 
					 fcpw::BoundingBox<DIM> &geometryBBox, bool addBBox) {
		throw std::runtime_error("setGeometry() not implemented for DIM=" + std::to_string(DIM));
	}
	
	const fcpw::Aggregate<3>* getAggregate(fcpw::Scene<3> &geometry) {
		fcpw::SceneData<3> *sceneData = geometry.getSceneData();
		return sceneData->soups.size() > 0 ? sceneData->aggregateInstancePtrs[0] : nullptr;
	}

	py::array_t<float> getPositions(fcpw::SceneData<3>* sceneData) const {
		if (sceneData->soups.size() == 0) return py::array_t<float>();
		std::vector<Vector3> &p = sceneData->soups[0].positions;
		py::array_t<float> result(p.size() * 3, (float*)p.data(), py::capsule(p.data(), [](void *data) {}));
		result.resize({long(p.size()), long(3)});
		return result;
	}	
	
	py::array_t<int> getIndices(fcpw::SceneData<3>* sceneData) const {
		if (sceneData->soups.size() == 0) return py::array_t<int>();
		std::vector<int> &i = sceneData->soups[0].indices;
		py::array_t<int> result(i.size(), (int*)i.data(), py::capsule(i.data(), [](void *data) {}));
		result.resize({long(i.size())});
		return result;
	}
	
	py::array_t<float> getNormals(fcpw::SceneData<3>* sceneData) const {
		if (sceneData->soups.size() == 0) return py::array_t<float>();
		std::vector<Vector3> &n = sceneData->soups[0].vNormals;
		py::array_t<float> result(n.size() * 3, (float*)n.data(), py::capsule(n.data(), [](void *data) {}));
		result.resize({long(n.size()), long(3)});
		return result;
	}

	const Vector<3>& getVertexPosition(fcpw::SceneData<3>* sceneData, int pIndex, int vLocalIndex) const {
		if (sceneData->soups.size() == 0) {
			throw std::runtime_error("Scene::getVertexPosition No soups found");
		}
		int vIndex = sceneData->soups[0].indices[DIM * pIndex + vLocalIndex];
		return sceneData->soups[0].positions[vIndex];
	}

	void registerDirichletUpdate() {
		fcpw::AggregateType aggregateType = buildBVH ?
											fcpw::AggregateType::Bvh_SurfaceArea :
											fcpw::AggregateType::Baseline;
		dirichlet.computeObjectNormals(0, computeWeightedNormals);
		dirichlet.build(aggregateType, true, false, false);
		zombie::populateGeometricQueries<DIM>(queries, bbox,
										      getAggregate(dirichlet), getAggregate(neumann),
										      neumannSamplingTraversalWeight);
	}

	void registerNeumannUpdate(bool computeWeightedNormals = false) {
		fcpw::AggregateType aggregateType = buildBVH ?
											fcpw::AggregateType::Bvh_SurfaceArea :
											fcpw::AggregateType::Baseline;
		neumann.computeObjectNormals(0, computeWeightedNormals);
		if (computeSilhouettes) neumann.computeSilhouettes({});
		neumann.build(aggregateType, true, false, false);
		zombie::populateGeometricQueries<DIM>(queries, bbox,
										      getAggregate(dirichlet), getAggregate(neumann),
										      neumannSamplingTraversalWeight);
	}
};

template<>
void MeshGeometry<2>::setGeometry(fcpw::Scene<3> &geometry, 
							      const py::array_t<float> &position, 
							      const py::array_t<int> &index, 
							      fcpw::BoundingBox<2> &geometryBBox,
							      bool addBBox) {
	py::buffer_info positionBuffer = position.request();
	if (positionBuffer.ndim != 2 || positionBuffer.shape[1] != 2) {
		throw std::runtime_error("Error: Position array should have dimensions Nx2");
	}

	py::buffer_info indexBuffer = index.request();
	if (indexBuffer.ndim != 2 || indexBuffer.shape[1] != 2) {
		throw std::runtime_error("Error: Indices array should have dimensions Nx2");
	}

	std::vector<std::vector<fcpw::PrimitiveType>> objectTypes(1, std::vector<fcpw::PrimitiveType>{fcpw::PrimitiveType::LineSegment});
	geometry.setObjectTypes(objectTypes);

	int nPositions = positionBuffer.shape[0];
	int V = addBBox ? nPositions + 4 : nPositions;
	geometry.setObjectVertexCount(V, 0);
	
	int nSegments = indexBuffer.shape[0];
	int L = addBBox ? nSegments + 4 : nSegments;
	geometry.setObjectLineSegmentCount(L, 0);

	float* positionData = static_cast<float*>(positionBuffer.ptr);
	for (int i = 0; i < nPositions; ++i) {
		Vector3 p = Vector3::Zero();
		p[0] = float(positionData[i * 2]);
		p[1] = float(positionData[i * 2 + 1]);
		geometryBBox.expandToInclude(Vector<2>(p[0], p[1]));
		geometry.setObjectVertex(p, i, 0);
	}
	
	Vector2 geometryBBoxCentroid = geometryBBox.centroid();
	geometryBBox.pMin = (geometryBBox.pMin - geometryBBoxCentroid) *  GEOMETRY_BBOX_SCALE_FACTOR + geometryBBoxCentroid;
	geometryBBox.pMax = (geometryBBox.pMax - geometryBBoxCentroid) *  GEOMETRY_BBOX_SCALE_FACTOR + geometryBBoxCentroid;

	int* indexData = static_cast<int*>(indexBuffer.ptr);
	for (int i = 0; i < nSegments; ++i) {
		int lineSegment[2] = {(int)indexData[i * 2], (int)indexData[i * 2 + 1]};
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
	
	if (addBBox) {
		Vector3 centroid = Vector3(bbox.centroid()[0], bbox.centroid()[1], 0);
		for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++) {
			float x = i == 0 ? bbox.pMin[0] : bbox.pMax[0];
			float y = j == 0 ? bbox.pMin[1] : bbox.pMax[1];
			Vector3 p = (Vector3(x, y, 0) - centroid) * BBOX_SCALE_FACTOR + centroid;
			geometry.setObjectVertex(p, nPositions + (i * 2 + j), 0);
		}
		std::vector<std::array<int, 2>> segments;
		segments.push_back({nPositions + 0, nPositions + 1});
		segments.push_back({nPositions + 1, nPositions + 3});
		segments.push_back({nPositions + 3, nPositions + 2});
		segments.push_back({nPositions + 2, nPositions + 0});
		for (int i = 0; i < 4; i++)
			geometry.setObjectLineSegment(segments[i].data(), nSegments + i, 0);
	}
}

template<>
void MeshGeometry<3>::setGeometry(fcpw::Scene<3> &geometry, 
							      const py::array_t<float> &position, 
							      const py::array_t<int> &index, 
							      fcpw::BoundingBox<3> &geometryBBox,
							      bool addBBox) {
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

	int nPositions = positionBuffer.shape[0];
	int V = addBBox ? nPositions + 8 : nPositions;
	geometry.setObjectVertexCount(V, 0);
	
	int nTriangles = indexBuffer.shape[0];
	int T = addBBox ? nTriangles + 12 : nTriangles;
	geometry.setObjectTriangleCount(T, 0);

	float* positionData = static_cast<float*>(positionBuffer.ptr);
	for (int i = 0; i < nPositions; ++i) {
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
	for (int i = 0; i < nTriangles; ++i) {
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
	
	if (addBBox) {
		Vector3 centroid = Vector3(bbox.centroid()[0], bbox.centroid()[1], bbox.centroid()[2]);
		for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		for (int k = 0; k < 2; k++) {
			float x = i == 0 ? bbox.pMin[0] : bbox.pMax[0];
			float y = j == 0 ? bbox.pMin[1] : bbox.pMax[1];
			float z = k == 0 ? bbox.pMin[2] : bbox.pMax[2];
			Vector3 p = (Vector3(x, y, z) - centroid) * BBOX_SCALE_FACTOR + centroid;
			geometry.setObjectVertex(p, nPositions + (i * 4 + j * 2 + k), 0);
		}

		std::vector<std::array<int, 3>> faces;
		// bottom
		faces.push_back({nPositions + 0, nPositions + 2, nPositions + 4});
		faces.push_back({nPositions + 2, nPositions + 6, nPositions + 4});
		// top
		faces.push_back({nPositions + 1, nPositions + 5, nPositions + 3});
		faces.push_back({nPositions + 7, nPositions + 3, nPositions + 5});
		// left
		faces.push_back({nPositions + 2, nPositions + 0, nPositions + 1});
		faces.push_back({nPositions + 2, nPositions + 1, nPositions + 3});
		// right
		faces.push_back({nPositions + 6, nPositions + 5, nPositions + 4});
		faces.push_back({nPositions + 6, nPositions + 7, nPositions + 5});
		// front
		faces.push_back({nPositions + 2, nPositions + 3, nPositions + 7});
		faces.push_back({nPositions + 2, nPositions + 7, nPositions + 6});
		// back
		faces.push_back({nPositions + 0, nPositions + 4, nPositions + 5});
		faces.push_back({nPositions + 0, nPositions + 5, nPositions + 1});
		for (int i = 0; i < 12; i++)
			geometry.setObjectTriangle(faces[i].data(), nTriangles + i, 0);
	}
}

template<>
void MeshGeometry<2>::setDirichletDisplacement(const MeshDisplacementType &type,
								  			   const VectorXi &params) {
	switch (type) {
		case MeshDisplacementType::VertexTranslation: {
			/*
				parameters: [p0.v0.x, p0.v0.y, p0.v1.x, p0.v1.y, p1.v0.x, p1.v0.y, p1.v1.x, p1.v1.y, ...]
			*/
			queries.computeDirichletDisplacement = [this](const Vector2& x) -> zombie::Differential<Vector2> {
				zombie::Differential<Vector2> v(Vector2::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;

				Vector3 queryPt = Vector3::Zero();
				queryPt.head(2) = x;
			
				fcpw::Interaction<3> interaction;
				fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
				dirichletAggregate->findClosestPoint(sphere, interaction, true);
				Vector3 projX = interaction.p;

				int pIndex = interaction.primitiveIndex;
				const Vector3& v0 = this->getVertexPosition(this->dirichlet.getSceneData(), pIndex, 0);
				const Vector3& v1 = this->getVertexPosition(this->dirichlet.getSceneData(), pIndex, 1);
				
				float length = (v1 - v0).norm();
				if (length < 1e-8) return v; 	// degenerate vertex, no derivative
				float t1 = std::clamp((projX - v0).norm() / length, 0.0f, 1.0f);
				float t0 = 1.0 - t1;
				if (this->flipOrientation) std::swap(t0, t1);
				v.ref(this->id + "." + std::to_string(pIndex) + ".0.0") = { t0, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".0.1") = { 0, t0};
				v.ref(this->id + "." + std::to_string(pIndex) + ".1.0") = { t1, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".1.1") = { 0, t1 };
				return v;
			};
		} break;
		case MeshDisplacementType::Translation: {
			/* 
				parameters: [tx, ty]
				mesh transform: [
					1	0   tx
					0   1	ty
					0   0   1
				]
			*/
			queries.computeDirichletDisplacement = [this](const Vector2& x) -> zombie::Differential<Vector<2>> {
				zombie::Differential<Vector2> v(Vector2::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;
				
				v.ref(this->id + ".0") = { 1.0, 0.0 };
				v.ref(this->id + ".1") = { 0.0, 1.0 };

				return v;
			};
		} break;
		case MeshDisplacementType::Affine: {
			int nParams = params.size();
			if (nParams != 6) {
				throw std::runtime_error("MeshDisplacementType::Affine requires 6 parameters (nParams = " + std::to_string(nParams));
			}

			Eigen::Matrix<float, 3, 3> meshTransform;
			meshTransform << params[0], params[1], params[2],
							 params[3], params[4], params[5],
							 0, 		0,		   1;
			Eigen::Matrix<float, 3, 3> invMeshTransform = meshTransform.inverse();

			queries.computeDirichletDisplacement = [this, invMeshTransform](const Vector2& x) -> zombie::Differential<Vector2> {
				zombie::Differential<Vector2> v(Vector2::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;

				Vector2 x0 = dehomogenize<2>(invMeshTransform * homogenize<2>(x));
				v.ref(this->id + ".0") = { x0[0], 0 };
				v.ref(this->id + ".1") = { x0[1], 0 };
				v.ref(this->id + ".2") = { 1, 0 };
				v.ref(this->id + ".3") = { 0, x0[0] };
				v.ref(this->id + ".4") = { 0, x0[1] };
				v.ref(this->id + ".5") = { 0, 1};

				return v;
			};
		} break;
		case MeshDisplacementType::EulerRodrigues:  {
			throw std::runtime_error("EulerRodrigues displacement not supported in 2D");
		} break;
		case MeshDisplacementType::None:
		default: {
			queries.computeDirichletDisplacement = [](const Vector2& x) -> zombie::Differential<Vector2> {
				return zombie::Differential<Vector2>(Vector2::Zero());
			};
		} break;
	}
}

template<>
void MeshGeometry<3>::setDirichletDisplacement(const MeshDisplacementType &type,
								  		       const VectorXi &params) {
	switch (type) {
		case MeshDisplacementType::VertexTranslation: {
			/*
			parameters: [p0.v0.x, p0.v0.y, p0.v0.z, p0.v1.x, p0.v1.y, p0.v1.z, p0.v2.x, p0.v2.y, p0.v2,z, p1.v0.x, p1.v0.y, p1.v0.z,...]
			*/
			queries.computeDirichletDisplacement = [this](const Vector3& x) -> zombie::Differential<Vector3> {
				zombie::Differential<Vector3> v(Vector3::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !dirichletBBox.contains(x)) return v;

				Vector3 queryPt = Vector3::Zero();
				queryPt.head(3) = x;
			
				fcpw::Interaction<3> interaction;
				fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
				dirichletAggregate->findClosestPoint(sphere, interaction, true);
				Vector3 projX = interaction.p;

				int pIndex = interaction.primitiveIndex;
				const Vector3& v0 = this->getVertexPosition(this->dirichlet.getSceneData(), pIndex, 0);
				const Vector3& v1 = this->getVertexPosition(this->dirichlet.getSceneData(), pIndex, 1);
				const Vector3& v2 = this->getVertexPosition(this->dirichlet.getSceneData(), pIndex, 2);
				Vector2 uv = barycentricCoordinates(x, v0, v1, v2);
				float t0 = uv[0];
				float t1 = uv[1];
				float t2 = 1.0f - t0 - t1;
				if (this->flipOrientation) std::swap(t0,  t1);
				v.ref(this->id + "." + std::to_string(pIndex) + ".0.0") = { t0, 0, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".0.1") = { 0, t0, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".0.2") = { 0, 0, t0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".1.0") = { t1, 0, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".1.1") = { 0, t1, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".1.2") = { 0, 0, t1};
				v.ref(this->id + "." + std::to_string(pIndex) + ".2.0") = { t2, 0, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".2.1") = { 0, t2, 0 };
				v.ref(this->id + "." + std::to_string(pIndex) + ".2.2") = { 0, 0, t2 };
				return v;
			};
		} break;
		case MeshDisplacementType::Translation: {
			queries.computeDirichletDisplacement = [this](const Vector3&x) -> zombie::Differential<Vector3> {
				zombie::Differential<Vector3> v(Vector3::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;
				v.ref(this->id + ".0") = { 1.0, 0.0, 0.0 };
				v.ref(this->id + ".1") = { 0.0, 1.0, 0.0 };
				v.ref(this->id + ".2") = { 0.0, 0.0, 1.0 };
				return v;
			};
		} break;
		case MeshDisplacementType::Affine: {
			int nParams = params.size();
			if (nParams != 12) {
				throw std::runtime_error("MeshDisplacementType::Affine requires 12 parameters (nParams = " + std::to_string(nParams));
			}

			Eigen::Matrix<float, 4, 4> meshTransform;
			meshTransform << params[0], params[1], params[2], 	params[3],
							 params[4], params[5], params[6], 	params[7],
							 params[8], params[9], params[10], 	params[11],
							 0, 		0,		   0,		  	1;
			Eigen::Matrix<float, 4, 4> invMeshTransform = meshTransform.inverse();

			queries.computeDirichletDisplacement = [this, invMeshTransform](const Vector3&x) -> zombie::Differential<Vector3> {
				zombie::Differential<Vector3> v(Vector3::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;

				Vector3 x0 = dehomogenize<3>(invMeshTransform * homogenize<3>(x));
				
				v.ref(this->id + ".0") = { x0[0], 0 , 0 };
				v.ref(this->id + ".1") = { x0[1], 0 , 0 };
				v.ref(this->id + ".2") = { x0[2], 0 , 0 };
				v.ref(this->id + ".3") = { 1, 0 , 0 };
				
				v.ref(this->id + ".4") = { 0, x0[0], 0 };
				v.ref(this->id + ".5") = { 0, x0[1], 0 };
				v.ref(this->id + ".6") = { 0, x0[2], 0 };
				v.ref(this->id + ".7") = { 0, 1, 0 };
				
				v.ref(this->id + ".8") = { 0, 0, x0[0]};
				v.ref(this->id + ".9") = { 0, 0, x0[1]};
				v.ref(this->id + ".10") = { 0, 0, x0[2]};
				v.ref(this->id + ".11") = { 0, 0, 1};
				return v;
			};
		} break;
		case MeshDisplacementType::EulerRodrigues: {
			int nParams = params.size();
			if (nParams != 7) {
				throw std::runtime_error("MeshDisplacementType::EulerRodrigues requires 7 parameters (nParams = " + std::to_string(nParams));
			}
			float a = params[0];
			float b = params[1];
			float c = params[2];
			float d = params[3];
			float tx = params[4];
			float ty = params[5];
			float tz = params[6];

			float a2 = a * a;
			float b2 = b * b;
			float c2 = c * c;
			float d2 = d * d;

			Eigen::Matrix<float, 4, 4> meshTransform;
			meshTransform << (a2 + b2 - c2 - d2), 	2 * (b * c - a * d), 	2 * (b * d + a * c),	tx,
							 2 * (b * c + a * d),   (a2 - b2 + c2 - d2), 	2 * (c * d - a * b), 	ty,
							 2 * (b * d - a * c),	2 * (c * d + a * b), 	(a2 - b2 - c2 + d2), 	tz,
							 0, 						0,		   					0,		  		1;
			Eigen::Matrix<float, 4, 4> invMeshTransform = meshTransform.inverse();
			Eigen::Matrix<float, 3, 3> dTda, dTdb, dTdc, dTdd;
			dTda << a,	-d,	c,
					d,  a, 	-b,
					-c,	b, 	a;
			dTda *= 2;
			dTdb << b, 	c, 	d,
					c,  -b, -a,
					d,	a, 	-b;
			dTdb *= 2;
			dTdc << -c, b, 	a,
					b,  c, d,
					-a, d, 	-c;
			dTdc *= 2;
			dTdd << -d, -a, b,
					a,  -d,	c,
					b,	c, 	d;
			dTdd *= 2;
			queries.computeDirichletDisplacement = [this, invMeshTransform, dTda, dTdb, dTdc, dTdd](const Vector3&x) -> zombie::Differential<Vector3> {
				zombie::Differential<Vector3> v(Vector3::Zero());
				const fcpw::Aggregate<3> *dirichletAggregate = this->getAggregate(this->dirichlet);
				if (dirichletAggregate == nullptr || !this->dirichletBBox.contains(x)) return v;
				Vector3 x0 = dehomogenize<3>(invMeshTransform * homogenize<3>(x));
				v.ref(this->id + "#rotation.0") = dTda * x0;
				v.ref(this->id + "#rotation.1") = dTdb * x0;
				v.ref(this->id + "#rotation.2") = dTdc * x0;
				v.ref(this->id + "#rotation.3") = dTdd * x0;
				v.ref(this->id + "#translation.0") = { 1, 0, 0 };
				v.ref(this->id + "#translation.1") = { 0, 1, 0 };
				v.ref(this->id + "#translation.2") = { 0, 0, 1};
				return v;
			};

		} break;
		case MeshDisplacementType::None:
		default: {
			queries.computeDirichletDisplacement = [](const Vector3& x) -> zombie::Differential<Vector3> {
				return zombie::Differential<Vector3>(Vector3::Zero());
			};
		} break;
	}
}

}; // pyzombie

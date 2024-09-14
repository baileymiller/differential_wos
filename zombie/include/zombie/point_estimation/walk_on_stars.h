#pragma once

#include <zombie/core/pde.h>
#include <zombie/core/geometric_queries.h>
#include <zombie/core/sampling.h>
#include <zombie/core/distributions.h>
#include <zombie/core/value.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

#define RADIUS_SHRINK_PERCENTAGE 0.99f

namespace zombie {

enum class EstimationQuantity {
	Solution,
	SolutionAndGradient,
	None
};

enum class SampleType {
	InDomain, // applies to both interior and exterior sample points for closed domains
	OnDirichletBoundary,
	OnNeumannBoundary
};

enum class WalkCompletionCode {
	ReachedDirichletBoundary,
	TerminatedWithRussianRoulette,
	ExceededMaxWalkLength,
	EscapedDomain
};

template <typename T>
struct RecursiveBoundaryData;

template <typename T>
class ProductEstimate;

template <typename T, int DIM>
struct SamplePoint;

template <int DIM>
struct SampleEstimationData;

template <typename T>
struct WalkSettings;

// NOTE: For data with multiple channels (e.g., 2D or 3D positions, rgb etc.), use
// Eigen::Array (in place of Eigen::VectorXf) as it supports component wise operations
template <typename T, int DIM>
class SampleStatistics;

template <typename T, int DIM>
struct WalkState;

template <typename T, int DIM>
class WalkOnStars {
public:
	// constructor
	WalkOnStars(const GeometricQueries<DIM>& queries_,
				std::function<void(WalkState<T, DIM>&)> getTerminalContribution_={}):
				queries(queries_), getTerminalContribution(getTerminalContribution_) {}

	// solves the given PDE at the input point; NOTE: assumes the point does not
	// lie on the boundary when estimating the gradient
	void solve(const PDE<T, DIM>& pde,
			   const WalkSettings<T>& walkSettings,
			   const SampleEstimationData<DIM>& estimationData,
			   SamplePoint<T, DIM>& samplePt) const {
		switch (estimationData.estimationQuantity) {
			case EstimationQuantity::Solution: {
				estimateSolution(pde, walkSettings, estimationData.nWalks, 
								 estimationData.nRecursiveWalks, samplePt);
			} break;
			case EstimationQuantity::SolutionAndGradient: {
				estimateSolutionAndGradient(pde, walkSettings, estimationData.directionForDerivative,
											estimationData.nWalks, estimationData.nRecursiveWalks, samplePt);
			} break;
			case EstimationQuantity::None: 
			default:
				break;
		}
	}

	// solves the given PDE at the input points (in parallel by default); NOTE:
	// assumes points do not lie on the boundary when estimating gradients
	void solve(const PDE<T, DIM>& pde,
			   const WalkSettings<T>& walkSettings,
			   const std::vector<SampleEstimationData<DIM>>& estimationData,
			   std::vector<SamplePoint<T, DIM>>& samplePts,
			   bool runSingleThreaded=false,
			   std::function<void(int, int)> reportProgress={}) const {
		// solve the PDE at each point independently
		int nPoints = (int)samplePts.size();
		if (runSingleThreaded || walkSettings.printLogs) {
			for (int i = 0; i < nPoints; i++) {
				solve(pde, walkSettings, estimationData[i], samplePts[i]);
				if (reportProgress) reportProgress(1, 0);
			}

		} else {
			auto run = [&](const tbb::blocked_range<int>& range) {
				for (int i = range.begin(); i < range.end(); ++i) {
					solve(pde, walkSettings, estimationData[i], samplePts[i]);
				}

				if (reportProgress) {
					int tbb_thread_id = tbb::this_task_arena::current_thread_index();
					reportProgress(range.end() - range.begin(), tbb_thread_id);
				}
			};

			tbb::blocked_range<int> range(0, nPoints);
			tbb::parallel_for(range, run);
		}
	}

private:
	// performs a single reflecting random walk starting at the input point
	WalkCompletionCode walk(const PDE<T, DIM>& pde,
							const WalkSettings<T>& walkSettings,
							float dirichletDist, float firstSphereRadius,
							bool flipNormalOrientation, pcg32& sampler,
							std::unique_ptr<GreensFnBall<DIM>>& greensFn,
							WalkState<T, DIM>& state) const {
		// recursively perform a random walk till it reaches the Dirichlet boundary
		bool firstStep = true;
		float randNumsForNeumannSampling[DIM];

		while (dirichletDist > walkSettings.epsilonShell) {
			// compute the star radius
			float starRadius;
			if (firstStep && firstSphereRadius > 0.0f) {
				starRadius = firstSphereRadius;

			} else {
				// for problems with double-sided boundary conditions, flip the current
				// normal orientation if the geometry is front-facing
				flipNormalOrientation = false;
				if (walkSettings.solveDoubleSided && state.onNeumannBoundary) {
					if (state.prevDistance > 0.0f && state.prevDirection.dot(state.currentNormal) < 0.0f) {
						state.currentNormal *= -1.0f;
						flipNormalOrientation = true;
					}
				}

				if (walkSettings.stepsBeforeUsingMaximalSpheres <= state.walkLength) {
					starRadius = dirichletDist;

				} else {
					// NOTE: using dirichletDist as the maximum radius for the closest silhouette
					// query can result in a smaller than maximal star-shaped region: should ideally
					// use the distance to the closest visible Dirichlet point
					starRadius = queries.computeStarRadius(state.currentPt, walkSettings.minStarRadius,
														   dirichletDist, walkSettings.silhouettePrecision,
														   flipNormalOrientation);

					// shrink the radius slightly for numerical robustness---using a conservative
					// distance does not impact correctness
					if (walkSettings.minStarRadius <= dirichletDist) {
						starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius, walkSettings.minStarRadius);
					}
				}
			}

			// update the ball center and radius
			greensFn->updateBall(state.currentPt, starRadius);

			// sample a direction uniformly
			Vector<DIM> direction = sampleUnitSphereUniform<DIM>(sampler);

			// perform hemispherical sampling if on the Neumann boundary, which cancels
			// the alpha term in our integral expression
			if (state.onNeumannBoundary && state.currentNormal.dot(direction) > 0.0f) {
				direction *= -1.0f;
			}

			// check if there is an intersection with the Neumann boundary along the ray:
			// currentPt + starRadius * direction
			IntersectionPoint<DIM> intersectionPt;
			bool intersectedNeumann = queries.intersectWithNeumann(state.currentPt, state.currentNormal, direction,
																   starRadius, state.onNeumannBoundary, intersectionPt);

			// check if there is no intersection with the Neumann boundary
			if (!intersectedNeumann) {
				// apply small offset to the current pt for numerical robustness if it on
				// the Neumann boundary---the same offset is applied during ray intersections
				Vector<DIM> currentPt = state.onNeumannBoundary ?
										queries.offsetPointAlongDirection(state.currentPt, -state.currentNormal) :
										state.currentPt;

				// set intersectionPt to a point on the spherical arc of the ball
				intersectionPt.pt = currentPt + starRadius*direction;
				intersectionPt.dist = starRadius;
			}

			if (!walkSettings.ignoreNeumannContribution) {
				// compute the non-zero Neumann contribution inside the star-shaped region;
				// define the Neumann value to be zero outside this region
				BoundarySample<DIM> neumannSample;
				for (int i = 0; i < DIM; i++) randNumsForNeumannSampling[i] = sampler.nextFloat();
				if (queries.sampleNeumann(state.currentPt, starRadius, randNumsForNeumannSampling, neumannSample)) {
					Vector<DIM> directionToSample = neumannSample.pt - state.currentPt;
					float distToSample = directionToSample.norm();
					float alpha = state.onNeumannBoundary ? 2.0f : 1.0f;
					bool estimateBoundaryNormalAligned = false;

					if (walkSettings.solveDoubleSided) {
						// normalize the direction to the sample, and flip the sample normal
						// orientation if the geometry is front-facing; NOTE: using a precision
						// parameter since unlike direction sampling, samples can lie on the same
						// halfplane as the current walk location
						directionToSample /= distToSample;
						if (flipNormalOrientation) {
							neumannSample.normal *= -1.0f;
							estimateBoundaryNormalAligned = true;

						} else if (directionToSample.dot(neumannSample.normal) < -walkSettings.silhouettePrecision) {
							bool flipNeumannSampleNormal = true;
							if (alpha > 1.0f) {
								// on concave boundaries, we want to sample back-facing neumann
								// values on front-facing geometry below the hemisphere, so we
								// avoid flipping the normal orientation in this case
								flipNeumannSampleNormal = directionToSample.dot(state.currentNormal) <
														  -walkSettings.silhouettePrecision;
							}

							if (flipNeumannSampleNormal) {
								neumannSample.normal *= -1.0f;
								estimateBoundaryNormalAligned = true;
							}
						}
					}

					if (neumannSample.pdf > 0.0f && distToSample < starRadius &&
						!queries.intersectsWithNeumann(state.currentPt, neumannSample.pt, state.currentNormal,
													   neumannSample.normal, state.onNeumannBoundary, true)) {
						float G = greensFn->evaluate(state.currentPt, neumannSample.pt);
						Value<T, DIM> h = walkSettings.solveDoubleSided ?
							              pde.neumannDoubleSided(neumannSample.pt, estimateBoundaryNormalAligned) :
							              pde.neumann(neumannSample.pt);
						state.totalNeumannContribution += state.throughput*alpha*G*h/neumannSample.pdf;
					}
				}
			}

			if (!walkSettings.ignoreSourceContribution) {
				// compute the source contribution inside the star-shaped region;
				// define the source value to be zero outside this region
				float sourcePdf;
				Vector<DIM> sourcePt = greensFn->sampleVolume(direction, sampler, sourcePdf);
				if (greensFn->r <= intersectionPt.dist) {
					// NOTE: hemispherical sampling causes the alpha term to cancel when
					// currentPt is on the Neumann boundary; in this case, the green's function
					// norm remains unchanged even though our domain is a hemisphere;
					// for double-sided problems in watertight domains, both the current pt
					// and source pt lie either inside or outside the domain by construction
					Value<T, DIM> sourceContribution = greensFn->norm()*pde.source(sourcePt);
					state.totalSourceContribution += state.throughput*sourceContribution;
				}
			}

			// update walk position
			state.prevDistance = intersectionPt.dist;
			state.prevDirection = direction;
			state.currentPt = intersectionPt.pt;
			state.currentNormal = intersectionPt.normal; // NOTE: stale unless intersectedNeumann is true
			state.onNeumannBoundary = intersectedNeumann;

			// check if the current pt lies outside the domain; for interior problems,
			// this tests for walks that escape due to numerical error
			if (!state.onNeumannBoundary && queries.outsideBoundingDomain(state.currentPt)) {
				if (walkSettings.printLogs) {
					std::cout << "Walk escaped domain!" << std::endl;
				}

				return WalkCompletionCode::EscapedDomain;
			}

			// update the walk throughput and use russian roulette to decide whether
			// to terminate the walk
			state.throughput *= greensFn->directionSampledPoissonKernel(state.currentPt);
			if (state.throughput < walkSettings.russianRouletteThreshold) {
				float survivalProb = state.throughput/walkSettings.russianRouletteThreshold;
				if (survivalProb < sampler.nextFloat()) {
					state.throughput = 0.0f;
					return WalkCompletionCode::TerminatedWithRussianRoulette;
				}

				state.throughput = walkSettings.russianRouletteThreshold;
			}

			// update the walk length and break if the max walk length is exceeded
			state.walkLength++;
			if (state.walkLength > walkSettings.maxWalkLength) {
				if (walkSettings.printLogs && !getTerminalContribution) {
					std::cout << "Maximum walk length exceeded!" << std::endl;
				}

				return WalkCompletionCode::ExceededMaxWalkLength;
			}

			// check whether to start applying Tikhonov regularization
			if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == state.walkLength)  {
				greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);
			}

			// compute the distance to the dirichlet boundary
			dirichletDist = queries.computeDistToDirichlet(state.currentPt, false);
			firstStep = false;
		}

		return WalkCompletionCode::ReachedDirichletBoundary;
	}

	void setTerminalContribution(WalkCompletionCode code, const PDE<T, DIM>& pde,
								 const WalkSettings<T>& walkSettings,
								 WalkState<T, DIM>& state,
								 const RecursiveBoundaryData<T> &recursiveBoundaryData) const {
		if (code == WalkCompletionCode::ReachedDirichletBoundary && !walkSettings.ignoreDirichletContribution) {
			// project the walk position to the Dirichlet boundary and grab the known boundary value
			float signedDistance;
			Vector<DIM> dirichletBoundaryPt = state.currentPt;
			queries.projectToDirichlet(dirichletBoundaryPt, state.currentNormal, signedDistance, 
									   walkSettings.solveDoubleSided || !walkSettings.ignoreShapeDifferential);

			Value<T, DIM> dirichlet = walkSettings.initVal;
			if (!walkSettings.ignoreDirichletContribution) {
				dirichlet = walkSettings.solveDoubleSided ?
						    pde.dirichletDoubleSided(dirichletBoundaryPt,  signedDistance > 0.0f) :
						    pde.dirichlet(dirichletBoundaryPt);
			}
			state.terminalContribution = dirichlet;

			if (!walkSettings.ignoreShapeDifferential) {
				bool flipNormal = (dirichletBoundaryPt - state.currentPt).dot(state.currentNormal) < 0;
				Vector<DIM> normal = flipNormal ? -state.currentNormal : state.currentNormal;
				Differential<float> Vn = dot<DIM>(queries.computeDirichletDisplacement(dirichletBoundaryPt), normal);
				T differentialDirichlet = dirichlet.gradient.dot(normal) - recursiveBoundaryData.normalDerivative;
				state.terminalContribution.differential += (Vn * differentialDirichlet);
			}

		} else if (code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution) {
			// get the user-specified terminal contribution
			getTerminalContribution(state);

		} else {
			// terminated with russian roulette or ignoring Dirichlet boundary values
			state.terminalContribution = walkSettings.initVal;
		}
	}

	// estimates only the solution of the given PDE at the input point
	void estimateSolution(const PDE<T, DIM>& pde,
						  const WalkSettings<T>& walkSettings,
						  int nWalks, int nRecursiveWalks, SamplePoint<T, DIM>& samplePt) const {
		// initialize statistics if there are no previous estimates
		bool hasPrevEstimates = samplePt.statistics != nullptr;
		if (!hasPrevEstimates) {
			samplePt.statistics = std::make_shared<SampleStatistics<T, DIM>>(walkSettings.initVal);
		}

		// check if the sample pt is on the Dirichlet boundary (and does not require recursive walks)
		if (samplePt.type == SampleType::OnDirichletBoundary && nRecursiveWalks <= 0) {
			if (!hasPrevEstimates) {
				// record the known boundary value
				Value<T, DIM> totalContribution = walkSettings.initVal;
				if (!walkSettings.ignoreDirichletContribution) {
					totalContribution = walkSettings.solveDoubleSided ?
										pde.dirichletDoubleSided(samplePt.pt, samplePt.estimateBoundaryNormalAligned) :
										pde.dirichlet(samplePt.pt);
				}
				
				// update statistics and set the first sphere radius to 0
				samplePt.statistics->addSolutionEstimate(totalContribution);
				samplePt.firstSphereRadius = 0.0f;
			}

			// no need to run any random walks
			return;

		} else if (samplePt.dirichletDist <= walkSettings.epsilonShell) {
			// run just a single walk since the sample pt is inside the epsilon shell
			nWalks = 1;
		}

		// for problems with double-sided boundary conditions, initialize the direction
		// of approach for walks, and flip the current normal orientation if the geometry
		// is front-facing
		Vector<DIM> currentNormal = samplePt.normal;
		Vector<DIM> prevDirection = samplePt.normal;
		float prevDistance = std::numeric_limits<float>::max();
		bool flipNormalOrientation = false;

		if (walkSettings.solveDoubleSided && samplePt.type == SampleType::OnNeumannBoundary) {
			if (samplePt.estimateBoundaryNormalAligned) {
				currentNormal *= -1.0f;
				prevDirection *= -1.0f;
				flipNormalOrientation = true;
			}
		}

		// precompute the first sphere radius for all walks
		if (!hasPrevEstimates) {
			if (samplePt.dirichletDist > walkSettings.epsilonShell && walkSettings.stepsBeforeUsingMaximalSpheres != 0) {
				// compute the star radius; NOTE: using dirichletDist as the maximum radius for
				// the closest silhouette query can result in a smaller than maximal star-shaped
				// region: should ideally use the distance to the closest visible Dirichlet point
				float starRadius = queries.computeStarRadius(samplePt.pt, walkSettings.minStarRadius,
															 samplePt.dirichletDist, walkSettings.silhouettePrecision,
															 flipNormalOrientation);

				// shrink the radius slightly for numerical robustness---using a conservative
				// distance does not impact correctness
				if (walkSettings.minStarRadius <= samplePt.dirichletDist) {
					starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius, walkSettings.minStarRadius);
				}

				samplePt.firstSphereRadius = starRadius;

			} else {
				samplePt.firstSphereRadius = samplePt.dirichletDist;
			}
		}

		std::shared_ptr<ProductEstimate<T>> productEstimate;
		if (walkSettings.solutionWeightedDifferentialBatchSize > 0) {
			productEstimate = std::make_shared<ProductEstimate<T>>(
				walkSettings.solutionWeightedDifferentialBatchSize, walkSettings.initVal);
		}

		// perform random walks
		for (int w = 0; w < nWalks; w++) {
			// initialize the greens function
			std::unique_ptr<GreensFnBall<DIM>> greensFn = nullptr;
			if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
				greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);

			} else {
				greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
			}

			// initialize the walk state
			WalkState<T, DIM> state(samplePt.pt, currentNormal, prevDirection, prevDistance,
									1.0f, samplePt.type == SampleType::OnNeumannBoundary, 0,
									walkSettings.initVal);

			// perform walk
			WalkCompletionCode code = walk(pde, walkSettings, samplePt.dirichletDist,
										   samplePt.firstSphereRadius, flipNormalOrientation,
										   samplePt.sampler, greensFn, state);

			if ((code == WalkCompletionCode::ReachedDirichletBoundary ||
				 code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
				(code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
				
				RecursiveBoundaryData<T> recursiveBoundaryData(walkSettings.initVal);
				if (code == WalkCompletionCode::ReachedDirichletBoundary) {
					estimateRecursiveBoundaryData(pde, walkSettings, nRecursiveWalks, 
												  state.currentPt, recursiveBoundaryData);
				}

				// compute the walk contribution
				setTerminalContribution(code, pde, walkSettings, state, recursiveBoundaryData);
				Value<T, DIM> totalContribution = state.throughput*state.terminalContribution +
									              state.totalNeumannContribution +
									              state.totalSourceContribution;
				
				// update statistics
				samplePt.statistics->addSolutionEstimate(totalContribution);
				samplePt.statistics->addWalkLength(state.walkLength);

				// compute unbiased estimate of the product u * u'
				if (productEstimate) {
					productEstimate->recordContributions(w, totalContribution.data, totalContribution.differential);
					if (productEstimate->isCacheFull(w)) {
						samplePt.statistics->addSolutionWeightedDifferential(productEstimate->compute(walkSettings.initVal));
						productEstimate->reset(walkSettings.initVal);
					}
				}
			}
		}
	}

	// estimates the solution and gradient of the given PDE at the input point;
	// NOTE: assumes the point does not lie on the boundary; the directional derivative
	// can be accessed through samplePt.statistics->getEstimatedDerivative()
	void estimateSolutionAndGradient(const PDE<T, DIM>& pde,
									 const WalkSettings<T>& walkSettings,
									 const Vector<DIM>& directionForDerivative,
									 int nWalks, int nRecursiveWalks, 
									 SamplePoint<T, DIM>& samplePt) const {
		// initialize statistics if there are no previous estimates
		bool hasPrevEstimates = samplePt.statistics != nullptr;
		if (!hasPrevEstimates) {
			samplePt.statistics = std::make_shared<SampleStatistics<T, DIM>>(walkSettings.initVal);
		}

		// reduce nWalks by 2 if using antithetic sampling
		int nAntitheticIters = 1;
		if (walkSettings.useGradientAntitheticVariates) {
			nWalks = std::max(1, nWalks/2);
			nAntitheticIters = 2;
		}

		// use the distance to the boundary as the first sphere radius for all walks;
		// shrink the radius slightly for numerical robustness---using a conservative
		// distance does not impact correctness
		float boundaryDist = std::min(samplePt.dirichletDist, samplePt.neumannDist);
		samplePt.firstSphereRadius = RADIUS_SHRINK_PERCENTAGE*boundaryDist;

		// generate stratified samples
		std::vector<float> stratifiedSamples;
		generateStratifiedSamples<DIM - 1>(stratifiedSamples, 2*nWalks, samplePt.sampler);

		// perform random walks
		for (int w = 0; w < nWalks; w++) {
			// initialize temporary variables for antithetic sampling
			float boundaryPdf, sourcePdf;
			Vector<DIM> boundaryPt, sourcePt;
			unsigned int seed = generateSeed();

			// compute control variates for the gradient estimate
			Value<T, DIM> boundaryGradientControlVariate = walkSettings.initVal;
			Value<T, DIM> sourceGradientControlVariate = walkSettings.initVal;
			if (walkSettings.useGradientControlVariates) {
				boundaryGradientControlVariate.data = samplePt.statistics->getEstimatedSolution();
				boundaryGradientControlVariate.differential = samplePt.statistics->getEstimatedDifferential();

				sourceGradientControlVariate.data = samplePt.statistics->getMeanFirstSourceContribution();
				sourceGradientControlVariate.differential = samplePt.statistics->getMeanFirstSourceDifferentialContribution();
			}

			for (int antitheticIter = 0; antitheticIter < nAntitheticIters; antitheticIter++) {
				// initialize the greens function
				std::unique_ptr<GreensFnBall<DIM>> greensFn = nullptr;
				if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
					greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);

				} else {
					greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
				}

				// initialize the walk state
				WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
										0.0f, 1.0f, false, 0, walkSettings.initVal);

				// update the ball center and radius
				greensFn->updateBall(state.currentPt, samplePt.firstSphereRadius);

				// compute the source contribution inside the ball
				if (!walkSettings.ignoreSourceContribution) {
					if (antitheticIter == 0) {
						float *u = &stratifiedSamples[(DIM - 1)*(2*w + 0)];
						Vector<DIM> sourceDirection = sampleUnitSphereUniform<DIM>(u);
						sourcePt = greensFn->sampleVolume(sourceDirection, samplePt.sampler, sourcePdf);

					} else {
						Vector<DIM> sourceDirection = sourcePt - state.currentPt;
						greensFn->yVol = state.currentPt - sourceDirection;
						greensFn->r = sourceDirection.norm();
					}

					float greensFnNorm = greensFn->norm();
					Value<T, DIM> sourceContribution = greensFnNorm*pde.source(greensFn->yVol);
					state.totalSourceContribution += state.throughput*sourceContribution;
					state.firstSourceContribution = sourceContribution;
					state.sourceGradientDirection = greensFn->gradient()/(sourcePdf*greensFnNorm);
				}

				// sample a point uniformly on the sphere; update the current position
				// of the walk, its throughput and record the boundary gradient direction
				if (antitheticIter == 0) {
					float *u = &stratifiedSamples[(DIM - 1)*(2*w + 1)];
					Vector<DIM> boundaryDirection;
					if (walkSettings.useCosineSamplingForDerivatives) {
						boundaryDirection = sampleUnitHemisphereCosine<DIM>(u);
						if (samplePt.sampler.nextFloat() < 0.5f) boundaryDirection[DIM - 1] *= -1.0f;
						boundaryPdf = 0.5f*pdfSampleUnitHemisphereCosine<DIM>(std::fabs(boundaryDirection[DIM - 1]));
						transformCoordinates<DIM>(directionForDerivative, boundaryDirection);

					} else {
						boundaryDirection = sampleUnitSphereUniform<DIM>(u);
						boundaryPdf = pdfSampleSphereUniform<DIM>(1.0f);
					}

					greensFn->ySurf = greensFn->c + greensFn->R*boundaryDirection;
					boundaryPt = greensFn->ySurf;

				} else {
					Vector<DIM> boundaryDirection = boundaryPt - state.currentPt;
					greensFn->ySurf = state.currentPt - boundaryDirection;
				}

				state.prevDistance = greensFn->R;
				state.prevDirection = (greensFn->ySurf - state.currentPt)/greensFn->R;
				state.currentPt = greensFn->ySurf;
				state.throughput *= greensFn->poissonKernel()/boundaryPdf;
				state.boundaryGradientDirection = greensFn->poissonKernelGradient()/(boundaryPdf*state.throughput);

				// compute the distance to the Dirichlet boundary
				float dirichletDist = queries.computeDistToDirichlet(state.currentPt, false);

				// perform walk
				samplePt.setRNG(seed);
				WalkCompletionCode code = walk(pde, walkSettings, dirichletDist, 0.0f,
											   false, samplePt.sampler, greensFn, state);

				if ((code == WalkCompletionCode::ReachedDirichletBoundary ||
					 code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
					(code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
					
					RecursiveBoundaryData recursiveBoundaryData(walkSettings.initVal);
					if (code == WalkCompletionCode::ReachedDirichletBoundary) {
						estimateRecursiveBoundaryData(pde, walkSettings, nRecursiveWalks, 
													  state.currentPt, recursiveBoundaryData);
					}

					// compute the walk contribution
					setTerminalContribution(code, pde, walkSettings, state, recursiveBoundaryData);
					Value<T, DIM> totalContribution = state.throughput*state.terminalContribution +
										              state.totalNeumannContribution +
										              state.totalSourceContribution;

					// compute the gradient contribution
                    Value<T, DIM> boundaryContribution = totalContribution - state.firstSourceContribution;
					Value<T, DIM> boundaryGradientMagnitude = boundaryContribution - boundaryGradientControlVariate;
					Value<T, DIM> sourceGradientMagnitude = state.firstSourceContribution - sourceGradientControlVariate;

					Value<SpatialGradient<T, DIM>, DIM> boundaryGradientEstimate = boundaryGradientMagnitude * state.boundaryGradientDirection;
                    Value<SpatialGradient<T, DIM>, DIM> sourceGradientEstimate = sourceGradientMagnitude * state.sourceGradientDirection;

					float cosineBoundary = state.boundaryGradientDirection.dot(directionForDerivative);
					float cosineSource = state.sourceGradientDirection.dot(directionForDerivative);
					
					Value<T, DIM> directionalDerivative = (boundaryGradientMagnitude * cosineBoundary +
														   sourceGradientMagnitude * cosineSource);

					// update statistics
					samplePt.statistics->addSolutionEstimate(totalContribution);
					samplePt.statistics->addFirstSourceContribution(state.firstSourceContribution);
					samplePt.statistics->addGradientEstimate(boundaryGradientEstimate, sourceGradientEstimate);
					samplePt.statistics->addDerivativeContribution(directionalDerivative);
					samplePt.statistics->addWalkLength(state.walkLength);
				}
			}
		}
	}

	void estimateRecursiveBoundaryData(const PDE<T, DIM>& pde,
						               const WalkSettings<T>& walkSettings,
						  	           int nWalks, const Vector<DIM>& terminalPt, 
									   RecursiveBoundaryData<T> &recursiveBoundaryData) const {
		if (nWalks == 0) return;
		
		float signedDistance;
		Vector<DIM> evalPt = terminalPt;
		Vector<DIM> normal;

		// offset eval point from boundary, then run recursive walks
		queries.projectToDirichlet(evalPt, normal, signedDistance, true);
		if ((evalPt - terminalPt).dot(normal) < 0) normal *= -1.0f;
		Vector<DIM> offsetPt = evalPt - normal * walkSettings.boundaryGradientOffset;

		float dirichletDist = queries.computeDistToDirichlet(offsetPt, false);
		float neumannDist = queries.computeDistToNeumann(offsetPt, false);
		SamplePoint<T, DIM> samplePt(offsetPt, normal, SampleType::InDomain,
									 1.0f, dirichletDist, neumannDist, walkSettings.initVal);

		if (walkSettings.useFiniteDifferences) {
			estimateSolution(pde, walkSettings, nWalks, 0, samplePt);
			T solution = samplePt.statistics->getEstimatedSolution();
			T dirichlet = walkSettings.solveDoubleSided ?
						  pde.dirichletDoubleSided(evalPt, signedDistance > 0.0f) :
						  pde.dirichlet(evalPt);
			recursiveBoundaryData.normalDerivative = (dirichlet - solution) / walkSettings.boundaryGradientOffset;
		} else {
			estimateSolutionAndGradient(pde, walkSettings, normal, nWalks, 0, samplePt);
			recursiveBoundaryData.normalDerivative = samplePt.statistics->getEstimatedDerivative();
		}
	}
	
	// members
	const GeometricQueries<DIM>& queries;
	std::function<void(WalkState<T, DIM>&)> getTerminalContribution;
};

template <typename T>
struct RecursiveBoundaryData {
	RecursiveBoundaryData(T initVal):
		normalDerivative(initVal) {};

	T normalDerivative;
};

template <typename T, int DIM>
struct SamplePoint {
	// constructor
	SamplePoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_, SampleType type_,
				float pdf_, float dirichletDist_, float neumannDist_, T initVal_):
				pt(pt_), normal(normal_), type(type_), pdf(pdf_),
				dirichletDist(dirichletDist_),
				neumannDist(neumannDist_),
				firstSphereRadius(0.0f),
				estimateBoundaryNormalAligned(false) {
		reset(initVal_);
		setRNG(generateSeed());
	}

	// resets solution data
	void reset(T initVal) {
		statistics = nullptr;
		solution = initVal;
		normalDerivative = initVal;
		source = initVal;
	}

	void setRNG(uint64_t seed) {
		sampler.seed(seed);
	}
	
	uint64_t getSamplerState() const {
		return sampler.state;
	}

	uint64_t getSamplerInc() const {
		return sampler.inc;
	}

	// members
	pcg32 sampler;
	Vector<DIM> pt;
	Vector<DIM> normal;
	SampleType type;
	float pdf;
	float dirichletDist;
	float neumannDist;
	float firstSphereRadius; // populated by WalkOnStars
	bool estimateBoundaryNormalAligned;
	std::shared_ptr<SampleStatistics<T, DIM>> statistics; // populated by WalkOnStars
	T solution, normalDerivative, source; // not populated by WalkOnStars, but available for downstream use (e.g. boundary value caching)
};

template <int DIM>
struct SampleEstimationData {
	// constructors
	SampleEstimationData(): nWalks(0), nRecursiveWalks(0), estimationQuantity(EstimationQuantity::None),
							directionForDerivative(Vector<DIM>::Zero()) {
		directionForDerivative(0) = 1.0f;
	}
	SampleEstimationData(int nWalks_, EstimationQuantity estimationQuantity_,
						 Vector<DIM> directionForDerivative_=Vector<DIM>::Zero()):
						 nWalks(nWalks_), nRecursiveWalks(0), estimationQuantity(estimationQuantity_),
						 directionForDerivative(directionForDerivative_) {}
	SampleEstimationData(int nWalks_, int nRecursiveWalks_, EstimationQuantity estimationQuantity_,
						 Vector<DIM> directionForDerivative_=Vector<DIM>::Zero()):
						 nWalks(nWalks_), nRecursiveWalks(nRecursiveWalks_),
						 estimationQuantity(estimationQuantity_),
						 directionForDerivative(directionForDerivative_) {}

	// members
	int nWalks;
	int nRecursiveWalks;
	EstimationQuantity estimationQuantity;
	Vector<DIM> directionForDerivative; // needed only for computing direction derivatives
};

template <typename T>
struct WalkSettings {
	// constructors
	WalkSettings(T initVal_, float epsilonShell_, float minStarRadius_,
				 int maxWalkLength_, bool solveDoubleSided_):
				 initVal(initVal_),
				 epsilonShell(epsilonShell_),
				 minStarRadius(minStarRadius_),
				 silhouettePrecision(1e-3f),
				 russianRouletteThreshold(0.0f),
				 boundaryGradientOffset(1e-2f),
				 maxWalkLength(maxWalkLength_),
				 stepsBeforeApplyingTikhonov(maxWalkLength_),
				 stepsBeforeUsingMaximalSpheres(maxWalkLength_),
				 solutionWeightedDifferentialBatchSize(2),
				 solveDoubleSided(solveDoubleSided_),
				 useGradientControlVariates(true),
				 useGradientAntitheticVariates(true),
				 useCosineSamplingForDerivatives(false),
				 useFiniteDifferences(true),
				 ignoreDirichletContribution(false),
				 ignoreNeumannContribution(false),
				 ignoreSourceContribution(false),
				 ignoreShapeDifferential(false),
				 printLogs(false) {}

	WalkSettings(T initVal_, float epsilonShell_, float minStarRadius_,
				 float silhouettePrecision_, float russianRouletteThreshold_,
				 int maxWalkLength_, int stepsBeforeApplyingTikhonov_,
				 int stepsBeforeUsingMaximalSpheres_, bool solveDoubleSided_,
				 bool useGradientControlVariates_, bool useGradientAntitheticVariates_,
				 bool useCosineSamplingForDerivatives_, bool ignoreDirichletContribution_,
				 bool ignoreNeumannContribution_, bool ignoreSourceContribution_,
				 bool printLogs_):
				 initVal(initVal_),
				 epsilonShell(epsilonShell_),
				 minStarRadius(minStarRadius_),
				 silhouettePrecision(silhouettePrecision_),
				 russianRouletteThreshold(russianRouletteThreshold_),
				 maxWalkLength(maxWalkLength_),
				 stepsBeforeApplyingTikhonov(stepsBeforeApplyingTikhonov_),
				 stepsBeforeUsingMaximalSpheres(stepsBeforeUsingMaximalSpheres_),
				 solutionWeightedDifferentialBatchSize(2),
				 solveDoubleSided(solveDoubleSided_),
				 useGradientControlVariates(useGradientControlVariates_),
				 useGradientAntitheticVariates(useGradientAntitheticVariates_),
				 useCosineSamplingForDerivatives(useCosineSamplingForDerivatives_),
				 useFiniteDifferences(true),
				 ignoreDirichletContribution(ignoreDirichletContribution_),
				 ignoreNeumannContribution(ignoreNeumannContribution_),
				 ignoreSourceContribution(ignoreSourceContribution_),
				 ignoreShapeDifferential(false),
				 printLogs(printLogs_) {}

	// members
	T initVal;
	float epsilonShell;
	float minStarRadius;
	float silhouettePrecision;
	float russianRouletteThreshold;
	float boundaryGradientOffset;
	int maxWalkLength;
	int stepsBeforeApplyingTikhonov;
	int stepsBeforeUsingMaximalSpheres;
	int solutionWeightedDifferentialBatchSize;
	bool solveDoubleSided; // NOTE: this flag should be set to true if domain is open
	bool useGradientControlVariates;
	bool useGradientAntitheticVariates;
	bool useCosineSamplingForDerivatives;
	bool useFiniteDifferences;
	bool ignoreDirichletContribution;
	bool ignoreNeumannContribution;
	bool ignoreSourceContribution;
	bool ignoreShapeDifferential;
	bool printLogs;
};

// cache estimates of A and B so that we can compute an unbiased estimate of A * B
// we specifically evaluate this as 1 / (N * N-1) \sum_{i=1}^N B_i * (\sum_{j=1}^{N} A_j - A_i)
// which requires us to keep track of \sum_{j=1}^N A_j and A_i, B_i for i = 1 to N
template <typename T>
class ProductEstimate {
public:
	ProductEstimate(int cacheSize_, T initVal): 
		sumA(initVal),
		cachedEstimates(cacheSize_, std::pair<T, Differential<T>>(initVal, initVal)) {}

	void recordContributions(int i, T estimateA, Differential<T> estimateB) {
		sumA += estimateA;
		cachedEstimates[i % cachedEstimates.size()] = std::pair<T, Differential<T>>(estimateA, estimateB);
	}

	bool isCacheFull(int i) {
		return i % cachedEstimates.size() == cachedEstimates.size() - 1;
	}

	Differential<T> compute(T initVal) {
		int cacheSize = cachedEstimates.size();
		if (cacheSize < 2) {
        	throw std::runtime_error("ProductEstimate::compute Cannot compute product estimates when batch is size 1.");
    	}

		Differential<T> productPairsTotal(initVal);
		for (const std::pair<T, Differential<T>>& pair: cachedEstimates) {
			T partialSum = (sumA - pair.first);
			productPairsTotal += pair.second * partialSum;
		}
		productPairsTotal /= (cacheSize * (cacheSize - 1));
		return productPairsTotal;
	}

	void reset(T initVal) {
		sumA = initVal;
		std::fill(cachedEstimates.begin(), cachedEstimates.end(),
				  std::pair<T, Differential<T>>(initVal, initVal));
	}

	T sumA;
	std::vector<std::pair<T, Differential<T>>> cachedEstimates;
};

template <typename T, int DIM>
class SampleStatistics {
public:
	// constructor
	SampleStatistics(T initVal) {
		reset(initVal);
	}

	// resets statistics
	void reset(T initVal) {
		solutionMean = initVal;
		solutionM2 = initVal;

		gradientMean = initVal;
		gradientM2 = initVal;

		totalFirstSourceContribution = initVal;
		totalDerivativeContribution = initVal;

		totalSolutionWeightedDifferential = Differential<T>(initVal);
		totalDifferentialContribution = Differential<T>(initVal);
		totalGradientDifferentialContribution = Differential<SpatialGradient<T, DIM>>(SpatialGradient<T, DIM>(initVal));
		totalDerivativeDifferentialContribution = Differential<T>(initVal);

		nSolutionEstimates = 0;
		nGradientEstimates = 0;
		nSolutionWeightedDifferentialEstimates = 0;
		totalWalkLength = 0;
	}

	// adds solution estimate to running sum
	void addSolutionEstimate(const Value<T, DIM>& estimate) {
		nSolutionEstimates += 1;
		update(estimate.data, solutionMean, solutionM2, nSolutionEstimates);
		totalDifferentialContribution += estimate.differential;
	}

	// adds gradient estimate to running sum
	void addGradientEstimate(const Value<SpatialGradient<T, DIM>, DIM> &boundaryEstimate, const Value<SpatialGradient<T, DIM>, DIM> &sourceEstimate) {
		nGradientEstimates += 1;
		update(boundaryEstimate.data + sourceEstimate.data, gradientMean, gradientM2, nGradientEstimates);
		totalGradientDifferentialContribution += boundaryEstimate.differential + sourceEstimate.differential;
	}

	// adds gradient estimate to running sum
	void addGradientEstimate(const Value<SpatialGradient<T, DIM>, DIM> &estimate) {
		nGradientEstimates += 1;
		update(estimate.data, gradientMean, gradientM2, nGradientEstimates);
		totalGradientDifferentialContribution += estimate.differential;
	}

	// adds source contribution for the first step to running sum
	void addFirstSourceContribution(const Value<T, DIM>& contribution) {
		totalFirstSourceContribution += contribution.data;
		totalFirstSourceDifferentialContribution += contribution.differential;
	}

	// adds derivative contribution to running sum
	void addDerivativeContribution(const T& contribution) {
		totalDerivativeContribution += contribution;
	}

	// adds the product of the solution and differential
	void addSolutionWeightedDifferential(const Differential<T>& contribution) {
		nSolutionWeightedDifferentialEstimates += 1;
		totalSolutionWeightedDifferential += contribution;
	}

	// adds walk length to running sum
	void addWalkLength(int length) {
		totalWalkLength += length;
	}

	// returns estimated solution
	T getEstimatedSolution() const {
		return solutionMean;
	}

	// returns variance of estimated solution
	T getEstimatedSolutionVariance() const {
		int N = std::max(1, nSolutionEstimates - 1);
		return solutionM2/float(N);
	}

	// returns estimated gradient
	const SpatialGradient<T, DIM> getEstimatedGradient() const {
		return gradientMean;
	}

	// returns variance of estimated gradient
	std::vector<T> getEstimatedGradientVariance() const {
		int N = std::max(1, nGradientEstimates - 1);
		std::vector<T> variance(DIM);

		for (int i = 0; i < DIM; i++) {
			variance[i] = gradientM2[i]/float(N);
		}

		return variance;
	}

	// returns mean source contribution for the first step
	T getMeanFirstSourceContribution() const {
		int N = std::max(1, nSolutionEstimates);
		return totalFirstSourceContribution/float(N);
	}	

	// returns estimated derivative
	T getEstimatedDerivative() const {
		int N = std::max(1, nSolutionEstimates);
		return totalDerivativeContribution/float(N);
	}

	// returns the estimated differentials
	Differential<T> getEstimatedDifferential() const {
		int N = std::max(1, nSolutionEstimates);
		return totalDifferentialContribution/float(N);
	}
	
	Differential<SpatialGradient<T, DIM>> getEstimatedGradientDifferential() const {
		int N = std::max(1, nGradientEstimates);
		return totalGradientDifferentialContribution/float(N);
	}

	Differential<T> getMeanFirstSourceDifferentialContribution() const {
		int N = std::max(1, nSolutionEstimates);
		return totalFirstSourceDifferentialContribution/float(N);
	}
	
	Differential<T> getEstimatedDerivativeDifferential() const {
		int N = std::max(1, nSolutionEstimates);
		return totalDerivativeDifferentialContribution/float(N);
	}

	Differential<T> getEstimatedSolutionWeightedDifferential() const {
		int N = std::max(1, nSolutionWeightedDifferentialEstimates);
		return totalSolutionWeightedDifferential/float(N);
	}

	// returns number of solution estimates
	int getSolutionEstimateCount() const {
		return nSolutionEstimates;
	}

	// returns number of gradient estimates
	int getGradientEstimateCount() const {
		return nGradientEstimates;
	}

	// returns mean walk length
	float getMeanWalkLength() const {
		int N = std::max(1, nSolutionEstimates);
		return (float)totalWalkLength/float(N);
	}

private:
	// updates statistics
	void update(const T& estimate, T& mean, T& M2, int N) {
		T delta = estimate - mean;
		mean += delta/float(N);
		T delta2 = estimate - mean;
		M2 += delta*delta2;
	}
	
	// updates statistics
	void update(const SpatialGradient<T, DIM>& estimate, 
				SpatialGradient<T, DIM>& mean, 
				SpatialGradient<T, DIM>& M2, 
				int N) {
		SpatialGradient<T, DIM> delta = estimate - mean;
		mean += delta/float(N);
		SpatialGradient<T, DIM> delta2 = estimate - mean;
		M2 += delta * delta2;
	}

	// members
	T solutionMean, solutionM2;									// u
	SpatialGradient<T, DIM> gradientMean, gradientM2;		    // grad u
	T totalFirstSourceContribution;								// f
	T totalDerivativeContribution;								// dudn
	
	Differential<T> totalDifferentialContribution;						 			  // u'
	Differential<SpatialGradient<T, DIM>> totalGradientDifferentialContribution; 	  // grad u'
	Differential<T> totalFirstSourceDifferentialContribution;			 			  // f'
	Differential<T> totalDerivativeDifferentialContribution;	        	 		  // du'dn
	Differential<T> totalSolutionWeightedDifferential;					 			  // u' * u

	int nSolutionEstimates, nGradientEstimates;
	int nSolutionWeightedDifferentialEstimates;
	int totalWalkLength;
};

template <typename T, int DIM>
struct WalkState {
	// constructor
	WalkState(const Vector<DIM>& currentPt_, const Vector<DIM>& currentNormal_,
			  const Vector<DIM>& prevDirection_, float prevDistance_, float throughput_,
			  bool onNeumannBoundary_, int walkLength_, T initVal_):
			  currentPt(currentPt_),
			  currentNormal(currentNormal_),
			  prevDirection(prevDirection_),
			  sourceGradientDirection(Vector<DIM>::Zero()),
			  boundaryGradientDirection(Vector<DIM>::Zero()),
			  prevDistance(prevDistance_),
			  throughput(throughput_),
			  onNeumannBoundary(onNeumannBoundary_),
			  terminalContribution(initVal_),
			  totalNeumannContribution(initVal_),
			  totalSourceContribution(initVal_),
			  firstSourceContribution(initVal_),
			  walkLength(walkLength_) {}

	// members
	Vector<DIM> currentPt;
	Vector<DIM> currentNormal;
	Vector<DIM> prevDirection;
	Vector<DIM> sourceGradientDirection;
	Vector<DIM> boundaryGradientDirection;

	float prevDistance;
	float throughput;
	bool onNeumannBoundary;
	
	Value<T, DIM> terminalContribution;
	Value<T, DIM> totalNeumannContribution;
	Value<T, DIM> totalSourceContribution;
	Value<T, DIM> firstSourceContribution;

	int walkLength;
};

} // zombie

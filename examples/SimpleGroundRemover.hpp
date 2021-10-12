/**
 * @file    SimpleGroundRemover.hpp
 *
 * @brief   btran
 *
 */

#pragma once

#include <pcl/features/normal_3d_omp.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_normal_parallel_plane.h>

namespace perception
{
template <typename PointCloudType>
inline pcl::PointCloud<pcl::Normal> calculateNormals(const typename pcl::PointCloud<PointCloudType>::Ptr& inCloud,
                                                     int numNeighbors = 10, int numThreads = 4)
{
    pcl::PointCloud<pcl::Normal> normals;
    pcl::NormalEstimationOMP<PointCloudType, pcl::Normal> estimator;
    estimator.setNumberOfThreads(numThreads);
    estimator.setInputCloud(inCloud);

    typename pcl::search::KdTree<PointCloudType>::Ptr tree(new pcl::search::KdTree<PointCloudType>);
    estimator.setSearchMethod(tree);
    estimator.setKSearch(numNeighbors);
    estimator.compute(normals);

    return normals;
}

template <typename PointCloudType>
inline std::vector<typename pcl::PointCloud<PointCloudType>::Ptr>
extractClustersBasedOnPlaneModel(const typename pcl::PointCloud<PointCloudType>::Ptr& inCloud,
                                 const pcl::ModelCoefficients& coeffs, const double threshDist)
{
    typename pcl::PointCloud<PointCloudType>::Ptr firstHalf(new pcl::PointCloud<PointCloudType>),
        secondHalf(new pcl::PointCloud<PointCloudType>);

    for (const auto& point : *inCloud) {
        pcl::pointToPlaneDistanceSigned<PointCloudType>(point, coeffs.values[0], coeffs.values[1], coeffs.values[2],
                                                        coeffs.values[3]) < threshDist
            ? firstHalf->points.emplace_back(point)
            : secondHalf->points.emplace_back(point);
    }

    return {firstHalf, secondHalf};
}

inline pcl::ModelCoefficients toModelCoefficients(const Eigen::VectorXf& eigenCoeffs)
{
    pcl::ModelCoefficients modelCoeffs;
    std::copy(eigenCoeffs.data(), eigenCoeffs.data() + eigenCoeffs.size(), std::back_inserter(modelCoeffs.values));
    return modelCoeffs;
}

template <typename PointCloudType> class SimpleGroundRemover
{
 public:
    using PointCloud = pcl::PointCloud<PointCloudType>;
    using PointCloudPtr = typename PointCloud::Ptr;

    struct Param {
        double angleEps = 7;
        double distanceEps = 0.05;

        // for normal estimation
        int numNeighbors = 10;
        int numThreads = 4;
    };

    explicit SimpleGroundRemover(const Param& param)
        : m_param(param)
    {
    }

    void run(const PointCloudPtr& inCloud, PointCloud* nonGround, PointCloud* ground = nullptr) const
    {
        typename pcl::SampleConsensusModelNormalParallelPlane<PointCloudType, pcl::Normal>::Ptr groundPlaneModel(
            new pcl::SampleConsensusModelNormalParallelPlane<PointCloudType, pcl::Normal>(inCloud));

        auto normals = calculateNormals<PointCloudType>(inCloud, m_param.numNeighbors, m_param.numThreads);
        groundPlaneModel->setInputNormals(
            pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>(std::move(normals))));
        groundPlaneModel->setEpsAngle(m_param.angleEps);
        groundPlaneModel->setAxis(Eigen::Vector3f(0, 0, 1));

        pcl::RandomSampleConsensus<PointCloudType> sac(groundPlaneModel, m_param.distanceEps);
        sac.computeModel();

        Eigen::VectorXf groundPlaneCoeffs;
        sac.getModelCoefficients(groundPlaneCoeffs);

        {
            pcl::Indices inliers;
            sac.getInliers(inliers);
            groundPlaneModel->optimizeModelCoefficients(inliers, groundPlaneCoeffs, groundPlaneCoeffs);
        }

        // flip ground plane's normal vector
        if (groundPlaneCoeffs[2] < 0) {
            for (int i = 0; i < 4; ++i) {
                groundPlaneCoeffs[i] *= -1;
            }
        }

        auto output = extractClustersBasedOnPlaneModel<PointCloudType>(inCloud, toModelCoefficients(groundPlaneCoeffs),
                                                                       2 * m_param.distanceEps);

        if (nonGround) {
            *nonGround = std::move(*output[1]);
        }

        if (ground) {
            *ground = std::move(*output[0]);
        }
    }

 private:
    Param m_param;
};
}  // namespace perception

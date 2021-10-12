/**
 * @file    CurvedVoxelClustering.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace perception
{
template <typename PointCloudType> class CurvedVoxelClustering
{
 public:
    using PointCloud = pcl::PointCloud<PointCloudType>;
    using PointCloudPtr = typename PointCloud::Ptr;

    struct Param {
        double deltaDistance = 0.5;         // meter
        double deltaPolarAngleDeg = 3;      // deg
        double deltaAzimuthAngleDeg = 1.3;  // deg
        int numMinPoints = 3;
    };

    explicit CurvedVoxelClustering(const Param& param);

    std::vector<std::vector<int>> run(const PointCloudPtr& inCloud) const;

 private:
    std::vector<std::vector<int>> getAllClusters(const std::vector<int>& clusterIndices) const;

    struct GridParam;
    GridParam estimateGridParam(const PointCloudPtr& inCloud, std::vector<std::array<double, 3>>& gridPoints) const;

    std::multimap<int, int> updateHashTable(const std::vector<std::array<double, 3>>& gridPoints,
                                            const GridParam& gridParam) const;

    std::array<int, 3> getGridIndices(const std::array<double, 3>& gridPoint, const GridParam& gridParam) const;

    std::vector<int> getNeighbors(const std::array<double, 3>& gridPoint, const GridParam& gridParam,
                                  const std::multimap<int, int>& hashTable) const;

    void mergeClusters(std::vector<int>& clusterIndices, int idx1, int idx2) const;

 private:
    Param m_param;
    std::array<double, 3> m_gridResolutions;
};
}  // namespace perception

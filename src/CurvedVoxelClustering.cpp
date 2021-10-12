/**
 * @file    CurvedVoxelClustering.cpp
 *
 * @author  btran
 *
 */

#include <set>

#include <pcl/common/angles.h>

#include <curved_voxel_clustering/CurvedVoxelClustering.hpp>

namespace perception
{
namespace
{
template <typename DataType>
inline bool almostEquals(const DataType input, const DataType other, DataType rtol = 1e-05, DataType atol = 1e-08)
{
    return std::abs(input - other) <= atol + rtol * std::abs(other);
}
}  // namespace

template <typename PointCloudType> struct CurvedVoxelClustering<PointCloudType>::GridParam {
    std::array<double, 3> minPoint;
    std::array<double, 3> maxPoint;
    std::array<int, 3> bucketNums;
};

template <typename PointCloudType>
CurvedVoxelClustering<PointCloudType>::CurvedVoxelClustering(const Param& param)
    : m_param(param)
    , m_gridResolutions(
          {param.deltaDistance, pcl::deg2rad(param.deltaPolarAngleDeg), pcl::deg2rad(param.deltaAzimuthAngleDeg)})
{
}

template <typename PointCloudType>
std::vector<std::vector<int>> CurvedVoxelClustering<PointCloudType>::run(const PointCloudPtr& inCloud) const
{
    if (inCloud->empty()) {
        return {};
    }

    std::vector<std::array<double, 3>> gridPoints;
    GridParam gridParam = this->estimateGridParam(inCloud, gridPoints);

    auto hashTable = this->updateHashTable(gridPoints, gridParam);

    std::vector<int> clusterIndices(inCloud->size(), -1);

    int curClusterIdx = 0;
    for (std::size_t i = 0; i < inCloud->size(); ++i) {
        if (clusterIndices[i] >= 0) {
            continue;
        }

        const auto& curGridPoint = gridPoints[i];
        std::vector<int> neighborIndices = this->getNeighbors(curGridPoint, gridParam, hashTable);
        for (int neighborIdx : neighborIndices) {
            if (neighborIdx == i) {
                continue;
            }

            int curPointVoxelIdx = clusterIndices[i];
            int neighborVoxelIdx = clusterIndices[neighborIdx];

            if (curPointVoxelIdx >= 0 && neighborVoxelIdx >= 0) {
                if (curPointVoxelIdx != neighborVoxelIdx) {
                    this->mergeClusters(clusterIndices, curPointVoxelIdx, neighborVoxelIdx);
                }
            } else {
                if (curPointVoxelIdx < 0) {
                    clusterIndices[i] = neighborVoxelIdx;
                } else {
                    clusterIndices[neighborIdx] = curPointVoxelIdx;
                }
            }
        }

        if (clusterIndices[i] < 0 && neighborIndices.size() >= m_param.numMinPoints) {
            for (int neighborIdx : neighborIndices) {
                clusterIndices[neighborIdx] = curClusterIdx;
            }
            curClusterIdx++;
        }
    }

    return this->getAllClusters(clusterIndices);
}

template <typename PointCloudType>
std::vector<std::vector<int>>
CurvedVoxelClustering<PointCloudType>::getAllClusters(const std::vector<int>& clusterIndices) const
{
    std::vector<std::pair<int, int>> clusterPointPairs;
    clusterPointPairs.reserve(clusterIndices.size());

    for (std::size_t i = 0; i < clusterIndices.size(); ++i) {
        clusterPointPairs.emplace_back(std::make_pair(clusterIndices[i], i));
    }

    std::sort(clusterPointPairs.begin(), clusterPointPairs.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::vector<std::vector<int>> allClusters;
    int curClusterIdx = -1;
    std::vector<int> curCluster;

    for (std::size_t i = 0; i < clusterIndices.size(); ++i) {
        const auto& curPair = clusterPointPairs[i];
        if (curClusterIdx != curPair.first) {
            if (curClusterIdx >= 0) {
                allClusters.emplace_back(std::move(curCluster));
            }

            curClusterIdx = curPair.first;
        }

        curCluster.emplace_back(curPair.second);
    }
    allClusters.emplace_back(std::move(curCluster));

    return allClusters;
}

template <typename PointCloudType>
typename CurvedVoxelClustering<PointCloudType>::GridParam
CurvedVoxelClustering<PointCloudType>::estimateGridParam(const PointCloudPtr& inCloud,
                                                         std::vector<std::array<double, 3>>& gridPoints) const
{
    GridParam gridParam;
    std::fill(gridParam.minPoint.begin(), gridParam.minPoint.end(), std::numeric_limits<double>::max());
    std::fill(gridParam.maxPoint.begin(), gridParam.maxPoint.end(), std::numeric_limits<double>::lowest());

    gridPoints.resize(inCloud->size());
    for (std::size_t i = 0; i < inCloud->size(); ++i) {
        const auto& point = inCloud->points[i];
        auto& gridPoint = gridPoints[i];  // range, polar angle, azimuth angle
        gridPoint[0] = point.getVector3fMap().norm();
        gridPoint[1] = almostEquals<double>(gridPoint[0], 0) ? 0 : std::acos(point.z / gridPoint[0]);
        gridPoint[2] = almostEquals<double>(point.x, 0) ? 0 : std::atan2(point.y, point.x);

        for (int i = 0; i < 3; ++i) {
            if (gridPoint[i] < gridParam.minPoint[i]) {
                gridParam.minPoint[i] = gridPoint[i];
            }

            if (gridPoint[i] > gridParam.maxPoint[i]) {
                gridParam.maxPoint[i] = gridPoint[i];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        gridParam.bucketNums[i] = std::ceil((gridParam.maxPoint[i] - gridParam.minPoint[i]) / m_gridResolutions[i]);
    }

    return gridParam;
}

template <typename PointCloudType>
std::multimap<int, int>
CurvedVoxelClustering<PointCloudType>::updateHashTable(const std::vector<std::array<double, 3>>& gridPoints,
                                                       const GridParam& gridParam) const
{
    std::multimap<int, int> hashTable;

    for (std::size_t pointIdx = 0; pointIdx < gridPoints.size(); ++pointIdx) {
        const auto& curGridPoint = gridPoints[pointIdx];
        std::array<int, 3> gridIndices = this->getGridIndices(curGridPoint, gridParam);

        int voxelIdx = gridIndices[0] * gridParam.bucketNums[1] * gridParam.bucketNums[2] +
                       gridIndices[1] * gridParam.bucketNums[2] + gridIndices[2];

        hashTable.emplace(voxelIdx, pointIdx);
    }

    return hashTable;
}

template <typename PointCloudType>
std::array<int, 3> CurvedVoxelClustering<PointCloudType>::getGridIndices(const std::array<double, 3>& gridPoint,
                                                                         const GridParam& gridParam) const
{
    std::array<int, 3> gridIndices;
    for (int i = 0; i < 3; ++i) {
        gridIndices[i] = (gridPoint[i] - gridParam.minPoint[i]) / m_gridResolutions[i];
    }
    return gridIndices;
}

template <typename PointCloudType>
std::vector<int> CurvedVoxelClustering<PointCloudType>::getNeighbors(const std::array<double, 3>& gridPoint,
                                                                     const GridParam& gridParam,
                                                                     const std::multimap<int, int>& hashTable) const
{
    std::array<int, 3> gridIndices = this->getGridIndices(gridPoint, gridParam);
    std::vector<int> allNeighborIndices;

    for (int i = std::max(0, gridIndices[0] - 1); i <= std::min(gridParam.bucketNums[0] - 1, gridIndices[0] + 1); ++i) {
        for (int j = std::max(0, gridIndices[1] - 1); j <= std::min(gridParam.bucketNums[1] - 1, gridIndices[1] + 1);
             ++j) {
            for (int k = gridIndices[2] - 1; k <= gridIndices[2] + 1; ++k) {
                int curAzimuthAngleIdx = k;
                if (curAzimuthAngleIdx < 0) {
                    curAzimuthAngleIdx = gridParam.bucketNums[2] - 1;
                }
                if (curAzimuthAngleIdx > gridParam.bucketNums[2] - 1) {
                    curAzimuthAngleIdx = 0;
                }

                int voxelIdx = i * gridParam.bucketNums[1] * gridParam.bucketNums[2] + j * gridParam.bucketNums[2] +
                               curAzimuthAngleIdx;
                auto findVoxel = hashTable.equal_range(voxelIdx);
                for (auto it = findVoxel.first; it != findVoxel.second; ++it) {
                    allNeighborIndices.emplace_back(it->second);
                }
            }
        }
    }

    return allNeighborIndices;
}

template <typename PointCloudType>
void CurvedVoxelClustering<PointCloudType>::mergeClusters(std::vector<int>& clusterIndices, int idx1, int idx2) const
{
    for (int i = 0; i < clusterIndices.size(); i++) {
        if (clusterIndices[i] == idx1) {
            clusterIndices[i] = idx2;
        }
    }
}

#undef INSTANTIATE_TEMPLATE
#define INSTANTIATE_TEMPLATE(DATA_TYPE) template class CurvedVoxelClustering<DATA_TYPE>;

INSTANTIATE_TEMPLATE(pcl::PointXYZ);
INSTANTIATE_TEMPLATE(pcl::PointXYZI);
INSTANTIATE_TEMPLATE(pcl::PointXYZRGB);
INSTANTIATE_TEMPLATE(pcl::PointNormal);
INSTANTIATE_TEMPLATE(pcl::PointXYZRGBNormal);
INSTANTIATE_TEMPLATE(pcl::PointXYZINormal);

#undef INSTANTIATE_TEMPLATE
}  // namespace perception

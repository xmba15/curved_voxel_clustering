/**
 * @file    ClusteringApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pcl/common/time.h>

#include <curved_voxel_clustering/curved_voxel_clustering.hpp>

#include "AppUtility.hpp"
#include "SimpleGroundRemover.hpp"

namespace
{
using PointCloudType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;
using CVCHandler = perception::CurvedVoxelClustering<PointCloudType>;
using GroundRemover = perception::SimpleGroundRemover<PointCloudType>;

auto viewer = ::initializeViewer();
auto timer = pcl::StopWatch();
int NUM_TEST = 10;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/pcl/file]\n";
        return EXIT_FAILURE;
    }

    const std::string pclFilePath = argv[1];
    PointCloudPtr inCloud(new PointCloud);
    if (pcl::io::loadPCDFile(pclFilePath, *inCloud) == -1) {
        std::cerr << "Failed to load pcl file\n";
        return EXIT_FAILURE;
    }

    std::cout << "total number of points: " << inCloud->size() << "\n";

    GroundRemover::Param groundRemoverParam;
    GroundRemover groundRemover(groundRemoverParam);
    PointCloudPtr ground(new PointCloud);
    groundRemover.run(inCloud, &*inCloud, &*ground);

    std::cout << "number of points after ground removal: " << inCloud->size() << "\n";

    CVCHandler::Param param;
    CVCHandler cvcHandler(param);

    std::vector<std::vector<int>> clusterIndices;
    timer.reset();
    for (int i = 0; i < NUM_TEST; ++i) {
        clusterIndices = cvcHandler.run(inCloud);
    }

    std::cout << "number of clusters: " << clusterIndices.size() << "\n";
    std::cout << "processing time for clustering (cpu): " << timer.getTime() / NUM_TEST << "[ms]\n";

    std::vector<PointCloudPtr> clusters = ::toClusters<PointCloudType>(clusterIndices, inCloud);

    auto [colors, colorHandlers] = ::initPclColorHandlers<PointCloudType>(clusters);

    for (std::size_t i = 0; i < clusters.size(); ++i) {
        const auto& cluster = clusters[i];
        viewer->addPointCloud<PointCloudType>(cluster, colorHandlers[i], std::to_string(i));
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, std::to_string(i));
    }

    viewer->addPointCloud<PointCloudType>(
        ground, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(ground, 0, 255, 0), "ground");

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return EXIT_SUCCESS;
}

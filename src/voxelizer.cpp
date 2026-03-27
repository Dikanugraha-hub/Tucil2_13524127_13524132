#include "voxelizer.h"
#include "octree.h"
#include <chrono>
#include <future>

VoxelizationResult voxelizeMesh(
    const vector<Vertex>& vertices,
    const vector<Face>& faces,
    const VoxelizerConfig& config
) {
    VoxelizationResult result;
    result.originalVertices = vertices;
    result.originalFaces = faces;
    const std::atomic_bool* cancelToken = config.cancelToken;

    if (vertices.empty() || faces.empty()) {
        return result;
    }

    auto t0 = chrono::high_resolution_clock::now();
    result.rootBox = computeBoundingBox(vertices);
    if (cancelToken && cancelToken->load()) {
        result.cancelled = true;
        return result;
    }

    OctreeNode root(result.rootBox, 0);
    if (config.useAsync) {
        auto worker = async(
            launch::async,
            [&root, &vertices, &faces, &config]() {
                buildOctree(&root, vertices, faces, config.maxDepth, config.cancelToken);
            }
        );
        worker.get();
    } else {
        buildOctree(&root, vertices, faces, config.maxDepth, config.cancelToken);
    }

    if (cancelToken && cancelToken->load()) {
        result.cancelled = true;
        auto t1 = chrono::high_resolution_clock::now();
        result.elapsedSeconds = chrono::duration<double>(t1 - t0).count();
        return result;
    }

    extractVoxels(&root, result.voxelVertices, result.voxelFaces, config.maxDepth, config.cancelToken);
    if (cancelToken && cancelToken->load()) {
        result.cancelled = true;
    }

    auto t1 = chrono::high_resolution_clock::now();
    result.elapsedSeconds = chrono::duration<double>(t1 - t0).count();
    result.totalVoxels = static_cast<int>(result.voxelFaces.size() / 12);
    return result;
}

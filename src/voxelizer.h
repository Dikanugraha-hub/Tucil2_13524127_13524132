#ifndef VOXELIZER_H
#define VOXELIZER_H

#include "types.h"
#include <atomic>

struct VoxelizerConfig {
    int maxDepth = 5;
    bool useAsync = true;
    const std::atomic_bool* cancelToken = nullptr;
};

VoxelizationResult voxelizeMesh(
    const vector<Vertex>& vertices,
    const vector<Face>& faces,
    const VoxelizerConfig& config
);

#endif

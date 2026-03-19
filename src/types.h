#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include <mutex>

struct Vertex {
    float x, y, z;
};

struct Face {
    int v1, v2, v3;
};

struct MyBoundingBox {
    Vertex minPoint;
    Vertex maxPoint;
};

struct VoxelizationResult {
    std::vector<Vertex> originalVertices;
    std::vector<Face> originalFaces;
    std::vector<Vertex> voxelVertices;
    std::vector<Face> voxelFaces;
    MyBoundingBox rootBox;
    std::string outputPath;
    int totalVoxels = 0;
    double elapsedSeconds = 0.0;
};

struct OctreeStats {
    std::vector<int> nodesCreatedPerDepth;
    std::vector<int> nodesPrunedPerDepth;
    std::mutex mtx;

    void init(int maxDepth) {
        nodesCreatedPerDepth.assign(maxDepth + 1, 0);
        nodesPrunedPerDepth.assign(maxDepth + 1, 0);
    }
};

extern OctreeStats stats;

struct OctreeNode {
    MyBoundingBox box;
    int depth;
    bool isLeaf;
    bool isOccupied;
    OctreeNode* children[8];

    OctreeNode(MyBoundingBox b, int d)
        : box(b), depth(d), isLeaf(true), isOccupied(false) {
        for (int i = 0; i < 8; i++) children[i] = nullptr;
    }

    ~OctreeNode() {
        for (int i = 0; i < 8; i++) delete children[i];
    }
};

#endif

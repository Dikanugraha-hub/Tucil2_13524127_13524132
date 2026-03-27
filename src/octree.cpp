#include "octree.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>

OctreeStats stats;

namespace {

bool boxesOverlap(const MyBoundingBox& a, const MyBoundingBox& b) {
    return (a.minPoint.x <= b.maxPoint.x && a.maxPoint.x >= b.minPoint.x)
        && (a.minPoint.y <= b.maxPoint.y && a.maxPoint.y >= b.minPoint.y)
        && (a.minPoint.z <= b.maxPoint.z && a.maxPoint.z >= b.minPoint.z);
}

MyBoundingBox childBox(const MyBoundingBox& box, int childIndex) {
    Vertex c = {
        (box.minPoint.x + box.maxPoint.x) * 0.5f,
        (box.minPoint.y + box.maxPoint.y) * 0.5f,
        (box.minPoint.z + box.maxPoint.z) * 0.5f
    };

    MyBoundingBox child = box;
    child.minPoint.x = (childIndex & 1) ? c.x : box.minPoint.x;
    child.maxPoint.x = (childIndex & 1) ? box.maxPoint.x : c.x;
    child.minPoint.y = (childIndex & 2) ? c.y : box.minPoint.y;
    child.maxPoint.y = (childIndex & 2) ? box.maxPoint.y : c.y;
    child.minPoint.z = (childIndex & 4) ? c.z : box.minPoint.z;
    child.maxPoint.z = (childIndex & 4) ? box.maxPoint.z : c.z;
    return child;
}

void appendCube(const MyBoundingBox& box, vector<Vertex>& vertices, vector<Face>& faces) {
    int base = static_cast<int>(vertices.size()) + 1;

    vertices.push_back({box.minPoint.x, box.minPoint.y, box.minPoint.z}); // 0
    vertices.push_back({box.maxPoint.x, box.minPoint.y, box.minPoint.z}); // 1
    vertices.push_back({box.maxPoint.x, box.maxPoint.y, box.minPoint.z}); // 2
    vertices.push_back({box.minPoint.x, box.maxPoint.y, box.minPoint.z}); // 3
    vertices.push_back({box.minPoint.x, box.minPoint.y, box.maxPoint.z}); // 4
    vertices.push_back({box.maxPoint.x, box.minPoint.y, box.maxPoint.z}); // 5
    vertices.push_back({box.maxPoint.x, box.maxPoint.y, box.maxPoint.z}); // 6
    vertices.push_back({box.minPoint.x, box.maxPoint.y, box.maxPoint.z}); // 7

    auto add = [&](int a, int b, int c) { faces.push_back({base + a, base + b, base + c}); };
    add(0, 1, 2); add(0, 2, 3); // -Z
    add(4, 6, 5); add(4, 7, 6); // +Z
    add(0, 4, 5); add(0, 5, 1); // -Y
    add(3, 2, 6); add(3, 6, 7); // +Y
    add(1, 5, 6); add(1, 6, 2); // +X
    add(0, 3, 7); add(0, 7, 4); // -X
}

MyBoundingBox triangleBounds(const Vertex& v0, const Vertex& v1, const Vertex& v2) {
    MyBoundingBox triBox{};
    triBox.minPoint.x = std::min(v0.x, std::min(v1.x, v2.x));
    triBox.minPoint.y = std::min(v0.y, std::min(v1.y, v2.y));
    triBox.minPoint.z = std::min(v0.z, std::min(v1.z, v2.z));
    triBox.maxPoint.x = std::max(v0.x, std::max(v1.x, v2.x));
    triBox.maxPoint.y = std::max(v0.y, std::max(v1.y, v2.y));
    triBox.maxPoint.z = std::max(v0.z, std::max(v1.z, v2.z));
    return triBox;
}

Vertex sub(const Vertex& a, const Vertex& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vertex cross(const Vertex& a, const Vertex& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

float dot(const Vertex& a, const Vertex& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float absf(float v) {
    return std::fabs(v);
}

bool satAxisOverlap(const Vertex& axis, const Vertex& v0, const Vertex& v1, const Vertex& v2, const Vertex& halfSize) {
    const float axisLen2 = dot(axis, axis);
    if (axisLen2 < 1e-12f) return true;

    float p0 = dot(v0, axis);
    float p1 = dot(v1, axis);
    float p2 = dot(v2, axis);
    float minP = std::min(p0, std::min(p1, p2));
    float maxP = std::max(p0, std::max(p1, p2));

    float r = halfSize.x * absf(axis.x) + halfSize.y * absf(axis.y) + halfSize.z * absf(axis.z);
    return !(minP > r || maxP < -r);
}

struct TriangleData {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    MyBoundingBox bounds;
};

void incrementCreated(int depth) {
    if (depth < 0 || depth >= static_cast<int>(stats.nodesCreatedPerDepth.size())) return;
    std::lock_guard<mutex> lock(stats.mtx);
    stats.nodesCreatedPerDepth[depth]++;
}

void incrementPruned(int depth) {
    if (depth < 0 || depth >= static_cast<int>(stats.nodesPrunedPerDepth.size())) return;
    std::lock_guard<mutex> lock(stats.mtx);
    stats.nodesPrunedPerDepth[depth]++;
}

void buildOctreeFiltered(
    OctreeNode* node,
    const vector<TriangleData>& triangles,
    const vector<int>& candidateFaces,
    int maxDepth,
    const std::atomic_bool* cancelToken
) {
    if (!node) return;
    if (cancelToken && cancelToken->load()) {
        node->isLeaf = true;
        node->isOccupied = false;
        return;
    }

    vector<int> intersectingFaces;
    intersectingFaces.reserve(candidateFaces.size());
    for (int faceIndex : candidateFaces) {
        if (cancelToken && cancelToken->load()) {
            node->isLeaf = true;
            node->isOccupied = false;
            return;
        }
        const TriangleData& tri = triangles[faceIndex];
        if (!boxesOverlap(tri.bounds, node->box)) continue;
        if (triangleBoxIntersect(tri.v0, tri.v1, tri.v2, node->box)) {
            intersectingFaces.push_back(faceIndex);
        }
    }
    if (intersectingFaces.empty()) {
        node->isLeaf = true;
        node->isOccupied = false;
        return;
    }

    node->isOccupied = true;
    if (node->depth >= maxDepth) {
        node->isLeaf = true;
        return;
    }

    array<MyBoundingBox, 8> childBoxes;
    array<vector<int>, 8> childFaceLists;
    bool anyChildOccupied = false;

    for (int i = 0; i < 8; ++i) {
        childBoxes[i] = childBox(node->box, i);
    }

    for (int faceIndex : intersectingFaces) {
        const MyBoundingBox& triBox = triangles[faceIndex].bounds;
        for (int c = 0; c < 8; ++c) {
            if (boxesOverlap(triBox, childBoxes[c])) {
                childFaceLists[c].push_back(faceIndex);
                anyChildOccupied = true;
            }
        }
    }

    if (!anyChildOccupied) {
        node->isLeaf = true;
        return;
    }

    node->isLeaf = false;
    for (int c = 0; c < 8; ++c) {
        if (cancelToken && cancelToken->load()) {
            node->isLeaf = true;
            node->isOccupied = false;
            return;
        }
        if (childFaceLists[c].empty()) {
            incrementPruned(node->depth + 1);
            continue;
        }
        node->children[c] = new OctreeNode(childBoxes[c], node->depth + 1);
        incrementCreated(node->depth + 1);
        buildOctreeFiltered(node->children[c], triangles, childFaceLists[c], maxDepth, cancelToken);
    }
}

}

MyBoundingBox computeBoundingBox(const vector<Vertex>& vertices) {
    MyBoundingBox box{};
    if (vertices.empty()) return box;

    box.minPoint = vertices[0];
    box.maxPoint = vertices[0];
    for (const auto& v : vertices) {
        box.minPoint.x = std::min(box.minPoint.x, v.x);
        box.minPoint.y = std::min(box.minPoint.y, v.y);
        box.minPoint.z = std::min(box.minPoint.z, v.z);
        box.maxPoint.x = std::max(box.maxPoint.x, v.x);
        box.maxPoint.y = std::max(box.maxPoint.y, v.y);
        box.maxPoint.z = std::max(box.maxPoint.z, v.z);
    }
    return box;
}

bool triangleBoxIntersect(const Vertex& v0, const Vertex& v1, const Vertex& v2, const MyBoundingBox& box) {

    Vertex center{
        (box.minPoint.x + box.maxPoint.x) * 0.5f,
        (box.minPoint.y + box.maxPoint.y) * 0.5f,
        (box.minPoint.z + box.maxPoint.z) * 0.5f
    };
    Vertex half{
        (box.maxPoint.x - box.minPoint.x) * 0.5f,
        (box.maxPoint.y - box.minPoint.y) * 0.5f,
        (box.maxPoint.z - box.minPoint.z) * 0.5f
    };
    if (half.x < 1e-9f || half.y < 1e-9f || half.z < 1e-9f) return false;

    Vertex tv0 = sub(v0, center);
    Vertex tv1 = sub(v1, center);
    Vertex tv2 = sub(v2, center);

    Vertex e0 = sub(tv1, tv0);
    Vertex e1 = sub(tv2, tv1);
    Vertex e2 = sub(tv0, tv2);

    const array<Vertex, 3> boxAxes = {Vertex{1.0f, 0.0f, 0.0f}, Vertex{0.0f, 1.0f, 0.0f}, Vertex{0.0f, 0.0f, 1.0f}};
    const array<Vertex, 3> edges = {e0, e1, e2};

    for (const auto& e : edges) {
        for (const auto& a : boxAxes) {
            Vertex axis = cross(e, a);
            if (!satAxisOverlap(axis, tv0, tv1, tv2, half)) return false;
        }
    }

    if (!satAxisOverlap(boxAxes[0], tv0, tv1, tv2, half)) return false;
    if (!satAxisOverlap(boxAxes[1], tv0, tv1, tv2, half)) return false;
    if (!satAxisOverlap(boxAxes[2], tv0, tv1, tv2, half)) return false;

    Vertex normal = cross(e0, e1);
    if (!satAxisOverlap(normal, tv0, tv1, tv2, half)) return false;

    return true;
}

bool meshIntersectsBox(const MyBoundingBox& box, const vector<Vertex>& vertices, const vector<Face>& faces) {
    for (const auto& f : faces) {
        int i1 = f.v1 - 1, i2 = f.v2 - 1, i3 = f.v3 - 1;
        if (i1 < 0 || i2 < 0 || i3 < 0) continue;
        if (i1 >= static_cast<int>(vertices.size())
            || i2 >= static_cast<int>(vertices.size())
            || i3 >= static_cast<int>(vertices.size())) continue;
        if (triangleBoxIntersect(vertices[i1], vertices[i2], vertices[i3], box)) {
            return true;
        }
    }
    return false;
}

void buildOctree(
    OctreeNode* node,
    const vector<Vertex>& vertices,
    const vector<Face>& faces,
    int maxDepth,
    const std::atomic_bool* cancelToken
) {
    if (!node) return;
    if (cancelToken && cancelToken->load()) return;
    stats.init(maxDepth);
    incrementCreated(node->depth);
    vector<TriangleData> triangles;
    vector<int> candidates;
    triangles.reserve(faces.size());
    candidates.reserve(faces.size());

    for (size_t i = 0; i < faces.size(); ++i) {
        if (cancelToken && cancelToken->load()) return;
        const Face& f = faces[i];
        int i1 = f.v1 - 1;
        int i2 = f.v2 - 1;
        int i3 = f.v3 - 1;
        if (i1 < 0 || i2 < 0 || i3 < 0) continue;
        if (i1 >= static_cast<int>(vertices.size())
            || i2 >= static_cast<int>(vertices.size())
            || i3 >= static_cast<int>(vertices.size())) continue;
        TriangleData tri{};
        tri.v0 = vertices[i1];
        tri.v1 = vertices[i2];
        tri.v2 = vertices[i3];
        tri.bounds = triangleBounds(tri.v0, tri.v1, tri.v2);
        triangles.push_back(tri);
        candidates.push_back(static_cast<int>(triangles.size()) - 1);
    }

    buildOctreeFiltered(node, triangles, candidates, maxDepth, cancelToken);
}

void extractVoxels(
    OctreeNode* node,
    vector<Vertex>& voxelVertices,
    vector<Face>& voxelFaces,
    int maxDepth,
    const std::atomic_bool* cancelToken
) {
    if (!node) return;
    if (cancelToken && cancelToken->load()) return;
    if (!node->isOccupied) return;

    if (node->isLeaf || node->depth >= maxDepth) {
        appendCube(node->box, voxelVertices, voxelFaces);
        return;
    }

    for (int i = 0; i < 8; ++i) {
        if (cancelToken && cancelToken->load()) return;
        extractVoxels(node->children[i], voxelVertices, voxelFaces, maxDepth, cancelToken);
    }
}

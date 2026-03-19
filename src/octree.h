#ifndef OCTREE_H
#define OCTREE_H

#include "types.h"

MyBoundingBox computeBoundingBox(const vector<Vertex>& vertices);

bool triangleBoxIntersect(const Vertex& v0, const Vertex& v1, const Vertex& v2, const MyBoundingBox& box);

bool meshIntersectsBox(const MyBoundingBox& box, const vector<Vertex>& vertices, const vector<Face>& faces);

void buildOctree(OctreeNode* node, const vector<Vertex>& vertices, const vector<Face>& faces, int maxDepth);

void extractVoxels(OctreeNode* node, vector<Vertex>& voxelVertices, vector<Face>& voxelFaces, int maxDepth);

#endif

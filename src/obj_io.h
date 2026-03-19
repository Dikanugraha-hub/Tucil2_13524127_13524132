#ifndef OBJ_IO_H
#define OBJ_IO_H

#include "types.h"
#include <string>
#include <vector>

bool parseObj(const string& filename, vector<Vertex>& vertices, vector<Face>& faces);

void exportToObj(const string& filename, const vector<Vertex>& vertices, const vector<Face>& faces);

string getOutputFilename(const string& inputPath);

bool hasObjExtension(const string& path);

bool isVoxelizedObjName(const string& path);

vector<string> collectObjCandidates(const string& initialPath);

#endif

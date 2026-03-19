#include "obj_io.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

bool parseObj(const string& filename, vector<Vertex>& vertices, vector<Face>& faces) {
ifstream file(filename);
    if (!file.is_open()) {
cerr << "Gagal membuka file: " << filename << "\n";
        return false;
    }

string line;
    while (getline(file, line)) {
istringstream iss(line);
string prefix;
        iss >> prefix;

        if (prefix == "v") {
            Vertex v;
            if (iss >> v.x >> v.y >> v.z) vertices.push_back(v);
        } else if (prefix == "f") {
            Face f;
            if (iss >> f.v1 >> f.v2 >> f.v3) faces.push_back(f);
        }
    }
    return true;
}

void exportToObj(const string& filename, const vector<Vertex>& vertices, const vector<Face>& faces) {
ofstream outFile(filename);
    if (!outFile.is_open()) {
cerr << "Gagal membuat file output: " << filename << "\n";
        return;
    }

    for (const auto& v : vertices) {
        outFile << "v " << v.x << " " << v.y << " " << v.z << "\n";
    }
    for (const auto& f : faces) {
        outFile << "f " << f.v1 << " " << f.v2 << " " << f.v3 << "\n";
    }

    outFile.close();
cout << "File berhasil disimpan di: " << filename << "\n";
}

string getOutputFilename(const string& inputPath) {
    size_t dotPos = inputPath.find_last_of('.');
    if (dotPos == string::npos) return inputPath + "-voxelized.obj";
    return inputPath.substr(0, dotPos) + "-voxelized.obj";
}

bool hasObjExtension(const string& path) {
    if (path.size() < 4) return false;
string ext = path.substr(path.size() - 4);
transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".obj";
}

bool isVoxelizedObjName(const string& path) {
    const string suffix = "-voxelized.obj";
    if (path.size() < suffix.size()) return false;
    return path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0;
}

vector<string> collectObjCandidates(const string& initialPath) {
vector<string> candidates;
vector<filesystem::path> scanDirs = {
filesystem::current_path(),
filesystem::current_path() / "test"
    };

filesystem::path initial(initialPath);
    if (initial.has_parent_path()) {
        scanDirs.push_back(initial.parent_path());
    }

    for (const auto& dir : scanDirs) {
error_code ec;
        if (!filesystem::exists(dir, ec) || !filesystem::is_directory(dir, ec)) continue;

        for (const auto& entry : filesystem::directory_iterator(dir, ec)) {
            if (ec || !entry.is_regular_file()) continue;
string file = entry.path().string();
            if (hasObjExtension(file) && !isVoxelizedObjName(file)) {
                candidates.push_back(file);
            }
        }
    }

    if (hasObjExtension(initialPath) && !isVoxelizedObjName(initialPath)) {
        candidates.push_back(initialPath);
    }

sort(candidates.begin(), candidates.end());
    candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
}

#ifndef VIEWER_H
#define VIEWER_H

#include "types.h"
#include <string>
#include <vector>

struct ViewerConfig {
    int width = 1280;
    int height = 720;
    bool startSmooth = false;
};

void runViewer(const VoxelizationResult& result, const ViewerConfig& config);
void runVoxelizerGui(
    const vector<string>& objCandidates,
    int initialDepth,
    const string& initialPath,
    bool startSmooth
);

#endif

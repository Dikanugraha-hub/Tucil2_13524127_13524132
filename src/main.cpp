#include "obj_io.h"
#include "viewer.h"
#include <iostream>
#include <string>

namespace {

struct CliOptions {
    string inputPath;
    int depth = 5;
    bool smoothNormals = false;
};

void printUsage(const string& exeName) {
    cout << "Usage: " << exeName << " [input.obj] [options]\n";
    cout << "Options:\n";
    cout << "  --depth <n>     Kedalaman octree (default: 5)\n";
    cout << "  --smooth        Start viewer dengan smooth normal\n";
}

bool parseCli(int argc, char** argv, CliOptions& options) {
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--depth") {
            if (i + 1 >= argc) return false;
            options.depth = stoi(argv[++i]);
            if (options.depth < 0) options.depth = 0;
        } else if (arg == "--smooth") {
            options.smoothNormals = true;
        } else if (!arg.empty() && arg[0] != '-') {
            options.inputPath = arg;
        } else {
            return false;
        }
    }
    return true;
}

}

int main(int argc, char** argv) {
    CliOptions options;
    if (!parseCli(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    auto candidates = collectObjCandidates(options.inputPath);
    if (candidates.empty()) {
        cout << "Tidak ditemukan file OBJ yang valid.\n";
        return 1;
    }

    runVoxelizerGui(candidates, options.depth, options.inputPath, options.smoothNormals);

    return 0;
}

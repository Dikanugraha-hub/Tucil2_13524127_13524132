#include "viewer.h"
#include "obj_io.h"
#include "voxelizer.h"
#include "raylib.h"
#include "rlgl.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <string>
#include <vector>

namespace {

const Font* gUiFont = nullptr;
float gUiScale = 1.35f;

Vector3 toVec3(const Vertex& v) {
    return {v.x, v.y, v.z};
}

Vector3 subtract(const Vector3& a, const Vector3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vector3 cross(const Vector3& a, const Vector3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

Vector3 normalize(const Vector3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len <= 1e-6f) return {0.0f, 1.0f, 0.0f};
    return {v.x / len, v.y / len, v.z / len};
}

Mesh buildMeshFromFaces(
    const vector<Vertex>& vertices,
    const vector<Face>& faces,
    bool smoothNormals
) {
    Mesh mesh = {};
    if (vertices.empty() || faces.empty()) return mesh;

    vector<Vector3> perVertexNormals(vertices.size(), {0.0f, 0.0f, 0.0f});
    if (smoothNormals) {
        for (const auto& f : faces) {
            int i1 = f.v1 - 1;
            int i2 = f.v2 - 1;
            int i3 = f.v3 - 1;
            if (i1 < 0 || i2 < 0 || i3 < 0) continue;
            if (i1 >= static_cast<int>(vertices.size())
                || i2 >= static_cast<int>(vertices.size())
                || i3 >= static_cast<int>(vertices.size())) continue;

            Vector3 a = toVec3(vertices[i1]);
            Vector3 b = toVec3(vertices[i2]);
            Vector3 c = toVec3(vertices[i3]);
            Vector3 n = normalize(cross(subtract(b, a), subtract(c, a)));
            perVertexNormals[i1] = {perVertexNormals[i1].x + n.x, perVertexNormals[i1].y + n.y, perVertexNormals[i1].z + n.z};
            perVertexNormals[i2] = {perVertexNormals[i2].x + n.x, perVertexNormals[i2].y + n.y, perVertexNormals[i2].z + n.z};
            perVertexNormals[i3] = {perVertexNormals[i3].x + n.x, perVertexNormals[i3].y + n.y, perVertexNormals[i3].z + n.z};
        }
        for (auto& n : perVertexNormals) n = normalize(n);
    }

    int triCount = static_cast<int>(faces.size());
    mesh.vertexCount = triCount * 3;
    mesh.triangleCount = triCount;
    mesh.vertices = static_cast<float*>(MemAlloc(sizeof(float) * mesh.vertexCount * 3));
    mesh.normals = static_cast<float*>(MemAlloc(sizeof(float) * mesh.vertexCount * 3));

    int outVertex = 0;
    for (const auto& f : faces) {
        array<int, 3> idx = {f.v1 - 1, f.v2 - 1, f.v3 - 1};
        if (idx[0] < 0 || idx[1] < 0 || idx[2] < 0) continue;
        if (idx[0] >= static_cast<int>(vertices.size())
            || idx[1] >= static_cast<int>(vertices.size())
            || idx[2] >= static_cast<int>(vertices.size())) continue;

        Vector3 va = toVec3(vertices[idx[0]]);
        Vector3 vb = toVec3(vertices[idx[1]]);
        Vector3 vc = toVec3(vertices[idx[2]]);
        Vector3 flat = normalize(cross(subtract(vb, va), subtract(vc, va)));

        for (int k = 0; k < 3; ++k) {
            Vector3 p = toVec3(vertices[idx[k]]);
            Vector3 n = smoothNormals ? perVertexNormals[idx[k]] : flat;
            mesh.vertices[outVertex * 3 + 0] = p.x;
            mesh.vertices[outVertex * 3 + 1] = p.y;
            mesh.vertices[outVertex * 3 + 2] = p.z;
            mesh.normals[outVertex * 3 + 0] = n.x;
            mesh.normals[outVertex * 3 + 1] = n.y;
            mesh.normals[outVertex * 3 + 2] = n.z;
            outVertex++;
        }
    }

    mesh.vertexCount = outVertex;
    mesh.triangleCount = outVertex / 3;
    UploadMesh(&mesh, false);
    return mesh;
}

Shader createLightingShader() {
    const char* vsCode = R"(
    #version 330
    in vec3 vertexPosition;
    in vec3 vertexNormal;
    uniform mat4 mvp;
    uniform mat4 matModel;
    out vec3 fragPos;
    out vec3 fragNormal;
    void main() {
        fragPos = vec3(matModel * vec4(vertexPosition, 1.0));
        fragNormal = normalize(mat3(transpose(inverse(matModel))) * vertexNormal);
        gl_Position = mvp * vec4(vertexPosition, 1.0);
    })";

    const char* fsCode = R"(
    #version 330
    in vec3 fragPos;
    in vec3 fragNormal;
    out vec4 finalColor;
    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec4 objectColor;
    void main() {
        vec3 N = normalize(fragNormal);
        vec3 L = normalize(lightPos - fragPos);
        vec3 V = normalize(viewPos - fragPos);
        vec3 R = reflect(-L, N);

        float ambient = 0.25;
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(V, R), 0.0), 24.0) * 0.35;
        vec3 lit = objectColor.rgb * (ambient + diff) + vec3(spec);
        finalColor = vec4(lit, objectColor.a);
    })";

    return LoadShaderFromMemory(vsCode, fsCode);
}

float maxComponent(float a, float b, float c) {
    return std::max(a, std::max(b, c));
}

string baseName(const string& path) {
    filesystem::path p(path);
    string name = p.filename().string();
    return name.empty() ? path : name;
}

struct BuildOutput {
    bool success = false;
    string message;
    VoxelizationResult result;
};

struct OriginalPreviewOutput {
    bool success = false;
    string message;
    string inputPath;
    vector<Vertex> vertices;
    vector<Face> faces;
    MyBoundingBox box{};
};

MyBoundingBox computeBoxFromVertices(const vector<Vertex>& vertices) {
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

OriginalPreviewOutput loadOriginalPreview(const string& inputPath) {
    OriginalPreviewOutput out;
    out.inputPath = inputPath;
    if (!parseObj(inputPath, out.vertices, out.faces)) {
        out.message = "Gagal membaca OBJ.";
        return out;
    }
    if (out.vertices.empty() || out.faces.empty()) {
        out.message = "OBJ kosong atau tidak valid.";
        return out;
    }
    out.box = computeBoxFromVertices(out.vertices);
    out.success = true;
    out.message = "Preview original siap.";
    return out;
}

BuildOutput processVoxelization(
    const string& inputPath,
    const vector<Vertex>& vertices,
    const vector<Face>& faces,
    int depth,
    std::atomic_bool* cancelToken
) {
    BuildOutput out;
    auto t0 = chrono::high_resolution_clock::now();

    VoxelizerConfig cfg;
    cfg.maxDepth = std::max(0, depth);
    cfg.useAsync = false;
    cfg.cancelToken = cancelToken;

    out.result = voxelizeMesh(vertices, faces, cfg);
    if (out.result.cancelled || (cancelToken && cancelToken->load())) {
        out.message = "Proses voxelisasi dibatalkan.";
        return out;
    }
    if (out.result.voxelVertices.empty() || out.result.voxelFaces.empty()) {
        out.message = "Vokselisasi tidak menghasilkan mesh.";
        return out;
    }

    out.result.outputPath = getOutputFilename(inputPath);
    exportToObj(out.result.outputPath, out.result.voxelVertices, out.result.voxelFaces);
    auto t1 = chrono::high_resolution_clock::now();
    out.result.elapsedSeconds = chrono::duration<double>(t1 - t0).count();
    if (cancelToken && cancelToken->load()) {
        out.message = "Proses voxelisasi dibatalkan.";
        return out;
    }
    out.success = true;
    out.message = "Selesai";
    return out;
}

float measureTextWithCurrentFont(const string& text, float fontSize) {
    const float scaledSize = fontSize * gUiScale;
    if (gUiFont && gUiFont->texture.id != 0) {
        return MeasureTextEx(*gUiFont, text.c_str(), scaledSize, 1.2f).x;
    }
    return static_cast<float>(MeasureText(text.c_str(), static_cast<int>(scaledSize)));
}

void drawTextWithCurrentFont(const string& text, float x, float y, float fontSize, Color color) {
    const float scaledSize = fontSize * gUiScale;
    if (gUiFont && gUiFont->texture.id != 0) {
        DrawTextEx(*gUiFont, text.c_str(), Vector2{x, y}, scaledSize, 1.2f, color);
        return;
    }
    DrawText(text.c_str(), static_cast<int>(x), static_cast<int>(y), static_cast<int>(scaledSize), color);
}

float drawWrappedTextWithCurrentFont(
    const string& text,
    float x,
    float y,
    float maxWidth,
    float fontSize,
    float lineSpacing,
    Color color
) {
    if (text.empty()) return 0.0f;

    vector<string> words;
    string current;
    for (char ch : text) {
        if (ch == ' ') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(" ");
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) words.push_back(current);

    vector<string> lines;
    string line;
    for (const auto& token : words) {
        if (token == " ") {
            if (!line.empty()) line.push_back(' ');
            continue;
        }

        if (measureTextWithCurrentFont(token, fontSize) > maxWidth) {
            if (!line.empty()) {
                while (!line.empty() && line.back() == ' ') line.pop_back();
                lines.push_back(line);
                line.clear();
            }

            string chunk;
            for (char ch : token) {
                string next = chunk;
                next.push_back(ch);
                if (!chunk.empty() && measureTextWithCurrentFont(next, fontSize) > maxWidth) {
                    lines.push_back(chunk);
                    chunk.clear();
                }
                chunk.push_back(ch);
            }
            if (!chunk.empty()) line = chunk;
            continue;
        }

        string candidate = line.empty() ? token : (line + token);
        if (!line.empty() && measureTextWithCurrentFont(candidate, fontSize) > maxWidth) {
            while (!line.empty() && line.back() == ' ') line.pop_back();
            lines.push_back(line);
            line = token;
        } else {
            line = candidate;
        }
    }
    while (!line.empty() && line.back() == ' ') line.pop_back();
    if (!line.empty()) lines.push_back(line);
    if (lines.empty()) lines.push_back(text);

    float cursorY = y;
    const float scaledSize = fontSize * gUiScale;
    const float scaledSpacing = lineSpacing * gUiScale;
    for (const auto& ln : lines) {
        drawTextWithCurrentFont(ln, x, cursorY, fontSize, color);
        cursorY += scaledSize + scaledSpacing;
    }
    return cursorY - y;
}

bool drawButton(Rectangle bounds, const string& label, bool enabled) {
    Vector2 mouse = GetMousePosition();
    bool hover = CheckCollisionPointRec(mouse, bounds);
    Color fill = enabled ? (hover ? Color{72, 88, 133, 255} : Color{48, 58, 95, 255})
                         : Color{55, 55, 55, 255};
    DrawRectangleRec(bounds, fill);
    DrawRectangleLinesEx(bounds, 1.0f, enabled ? Color{150, 165, 210, 255} : DARKGRAY);
    const float fontSize = 28.0f;
    const float scaledSize = fontSize * gUiScale;
    float tw = measureTextWithCurrentFont(label, fontSize);
    drawTextWithCurrentFont(
        label,
        bounds.x + (bounds.width - tw) * 0.5f,
        bounds.y + (bounds.height - scaledSize) * 0.5f,
        fontSize,
        RAYWHITE
    );
    return enabled && hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

bool drawDropdown(
    Rectangle bounds,
    const vector<string>& options,
    int& selectedIndex,
    int& firstVisible,
    bool& open,
    bool enabled
) {
    bool changed = false;
    Vector2 mouse = GetMousePosition();
    bool hoverMain = CheckCollisionPointRec(mouse, bounds);
    Color fill = enabled ? (hoverMain ? Color{42, 47, 61, 255} : Color{35, 38, 48, 255})
                         : Color{55, 55, 55, 255};

    DrawRectangleRec(bounds, fill);
    DrawRectangleLinesEx(bounds, 1.0f, Color{90, 102, 139, 255});
    const string selectedText = options.empty() ? string("-") : baseName(options[selectedIndex]);
    DrawText(selectedText.c_str(), static_cast<int>(bounds.x + 10), static_cast<int>(bounds.y + 12), 18, RAYWHITE);
    DrawText(open ? "^" : "v", static_cast<int>(bounds.x + bounds.width - 20), static_cast<int>(bounds.y + 12), 18, RAYWHITE);

    if (enabled && hoverMain && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        open = !open;
    }

    if (open && enabled) {
        const float rowHeight = 34.0f;
        const int maxVisible = 7;
        int visibleCount = std::min(static_cast<int>(options.size()), maxVisible);
        if (firstVisible < 0) firstVisible = 0;
        if (firstVisible > static_cast<int>(options.size()) - visibleCount) {
            firstVisible = std::max(0, static_cast<int>(options.size()) - visibleCount);
        }
        Rectangle listRect{bounds.x, bounds.y + bounds.height + 4.0f, bounds.width, rowHeight * visibleCount};
        DrawRectangleRec(listRect, Color{28, 31, 40, 245});
        DrawRectangleLinesEx(listRect, 1.0f, Color{90, 102, 139, 255});

        if (CheckCollisionPointRec(mouse, listRect)) {
            float wheel = GetMouseWheelMove();
            if (wheel > 0.0f && firstVisible > 0) firstVisible--;
            if (wheel < 0.0f && firstVisible < static_cast<int>(options.size()) - visibleCount) firstVisible++;
        }

        for (int i = 0; i < visibleCount; ++i) {
            Rectangle row{listRect.x, listRect.y + i * rowHeight, listRect.width, rowHeight};
            bool rowHover = CheckCollisionPointRec(mouse, row);
            int optionIndex = firstVisible + i;
            if (rowHover) DrawRectangleRec(row, Color{54, 63, 88, 255});
            DrawText(baseName(options[optionIndex]).c_str(), static_cast<int>(row.x + 10), static_cast<int>(row.y + 8), 18, RAYWHITE);
            if (rowHover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
                selectedIndex = optionIndex;
                open = false;
                changed = true;
            }
        }

        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !hoverMain && !CheckCollisionPointRec(mouse, listRect)) {
            open = false;
        }
    }

    return changed;
}

vector<string> buildDepthStatsLines(const vector<int>& values) {
    vector<string> lines;
    for (int d = 1; d < static_cast<int>(values.size()); ++d) {
        lines.push_back(TextFormat("%d : %d", d, values[d]));
    }
    if (lines.empty()) lines.push_back("1 : 0");
    return lines;
}

string compactStats(const vector<string>& lines, size_t maxItems = 5) {
    if (lines.empty()) return "-";
    string out;
    size_t take = std::min(lines.size(), maxItems);
    for (size_t i = 0; i < take; ++i) {
        if (!out.empty()) out += " | ";
        out += lines[i];
    }
    if (lines.size() > maxItems) out += " | ...";
    return out;
}

}

void runViewer(const VoxelizationResult& result, const ViewerConfig& config) {
    if (result.voxelVertices.empty() || result.voxelFaces.empty()) {
        return;
    }

    InitWindow(config.width, config.height, "Viewer");
    SetTargetFPS(60);

    float sx = result.rootBox.maxPoint.x - result.rootBox.minPoint.x;
    float sy = result.rootBox.maxPoint.y - result.rootBox.minPoint.y;
    float sz = result.rootBox.maxPoint.z - result.rootBox.minPoint.z;
    float sceneSize = maxComponent(sx, sy, sz);
    if (sceneSize < 1e-3f) sceneSize = 1.0f;

    Vector3 target = {
        (result.rootBox.minPoint.x + result.rootBox.maxPoint.x) * 0.5f,
        (result.rootBox.minPoint.y + result.rootBox.maxPoint.y) * 0.5f,
        (result.rootBox.minPoint.z + result.rootBox.maxPoint.z) * 0.5f
    };
    Vector3 homePos = {target.x + sceneSize * 1.6f, target.y + sceneSize * 1.1f, target.z + sceneSize * 1.6f};

    Camera3D camera = {};
    camera.position = homePos;
    camera.target = target;
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;
    Vector3 initialOffset = subtract(homePos, target);
    float orbitRadius = sqrtf(
        initialOffset.x * initialOffset.x
        + initialOffset.y * initialOffset.y
        + initialOffset.z * initialOffset.z
    );
    if (orbitRadius < 1e-3f) orbitRadius = sceneSize * 2.0f;
    const float minRadius = sceneSize * 0.15f;
    const float maxRadius = sceneSize * 12.0f;
    float orbitYaw = atan2f(initialOffset.z, initialOffset.x);
    float orbitPitch = asinf(initialOffset.y / orbitRadius);

    bool smooth = config.startSmooth;
    bool showGrid = true;
    bool showWire = false;

    Mesh mesh = buildMeshFromFaces(result.voxelVertices, result.voxelFaces, smooth);
    Model model = LoadModelFromMesh(mesh);
    Shader shader = createLightingShader();
    model.materials[0].shader = shader;
    model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = ORANGE;

    int lightPosLoc = GetShaderLocation(shader, "lightPos");
    int viewPosLoc = GetShaderLocation(shader, "viewPos");
    int objectColorLoc = GetShaderLocation(shader, "objectColor");
    const float objectColor[4] = {1.0f, 0.6f, 0.1f, 1.0f};
    if (objectColorLoc >= 0) {
        SetShaderValue(shader, objectColorLoc, objectColor, SHADER_UNIFORM_VEC4);
    }

    while (!WindowShouldClose()) {
        Vector2 mouseDelta = GetMouseDelta();
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            orbitYaw -= mouseDelta.x * 0.0045f;
            orbitPitch += mouseDelta.y * 0.0035f;
        }

        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            orbitRadius *= (1.0f - wheel * 0.1f);
        }

        orbitPitch = std::clamp(orbitPitch, -1.35f, 1.35f);
        orbitRadius = std::clamp(orbitRadius, minRadius, maxRadius);
        camera.position = {
            target.x + orbitRadius * cosf(orbitPitch) * cosf(orbitYaw),
            target.y + orbitRadius * sinf(orbitPitch),
            target.z + orbitRadius * cosf(orbitPitch) * sinf(orbitYaw)
        };
        camera.target = target;

        if (IsKeyPressed(KEY_F)) {
            smooth = !smooth;
            UnloadModel(model);
            mesh = buildMeshFromFaces(result.voxelVertices, result.voxelFaces, smooth);
            model = LoadModelFromMesh(mesh);
            model.materials[0].shader = shader;
            model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = ORANGE;
            if (objectColorLoc >= 0) {
                SetShaderValue(shader, objectColorLoc, objectColor, SHADER_UNIFORM_VEC4);
            }
        }
        if (IsKeyPressed(KEY_G)) showGrid = !showGrid;
        if (IsKeyPressed(KEY_W)) showWire = !showWire;
        if (IsKeyPressed(KEY_R)) {
            camera.position = homePos;
            camera.target = target;
            orbitRadius = sqrtf(
                initialOffset.x * initialOffset.x
                + initialOffset.y * initialOffset.y
                + initialOffset.z * initialOffset.z
            );
            if (orbitRadius < 1e-3f) orbitRadius = sceneSize * 2.0f;
            orbitYaw = atan2f(initialOffset.z, initialOffset.x);
            orbitPitch = asinf(initialOffset.y / orbitRadius);
        }

        Vector3 lightPos = {target.x + sceneSize * 2.0f, target.y + sceneSize * 2.0f, target.z + sceneSize * 1.3f};
        SetShaderValue(shader, lightPosLoc, &lightPos.x, SHADER_UNIFORM_VEC3);
        SetShaderValue(shader, viewPosLoc, &camera.position.x, SHADER_UNIFORM_VEC3);

        BeginDrawing();
        Color bgColor{20, 22, 28, 255};
        ClearBackground(bgColor);
        BeginMode3D(camera);

        if (showGrid) DrawGrid(20, sceneSize / 10.0f);
        Vector3 modelPosition{0.0f, 0.0f, 0.0f};
        DrawModel(model, modelPosition, 1.0f, WHITE);

        if (showWire) {
            rlDisableBackfaceCulling();
            DrawModelWires(model, modelPosition, 1.0f, BLACK);
            rlEnableBackfaceCulling();
        }
        BoundingBox rootBounds{
            {result.rootBox.minPoint.x, result.rootBox.minPoint.y, result.rootBox.minPoint.z},
            {result.rootBox.maxPoint.x, result.rootBox.maxPoint.y, result.rootBox.maxPoint.z}
        };
        DrawBoundingBox(rootBounds, GRAY);

        EndMode3D();

        DrawRectangle(12, 12, 430, 132, Fade(BLACK, 0.55f));
        DrawText("Viewer", 24, 24, 22, RAYWHITE);
        DrawText(TextFormat("Output: %s", result.outputPath.c_str()), 24, 54, 16, LIGHTGRAY);
        DrawText(TextFormat("Voxel faces: %i | Voxels: %i", static_cast<int>(result.voxelFaces.size()), result.totalVoxels), 24, 76, 16, LIGHTGRAY);
        DrawText(TextFormat("Normals: %s", smooth ? "Smooth" : "Flat"), 24, 98, 16, LIGHTGRAY);
        DrawText("RMB drag: Orbit | Wheel: Zoom | F/W/G: Toggle | R: Reset", 24, 120, 16, LIGHTGRAY);

        EndDrawing();
    }

    UnloadModel(model);
    UnloadShader(shader);
    CloseWindow();
}

void runVoxelizerGui(
    const vector<string>& objCandidates,
    int initialDepth,
    const string& initialPath,
    bool startSmooth
) {
    if (objCandidates.empty()) return;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1360, 820, "Voxelizer Asick");
    SetTargetFPS(60);

    Font uiFont = GetFontDefault();
    bool customUiFontLoaded = false;
    const char* uiFontCandidates[] = {
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf"
    };
    for (const char* fontPath : uiFontCandidates) {
        if (!FileExists(fontPath)) continue;
        uiFont = LoadFontEx(fontPath, 48, nullptr, 0);
        customUiFontLoaded = true;
        break;
    }
    SetTextureFilter(uiFont.texture, TEXTURE_FILTER_BILINEAR);
    gUiFont = &uiFont;

    vector<string> objList = objCandidates;
    int selectedIndex = 0;
    if (!initialPath.empty()) {
        for (int i = 0; i < static_cast<int>(objList.size()); ++i) {
            if (objList[i] == initialPath) {
                selectedIndex = i;
                break;
            }
        }
    }
    int depth = std::clamp(initialDepth, 0, 10);
    bool smooth = startSmooth;
    bool showWire = false;
    int listScrollOffset = 0;
    const float sidebarWidth = 560.0f;
    bool orbitDragging = false;
    Vector2 lastOrbitMouse{0.0f, 0.0f};

    bool hasOriginal = false;
    bool hasVoxel = false;
    bool previewOriginal = true;
    bool isPreviewLoading = false;
    bool isVoxelLoading = false;
    bool showStatus = false;
    string statusText;
    std::atomic_bool cancelVoxelRequested{false};
    bool cancelUiRequested = false;
    vector<string> createdStatsLines;
    vector<string> prunedStatsLines;
    int voxelDepthUsed = depth;
    OriginalPreviewOutput currentOriginal;
    VoxelizationResult currentVoxel;
    future<OriginalPreviewOutput> previewJob;
    future<BuildOutput> voxelJob;
    double loadingStartedAt = 0.0;

    Mesh mesh = {};
    Model model = {};
    bool hasModel = false;
    Shader shader = createLightingShader();
    int lightPosLoc = GetShaderLocation(shader, "lightPos");
    int viewPosLoc = GetShaderLocation(shader, "viewPos");
    int objectColorLoc = GetShaderLocation(shader, "objectColor");
    const float objectColor[4] = {1.0f, 0.6f, 0.1f, 1.0f};
    if (objectColorLoc >= 0) SetShaderValue(shader, objectColorLoc, objectColor, SHADER_UNIFORM_VEC4);

    auto rebuildModelFrom = [&](const vector<Vertex>& vertices, const vector<Face>& faces) {
        if (hasModel) {
            UnloadModel(model);
            hasModel = false;
        }
        mesh = buildMeshFromFaces(vertices, faces, smooth);
        model = LoadModelFromMesh(mesh);
        model.materials[0].shader = shader;
        model.materials[0].maps[MATERIAL_MAP_DIFFUSE].color = ORANGE;
        if (objectColorLoc >= 0) SetShaderValue(shader, objectColorLoc, objectColor, SHADER_UNIFORM_VEC4);
        hasModel = true;
    };

    float sceneSize = 1.0f;
    Vector3 target = {0.0f, 0.0f, 0.0f};
    Vector3 homePos = {1.5f, 1.2f, 1.5f};
    Vector3 initialOffset = subtract(homePos, target);
    float orbitRadius = 2.0f;
    float orbitYaw = 0.0f;
    float orbitPitch = 0.0f;
    float minRadius = 0.2f;
    float maxRadius = 12.0f;

    Camera3D camera = {};
    camera.position = homePos;
    camera.target = target;
    camera.up = {0.0f, 1.0f, 0.0f};
    camera.fovy = 45.0f;
    camera.projection = CAMERA_PERSPECTIVE;

    auto resetCameraForBox = [&](const MyBoundingBox& box) {
        float sx = box.maxPoint.x - box.minPoint.x;
        float sy = box.maxPoint.y - box.minPoint.y;
        float sz = box.maxPoint.z - box.minPoint.z;
        sceneSize = maxComponent(sx, sy, sz);
        if (sceneSize < 1e-3f) sceneSize = 1.0f;

        target = {
            (box.minPoint.x + box.maxPoint.x) * 0.5f,
            (box.minPoint.y + box.maxPoint.y) * 0.5f,
            (box.minPoint.z + box.maxPoint.z) * 0.5f
        };
        homePos = {target.x + sceneSize * 1.6f, target.y + sceneSize * 1.1f, target.z + sceneSize * 1.6f};
        initialOffset = subtract(homePos, target);
        orbitRadius = sqrtf(initialOffset.x * initialOffset.x + initialOffset.y * initialOffset.y + initialOffset.z * initialOffset.z);
        if (orbitRadius < 1e-3f) orbitRadius = sceneSize * 2.0f;
        orbitYaw = atan2f(initialOffset.z, initialOffset.x);
        orbitPitch = asinf(initialOffset.y / orbitRadius);
        minRadius = sceneSize * 0.15f;
        maxRadius = sceneSize * 12.0f;
        camera.position = homePos;
        camera.target = target;
    };

    auto activateCurrentPreview = [&]() {
        if (previewOriginal && hasOriginal) {
            resetCameraForBox(currentOriginal.box);
            rebuildModelFrom(currentOriginal.vertices, currentOriginal.faces);
        } else if (!previewOriginal && hasVoxel) {
            resetCameraForBox(currentVoxel.rootBox);
            rebuildModelFrom(currentVoxel.voxelVertices, currentVoxel.voxelFaces);
        } else if (hasModel) {
            UnloadModel(model);
            hasModel = false;
        }
    };

    auto startOriginalPreviewLoad = [&](int idx) {
        if (idx < 0 || idx >= static_cast<int>(objList.size())) return;
        string selectedPath = objList[idx];
        isPreviewLoading = true;
        showStatus = true;
        statusText = "Memuat model original...";
        loadingStartedAt = GetTime();
        previewJob = async(launch::async, [selectedPath]() { return loadOriginalPreview(selectedPath); });
    };

    startOriginalPreviewLoad(selectedIndex);

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F11)) ToggleFullscreen();

        if (isPreviewLoading && previewJob.valid()) {
            using namespace std::chrono_literals;
            if (previewJob.wait_for(0ms) == future_status::ready) {
                OriginalPreviewOutput out = previewJob.get();
                isPreviewLoading = false;
                showStatus = true;
                if (out.success) {
                    currentOriginal = std::move(out);
                    hasOriginal = true;
                    previewOriginal = true;
                    statusText = "Preview original siap.";
                    activateCurrentPreview();
                } else {
                    hasOriginal = false;
                    statusText = out.message;
                }
            }
        }

        if (isVoxelLoading && voxelJob.valid()) {
            using namespace std::chrono_literals;
            if (voxelJob.wait_for(0ms) == future_status::ready) {
                BuildOutput out = voxelJob.get();
                isVoxelLoading = false;
                cancelUiRequested = false;
                showStatus = true;
                if (out.success) {
                    currentVoxel = out.result;
                    hasVoxel = true;
                    previewOriginal = false;
                    statusText = TextFormat("Voxel selesai: %d voxel | %.3f dtk", currentVoxel.totalVoxels, currentVoxel.elapsedSeconds);
                    createdStatsLines = buildDepthStatsLines(stats.nodesCreatedPerDepth);
                    prunedStatsLines = buildDepthStatsLines(stats.nodesPrunedPerDepth);
                    activateCurrentPreview();
                } else {
                    hasVoxel = false;
                    statusText = out.message;
                }
                cancelVoxelRequested.store(false);
            }
        }

        const int screenWInput = GetScreenWidth();
        const int screenHInput = GetScreenHeight();
        const Rectangle panelRectInput{8.0f, 8.0f, sidebarWidth, static_cast<float>(screenHInput - 16)};
        const Rectangle viewportRectInput{
            panelRectInput.x + panelRectInput.width + 8.0f,
            8.0f,
            static_cast<float>(screenWInput) - (panelRectInput.x + panelRectInput.width + 16.0f),
            static_cast<float>(screenHInput - 16)
        };

        if (hasModel) {
            const Vector2 mousePos = GetMousePosition();
            const bool pointerOnViewport = CheckCollisionPointRec(mousePos, viewportRectInput);
            if (pointerOnViewport && (IsMouseButtonPressed(MOUSE_BUTTON_LEFT) || IsMouseButtonPressed(MOUSE_BUTTON_RIGHT))) {
                orbitDragging = true;
                lastOrbitMouse = mousePos;
            }
            if (!IsMouseButtonDown(MOUSE_BUTTON_LEFT) && !IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
                orbitDragging = false;
            }
            if (orbitDragging) {
                Vector2 delta{mousePos.x - lastOrbitMouse.x, mousePos.y - lastOrbitMouse.y};
                orbitYaw -= delta.x * 0.0060f;
                orbitPitch += delta.y * 0.0048f;
                lastOrbitMouse = mousePos;
            }

            if (IsKeyDown(KEY_LEFT)) orbitYaw += 0.028f;
            if (IsKeyDown(KEY_RIGHT)) orbitYaw -= 0.028f;
            if (IsKeyDown(KEY_UP)) orbitPitch -= 0.022f;
            if (IsKeyDown(KEY_DOWN)) orbitPitch += 0.022f;

            float wheel = GetMouseWheelMove();
            if (pointerOnViewport && wheel != 0.0f) orbitRadius *= (1.0f - wheel * 0.12f);
            if (IsKeyDown(KEY_Q)) orbitRadius *= 1.015f;
            if (IsKeyDown(KEY_E)) orbitRadius *= 0.985f;

            orbitPitch = std::clamp(orbitPitch, -1.35f, 1.35f);
            orbitRadius = std::clamp(orbitRadius, minRadius, maxRadius);
            camera.position = {
                target.x + orbitRadius * cosf(orbitPitch) * cosf(orbitYaw),
                target.y + orbitRadius * sinf(orbitPitch),
                target.z + orbitRadius * cosf(orbitPitch) * sinf(orbitYaw)
            };
            camera.target = target;
        }

        if (IsKeyPressed(KEY_R) && hasModel) {
            orbitRadius = sqrtf(initialOffset.x * initialOffset.x + initialOffset.y * initialOffset.y + initialOffset.z * initialOffset.z);
            if (orbitRadius < 1e-3f) orbitRadius = sceneSize * 2.0f;
            orbitYaw = atan2f(initialOffset.z, initialOffset.x);
            orbitPitch = asinf(initialOffset.y / orbitRadius);
            camera.position = homePos;
            camera.target = target;
            orbitDragging = false;
        }
        if (IsKeyPressed(KEY_W)) showWire = !showWire;
        if (IsKeyPressed(KEY_F) && hasModel && !isVoxelLoading && !isPreviewLoading) {
            smooth = !smooth;
            activateCurrentPreview();
        }

        const int screenW = GetScreenWidth();
        const int screenH = GetScreenHeight();
        const Rectangle panelRect{8.0f, 8.0f, sidebarWidth, static_cast<float>(screenH - 16)};
        const Rectangle viewportRect{
            panelRect.x + panelRect.width + 8.0f,
            8.0f,
            static_cast<float>(screenW) - (panelRect.x + panelRect.width + 16.0f),
            static_cast<float>(screenH - 16)
        };

        BeginDrawing();
        ClearBackground(Color{205, 210, 217, 255});
        DrawRectangleRec(panelRect, Color{233, 235, 238, 255});
        DrawRectangleLinesEx(panelRect, 1.0f, Color{168, 172, 180, 255});
        DrawRectangleRec(viewportRect, Color{49, 55, 68, 255});
        DrawRectangleLinesEx(viewportRect, 1.0f, Color{87, 94, 108, 255});

        if (hasModel) {
            Vector3 lightPos = {target.x + sceneSize * 2.0f, target.y + sceneSize * 2.0f, target.z + sceneSize * 1.3f};
            SetShaderValue(shader, lightPosLoc, &lightPos.x, SHADER_UNIFORM_VEC3);
            SetShaderValue(shader, viewPosLoc, &camera.position.x, SHADER_UNIFORM_VEC3);
            BeginMode3D(camera);
            Vector3 modelPosition{0.0f, 0.0f, 0.0f};
            DrawModel(model, modelPosition, 1.0f, WHITE);
            if (showWire) {
                rlDisableBackfaceCulling();
                DrawModelWires(model, modelPosition, 1.0f, BLACK);
                rlEnableBackfaceCulling();
            }
            const MyBoundingBox& box = previewOriginal ? currentOriginal.box : currentVoxel.rootBox;
            BoundingBox rootBounds{
                {box.minPoint.x, box.minPoint.y, box.minPoint.z},
                {box.maxPoint.x, box.maxPoint.y, box.maxPoint.z}
            };
            DrawBoundingBox(rootBounds, GRAY);
            EndMode3D();
        } else {
            drawTextWithCurrentFont("Belum ada model aktif.", viewportRect.x + 22, viewportRect.y + 22, 38, Color{200, 204, 214, 255});
            drawTextWithCurrentFont("Pilih file OBJ di list kiri untuk langsung memuat.", viewportRect.x + 22, viewportRect.y + 70, 27, Color{170, 174, 184, 255});
        }

        int px = static_cast<int>(panelRect.x) + 10;
        int py = static_cast<int>(panelRect.y) + 10;
        drawTextWithCurrentFont("Voxelizer ASICK", static_cast<float>(px), static_cast<float>(py), 34, Color{32, 34, 41, 255});
        py += 46;

        if (drawButton(Rectangle{static_cast<float>(px), static_cast<float>(py), panelRect.width - 20.0f, 48.0f}, "Refresh List", !isPreviewLoading && !isVoxelLoading)) {
            auto refreshed = collectObjCandidates(initialPath);
            if (!refreshed.empty()) {
                string prevPath = (selectedIndex >= 0 && selectedIndex < static_cast<int>(objList.size())) ? objList[selectedIndex] : "";
                objList = std::move(refreshed);
                selectedIndex = 0;
                for (int i = 0; i < static_cast<int>(objList.size()); ++i) {
                    if (objList[i] == prevPath) {
                        selectedIndex = i;
                        break;
                    }
                }
                listScrollOffset = std::max(0, std::min(listScrollOffset, std::max(0, static_cast<int>(objList.size()) - 1)));
                showStatus = true;
                statusText = "List file diperbarui.";
            }
        }
        py += 58;

        if (drawButton(Rectangle{static_cast<float>(px), static_cast<float>(py), panelRect.width - 20.0f, 48.0f},
                smooth ? "Mode Tampilan: Smooth" : "Mode Tampilan: Flat",
                !isPreviewLoading && !isVoxelLoading && hasModel)) {
            smooth = !smooth;
            activateCurrentPreview();
        }
        py += 62;

        const int depthPanelHeight = isVoxelLoading ? 182 : 136;
        DrawRectangle(px, py, static_cast<int>(panelRect.width - 20.0f), depthPanelHeight, Color{226, 229, 234, 255});
        DrawRectangleLines(px, py, static_cast<int>(panelRect.width - 20.0f), depthPanelHeight, Color{190, 194, 202, 255});
        drawTextWithCurrentFont("Pengaturan Depth", static_cast<float>(px + 10), static_cast<float>(py + 10), 26, Color{47, 50, 56, 255});
        drawTextWithCurrentFont(TextFormat("Depth saat ini: %d", depth), static_cast<float>(px + 10), static_cast<float>(py + 46), 23, Color{47, 50, 56, 255});
        if (drawButton(Rectangle{static_cast<float>(px + 10), static_cast<float>(py + 78), 48.0f, 38.0f}, "-", !isVoxelLoading && depth > 0)) depth--;
        if (drawButton(Rectangle{static_cast<float>(px + 64), static_cast<float>(py + 78), 48.0f, 38.0f}, "+", !isVoxelLoading && depth < 10)) depth++;
        if (drawButton(
                Rectangle{
                    static_cast<float>(px + 118),
                    static_cast<float>(py + 78),
                    panelRect.width - 20.0f - 118.0f - 10.0f,
                    38.0f
                },
                "Terapkan Depth",
                !isPreviewLoading && !isVoxelLoading && hasOriginal
            )) {
            isVoxelLoading = true;
            showStatus = false;
            cancelUiRequested = false;
            cancelVoxelRequested.store(false);
            voxelDepthUsed = depth;
            loadingStartedAt = GetTime();
            vector<Vertex> inputVertices = currentOriginal.vertices;
            vector<Face> inputFaces = currentOriginal.faces;
            string inputPath = currentOriginal.inputPath;
            voxelJob = async(launch::async, [inputPath, inputVertices, inputFaces, depth, &cancelVoxelRequested]() {
                return processVoxelization(inputPath, inputVertices, inputFaces, depth, &cancelVoxelRequested);
            });
        }
        if (isVoxelLoading) {
            if (drawButton(
                    Rectangle{
                        static_cast<float>(px + 10),
                        static_cast<float>(py + 126),
                        panelRect.width - 40.0f,
                        42.0f
                    },
                    cancelUiRequested ? "Membatalkan..." : "Cancel Voxel",
                    !cancelUiRequested
                )) {
                cancelUiRequested = true;
                cancelVoxelRequested.store(true);
                showStatus = true;
                statusText = "Permintaan cancel dikirim, menunggu worker berhenti...";
            }
        }
        py += depthPanelHeight + 12;

        drawTextWithCurrentFont("Pilih File OBJ:", static_cast<float>(px), static_cast<float>(py), 28, Color{47, 50, 56, 255});
        py += 40;
        const float listFontSize = 20.0f;
        const float listScaledFont = listFontSize * gUiScale;
        const int rowHeight = std::max(44, static_cast<int>(listScaledFont + 14.0f));
        const int visibleCount = 7;
        const int listHeight = rowHeight * visibleCount + 8;
        const Rectangle listRect{static_cast<float>(px), static_cast<float>(py), panelRect.width - 20.0f, static_cast<float>(listHeight)};
        DrawRectangleRec(listRect, Color{215, 218, 223, 255});
        DrawRectangleLinesEx(listRect, 1.0f, Color{181, 185, 194, 255});
        if (CheckCollisionPointRec(GetMousePosition(), listRect)) {
            float wheel = GetMouseWheelMove();
            if (wheel > 0.0f) listScrollOffset--;
            if (wheel < 0.0f) listScrollOffset++;
        }
        listScrollOffset = std::max(0, std::min(listScrollOffset, std::max(0, static_cast<int>(objList.size()) - visibleCount)));
        int maxRows = std::min(visibleCount, static_cast<int>(objList.size()) - listScrollOffset);
        for (int i = 0; i < maxRows; ++i) {
            int idx = listScrollOffset + i;
            Rectangle row{
                listRect.x + 2.0f,
                listRect.y + 2.0f + static_cast<float>(i * rowHeight),
                listRect.width - 4.0f,
                static_cast<float>(rowHeight - 2)
            };
            bool selected = (idx == selectedIndex);
            bool hover = CheckCollisionPointRec(GetMousePosition(), row);
            if (selected) DrawRectangleRec(row, Color{176, 199, 238, 255});
            else if (hover) DrawRectangleRec(row, Color{195, 203, 217, 255});
            drawTextWithCurrentFont(
                baseName(objList[idx]),
                row.x + 10,
                row.y + (static_cast<float>(rowHeight) - listScaledFont) * 0.5f,
                listFontSize,
                Color{38, 41, 47, 255}
            );
            if (hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT) && !isPreviewLoading && !isVoxelLoading) {
                selectedIndex = idx;
                hasVoxel = false;
                createdStatsLines.clear();
                prunedStatsLines.clear();
                if (!currentVoxel.outputPath.empty()) {
                    currentVoxel = VoxelizationResult{};
                }
                previewOriginal = true;
                startOriginalPreviewLoad(selectedIndex);
            }
        }
        py += listHeight + 12;

        float halfButtonW = (panelRect.width - 20.0f - 8.0f) * 0.5f;
        if (drawButton(Rectangle{static_cast<float>(px), static_cast<float>(py), halfButtonW, 48.0f}, "Tampil Original", !isPreviewLoading && !isVoxelLoading && hasOriginal)) {
            previewOriginal = true;
            activateCurrentPreview();
        }
        if (drawButton(Rectangle{static_cast<float>(px + halfButtonW + 8.0f), static_cast<float>(py), halfButtonW, 48.0f}, "Tampil Voxel", !isPreviewLoading && !isVoxelLoading)) {
            if (hasVoxel) {
                previewOriginal = false;
                activateCurrentPreview();
            } else {
                showStatus = true;
                statusText = "Belum ada hasil voxel. Klik Terapkan Depth terlebih dulu.";
            }
        }
        py += 52;

        const int infoFont = 22;
        drawTextWithCurrentFont(TextFormat("Depth Octree: %d", hasVoxel ? voxelDepthUsed : depth), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
        py += 24;
        drawTextWithCurrentFont(TextFormat("Model aktif: %s", hasModel ? (previewOriginal ? "Original" : "Voxel") : "-"), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
        py += 24;

        const float textBlockWidth = panelRect.width - 28.0f;
        if (isPreviewLoading || isVoxelLoading) {
            int phase = static_cast<int>(GetTime() * 4.0) % 4;
            string dots(static_cast<size_t>(phase), '.');
            const char* loadingLabel = (isVoxelLoading && cancelUiRequested) ? "Cancelling" : "Loading";
            drawTextWithCurrentFont(TextFormat("Status: %s%s", loadingLabel, dots.c_str()), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{164, 111, 24, 255});
            drawTextWithCurrentFont(TextFormat("Elapsed: %.1f s", GetTime() - loadingStartedAt), static_cast<float>(px + 190), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
        } else if (showStatus) {
            py += static_cast<int>(drawWrappedTextWithCurrentFont(
                TextFormat("Status: %s", statusText.c_str()),
                static_cast<float>(px),
                static_cast<float>(py),
                textBlockWidth,
                static_cast<float>(infoFont),
                0.12f,
                Color{47, 50, 56, 255}
            ));
        }
        py += 26;

        if (hasVoxel) {
            drawTextWithCurrentFont(TextFormat("Banyak voxel: %d", currentVoxel.totalVoxels), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
            py += 22;
            drawTextWithCurrentFont(TextFormat("Banyak vertex: %d", static_cast<int>(currentVoxel.voxelVertices.size())), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
            py += 22;
            drawTextWithCurrentFont(TextFormat("Banyak faces: %d", static_cast<int>(currentVoxel.voxelFaces.size())), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
            py += 22;
            drawTextWithCurrentFont(TextFormat("Waktu proses: %.3f detik", currentVoxel.elapsedSeconds), static_cast<float>(px), static_cast<float>(py), static_cast<float>(infoFont), Color{47, 50, 56, 255});
            py += 22;
            py += static_cast<int>(drawWrappedTextWithCurrentFont(
                TextFormat("Path output: %s", currentVoxel.outputPath.c_str()),
                static_cast<float>(px),
                static_cast<float>(py),
                textBlockWidth,
                19.0f,
                0.12f,
                Color{47, 50, 56, 255}
            ));
            py += 8;
            py += static_cast<int>(drawWrappedTextWithCurrentFont(
                TextFormat("Node terbentuk: %s", compactStats(createdStatsLines, 5).c_str()),
                static_cast<float>(px),
                static_cast<float>(py),
                textBlockWidth,
                19.0f,
                0.12f,
                Color{47, 50, 56, 255}
            ));
            py += 6;
            py += static_cast<int>(drawWrappedTextWithCurrentFont(
                TextFormat("Node tidak ditelusuri: %s", compactStats(prunedStatsLines, 5).c_str()),
                static_cast<float>(px),
                static_cast<float>(py),
                textBlockWidth,
                19.0f,
                0.12f,
                Color{47, 50, 56, 255}
            ));
            py += 6;
        }

        drawTextWithCurrentFont("Hint: orbit drag LMB/RMB, zoom scroll/Q/E", static_cast<float>(px), static_cast<float>(py), 19, Color{78, 82, 90, 255});
        py += 18;
        drawTextWithCurrentFont("Arrow: orbit | F11: fullscreen | R: reset", static_cast<float>(px), static_cast<float>(py), 19, Color{78, 82, 90, 255});

        if (hasVoxel) {
            DrawText(
                TextFormat("Voxel faces: %i | Voxels: %i", static_cast<int>(currentVoxel.voxelFaces.size()), currentVoxel.totalVoxels),
                static_cast<int>(viewportRect.x + 18),
                static_cast<int>(viewportRect.y + viewportRect.height - 30),
                20,
                Color{196, 201, 210, 255}
            );
        }

        EndDrawing();
    }

    if (isPreviewLoading && previewJob.valid()) previewJob.wait();
    if (isVoxelLoading && voxelJob.valid()) {
        cancelVoxelRequested.store(true);
        voxelJob.wait();
    }
    if (hasModel) UnloadModel(model);
    gUiFont = nullptr;
    if (customUiFontLoaded) UnloadFont(uiFont);
    UnloadShader(shader);
    CloseWindow();
}

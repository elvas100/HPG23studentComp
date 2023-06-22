#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define M_PI 3.14159265358979323846


constexpr float deg2rad(float degrees) {
    return degrees * (M_PI / 180.0f);
}

struct Vec3 {
    float x, y, z;

    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    Vec3 operator*(float scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }



    Vec3 operator/(float scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vec3& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vec3& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Vec3 normalize() const {
        float len = length();
        return Vec3(x / len, y / len, z / len);
    }

    static float dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    static Vec3 randomUnitHemisphere(const Vec3& normal) {
        /*todo*/
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0, 1);

        float u = dis(gen);
        float v = dis(gen);

        float theta = 2.0f * M_PI * u;
        float phi = std::acos(2.0f * v - 1.0f);

        float x = std::cos(theta) * std::sin(phi);
        float y = std::sin(theta) * std::sin(phi);
        float z = std::cos(phi);

        Vec3 tangent = (std::abs(normal.x) < 0.9f) ? Vec3(1, 0, 0) : Vec3(0, 1, 0);
        Vec3 bitangent = Vec3::cross(normal, tangent).normalize();
        tangent = Vec3::cross(bitangent, normal).normalize();

        return (tangent * x + bitangent * y + normal * z).normalize();
    }
};

struct Triangle {
    Vec3 v0, v1, v2;

    Triangle(const Vec3& v0, const Vec3& v1, const Vec3& v2)
        : v0(v0), v1(v1), v2(v2){}
};


struct Light {
    Vec3 position;
    Vec3 intensity;

    Light(const Vec3& position, const Vec3& intensity) : position(position), intensity(intensity) {}
};

struct Face {
    int v1, v2, v3;
    int vn1, vn2, vn3;
    int materialIndex; // Added material index for each face
};


struct Material {
    std::string name;
    Vec3 diffuseColor;
    Material(const Vec3& diffuseColor = Vec3(0, 0, 0)) : diffuseColor(diffuseColor) {}
};


struct Mesh {
    std::vector<Vec3> vertices;
    std::vector<Vec3> normals;
    std::vector<Face> faces;
    std::vector<Material> materials;
    std::vector<Light> lights;
};




int maxBounces = 2;


void setMaterial(Mesh& mesh, const std::string& materialName, int& currentMaterialIndex) {
    for (size_t i = 0; i < mesh.materials.size(); ++i) {
        if (mesh.materials[i].name == materialName) {
            currentMaterialIndex = i;
            break;
        }
    }
}

void loadObjFile(const std::string& objFilename, const std::string& mtlFilename, Mesh& mesh) {
    std::ifstream objFile(objFilename);
    if (!objFile.is_open()) {
        std::cerr << "Failed to open OBJ file: " << objFilename << std::endl;
        return;
    }

    std::ifstream mtlFile(mtlFilename);
    if (!mtlFile.is_open()) {
        std::cerr << "Failed to open MTL file: " << mtlFilename << std::endl;
        return;
    }

    std::string line;
    Material material;
    while (std::getline(mtlFile, line)) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "newmtl") {
            iss >> material.name;
        }
        else if (keyword == "Kd") {
            float r, g, b;
            iss >> r >> g >> b;
            material.diffuseColor = Vec3(r, g, b);
            mesh.materials.push_back(material);
        }


    }

    mtlFile.close();

    int currentMaterialIndex = -1;
    while (std::getline(objFile, line)) {
        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            mesh.vertices.emplace_back(x, y, z);
        }
        else if (keyword == "vn") {
            float x, y, z;
            iss >> x >> y >> z;
            mesh.normals.emplace_back(x, y, z);
        }

        else if (keyword == "usemtl") {
            std::string materialName;
            iss >> materialName;
            setMaterial(mesh, materialName, currentMaterialIndex); // Set the current material
        }

        else if (keyword == "f") {
            std::string v0Str, v1Str, v2Str;
            iss >> v0Str >> v1Str >> v2Str;

            int v0, v1, v2, n0, n1, n2;
            sscanf(v0Str.c_str(), "%d//%d", &v0, &n0);
            sscanf(v1Str.c_str(), "%d//%d", &v1, &n1);
            sscanf(v2Str.c_str(), "%d//%d", &v2, &n2);

            mesh.faces.emplace_back(v0 - 1, v1 - 1, v2 - 1, n0 - 1, n1 - 1, n2 - 1, currentMaterialIndex);

        }
    }

    objFile.close();
}


bool rayTriangleIntersect(const Vec3& rayOrigin, const Vec3& rayDirection, const Triangle& triangle, Vec3& intersectionPoint, Vec3& intersectionNormal) {
   /*todo*/

    const float epsilon = 0.00001f;
    const Vec3& v0 = triangle.v0;
    const Vec3& v1 = triangle.v1;
    const Vec3& v2 = triangle.v2;

    Vec3 edge1 = v1 - v0;
    Vec3 edge2 = v2 - v0;
    Vec3 triangleNormal = Vec3::cross(edge1, edge2).normalize();
    Vec3 h = Vec3::cross(rayDirection, edge2);
    float a = Vec3::dot(edge1, h);

    if (a > -epsilon && a < epsilon)
        return false;

    float f = 1.0f / a;
    Vec3 s = rayOrigin - v0;
    float u = f * Vec3::dot(s, h);

    if (u < 0.0f || u > 1.0f)
        return false;

    Vec3 q = Vec3::cross(s, edge1);
    float v = f * Vec3::dot(rayDirection, q);

    if (v < 0.0f || u + v > 1.0f)
        return false;

    float t = f * Vec3::dot(edge2, q);
    if (t > epsilon) {
        intersectionPoint = rayOrigin + rayDirection * t;
        Vec3 vertexNormal = (v0 * (1.0f - u - v) + v1 * u + v2 * v).normalize();
        intersectionNormal = (triangleNormal * (1.0f - u - v) + vertexNormal * u + vertexNormal * v).normalize();;
        return true;
    }

    return false;
}


Vec3 traceRay(const Mesh& mesh, const Vec3& rayOrigin, const Vec3& rayDirection, int bounce) {

    if (bounce > maxBounces)
        return Vec3(0, 0, 0);

    Vec3 hitP;
    Vec3 hitNorm;
    float tmin = std::numeric_limits<float>::max();
    int closestTriIndex = -1;


    // Find the closest triangle intersection
    for (int i = 0; i < mesh.faces.size(); ++i) {
        Vec3 point, normal;
        Triangle tri(mesh.vertices[mesh.faces[i].v1], mesh.vertices[mesh.faces[i].v2], mesh.vertices[mesh.faces[i].v3]);
        if (rayTriangleIntersect(rayOrigin, rayDirection, tri, point, normal)) {
            float distance = (point - rayOrigin).length();
            if (distance < tmin) {
                tmin = distance;
                closestTriIndex = i;
                hitP = point;
                hitNorm = normal;
            }
        }
    }


    if (closestTriIndex != -1) {
        const Face& face = mesh.faces[closestTriIndex];
        const Material& material = mesh.materials[face.materialIndex];


        Vec3 color(0, 0, 0);

        // Compute shading
        for (const Light& light : mesh.lights) {

            Vec3 L = (hitP - light.position);
           // Vec3 L = (light.position - hitP);

            float lightDistance = L.length();
            L /= lightDistance;
            bool isShadowed = false;

                // Check for shadow ray intersection
                for (int i = 0; i < mesh.faces.size(); ++i) {
                    if (i != closestTriIndex) {
                        Vec3 point, normal;
                        Triangle tri(mesh.vertices[mesh.faces[i].v1], mesh.vertices[mesh.faces[i].v2], mesh.vertices[mesh.faces[i].v3]);
                        if (rayTriangleIntersect(hitP, L, tri, point, normal)) {
                            isShadowed = true;
                            break;
                        }
                    }
                }

                if (!isShadowed) {
                    Vec3 diffuse = material.diffuseColor;
                    float intensity = light.intensity.length() / lightDistance;
                    Vec3 shading = diffuse * intensity;// *Vec3::dot(hitNorm, L);
                    color += shading;
                }

        }

        // Trace secondary diffuse ray
        if (bounce < maxBounces) {

            Vec3 diffuseDirection = Vec3::randomUnitHemisphere(hitNorm);
            Vec3 diffuseColor = traceRay(mesh, hitP, diffuseDirection, bounce + 1);
            color += Vec3(diffuseColor.x * material.diffuseColor.x, diffuseColor.y * material.diffuseColor.y,
                diffuseColor.z * material.diffuseColor.z);

        }
        return color;
    }

    return Vec3(0, 0, 0);

}

void renderImage(const Mesh& mesh) {

    int width = 400;
    int height = 400;
    std::vector<Vec3> imageBuffer(width * height);

    int numSamples = 16;

//1    Vec3 cameraPosition(-2, 0,0);
    Vec3 cameraPosition(0, 0, 17.7);

    Vec3 lookAtDirection(0.0, 0.0, 0.0);
    Vec3 cameraUp(0, 1, 0);
    float fov = 10;

    Vec3 lookDir = (lookAtDirection - cameraPosition).normalize();

    Vec3 u = Vec3::cross(lookDir, cameraUp).normalize();
    Vec3 v = Vec3::cross(u, lookDir).normalize();

    // Iterate over each pixel
    for (int pix = 0; pix < width * height; pix++) {

        int i = pix / width; //x
        int j = pix % height;//y


        Vec3 color;
        float x = 2.0f * (j - width / 2.f + 0.5f) / width;
        float y = 2.0f * (i - height / 2.f + 0.5f) / height;

        Vec3 direction = (lookDir + u * x + v * y).normalize();

        for (int spp =0; spp< numSamples; spp++){
            color += traceRay(mesh, cameraPosition, direction, 0);
        }
        color /= numSamples;
        imageBuffer[(height- j - 1) * width + (width - i -1)] = color;

    }

    // Convert image buffer to unsigned char array
    std::vector<unsigned char> imageData(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        imageData[i * 3] = static_cast<unsigned char>(std::clamp(imageBuffer[i].x * 255.0f, 0.0f, 255.0f));
        imageData[i * 3 + 1] = static_cast<unsigned char>(std::clamp(imageBuffer[i].y * 255.0f, 0.0f, 255.0f));
        imageData[i * 3 + 2] = static_cast<unsigned char>(std::clamp(imageBuffer[i].z * 255.0f, 0.0f, 255.0f));
    }

    // Save the image as PNG
    stbi_write_png("output.png", width, height, 3, imageData.data(), width * 3);
}

int main() {

    std::string objFile = "final.obj";
    std::string mtlFile = "final.mtl";


    Mesh scene;

    loadObjFile(objFile, mtlFile, scene);

    Light light(Vec3(8.8, 0, 2.5), Vec3(2.5, 2.5, 2.5));
    scene.lights.push_back(light);

    // Render the image
    renderImage(scene);

    return 0;
}

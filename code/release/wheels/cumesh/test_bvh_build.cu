// cuBVH's tree build, several threads against one, on random meshes: identical nodes and triangle order?
#include <gpu/bvh.cuh>
#include <chrono>
#include <cstdlib>
#include <random>
#include <vector>

using namespace cumesh::cubvh;

static std::vector<Triangle> random_mesh(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<Triangle> tris(n);
    for (size_t i = 0; i < n; ++i) {
        Eigen::Vector3f a(u(rng), u(rng), u(rng));
        Eigen::Vector3f b = a + Eigen::Vector3f(u(rng), u(rng), u(rng)) * 0.01f;
        Eigen::Vector3f c = a + Eigen::Vector3f(u(rng), u(rng), u(rng)) * 0.01f;
        tris[i] = {a, b, c, (int64_t)i};
    }
    return tris;
}

int main() {
    const size_t sizes[] = {4, 7, 8, 9, 16, 17, 33, 100, 257, 1000, 12345, 70000, 262145, 1u << 20, 2000003};
    _putenv_s("CUBVH_BUILD_CHECK", "1");
    for (size_t n : sizes) {
        auto tris = random_mesh(n, (unsigned)n);
        auto bvh = TriangleBvh::make();
        printf("n=%zu: ", n);
        bvh->build(tris, 8);
    }
    _putenv_s("CUBVH_BUILD_CHECK", "0");
    const size_t big = 4000000;
    for (const char* threads : {"1", "4", "16", "0"}) {
        if (threads[0] == '0') _putenv_s("CUBVH_BUILD_THREADS", ""); else _putenv_s("CUBVH_BUILD_THREADS", threads);
        auto tris = random_mesh(big, 7);
        auto bvh = TriangleBvh::make();
        auto t0 = std::chrono::steady_clock::now();
        bvh->build(tris, 8);
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%zu triangles, CUBVH_BUILD_THREADS=%s: %.2f s\n", big, threads[0] == '0' ? "auto" : threads, s);
    }
    return 0;
}

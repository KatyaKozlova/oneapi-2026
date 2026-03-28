#include <iostream>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float result = 0.0f;
    const float step = (end - start) / count;

    sycl::queue queue(device);

    {
        sycl::buffer<float> resultBuffer(&result, 1);

        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(resultBuffer, cgh, sycl::plus<>());

            cgh.parallel_for(
                sycl::range<2>(count, count),
                reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    float x = start + step * (idx.get(0) + 0.5f);
                    float y = start + step * (idx.get(1) + 0.5f);
                    sum += sycl::sin(x) * sycl::cos(y);
                }
            );
        }).wait();
    }

    return result * step * step;
}

int main() {
    try {
        // Integration parameters
        float start = 0.0f;
        float end = M_PI;
        int count = 256;

        // Select device (CPU for testing)
        sycl::device device = sycl::device(sycl::cpu_selector_v);
        sycl::queue q(device);

        std::cout << "Using device: " 
                  << device.get_info<sycl::info::device::name>() << std::endl;
        std::cout << "Parameters: start=" << start << ", end=" << end 
                  << ", count=" << count << std::endl;

        // Measure performance
        auto start_time = std::chrono::high_resolution_clock::now();
        float result = IntegralONEAPI(start, end, count, device);
        auto end_time = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end_time - start_time;

        // Output results
        std::cout << "\nResults:" << std::endl;
        std::cout << "Integral approximation: " << result << std::endl;
        std::cout << "Execution time: " << duration.count() << " sec" << std::endl;

    } catch (const sycl::exception& e) {
        std::cerr << "SYCL error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
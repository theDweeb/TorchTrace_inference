#include <torch/torch.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <bits/stdc++.h> 

// Benchmarking
void benchmark(const char* model_name, int img_size, int batch_size, int iterations) {
    torch::NoGradGuard no_grad;
    std::cout << "-> Loading model: " << model_name << "...\n";
    std::cout << "-> Batch size: " << batch_size << std::endl;
    std::cout << "-> Image size: " << img_size << std::endl;
    
    torch::jit::script::Module module = torch::jit::load(model_name);
    module.to(torch::kCUDA);

    std::vector<torch::jit::IValue> batch;
    batch.push_back(torch::randn({batch_size, 3, img_size, img_size}).set_requires_grad(false).cuda());
    float total_time = 0.0;

    system("python3 nc.py START");

    for (int i=0; i<iterations; i++) {        
        cudaDeviceSynchronize();
        std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();        
        
        auto output = module.forward(batch);
        //auto output_tensor = output.toTensor();
        cudaDeviceSynchronize();

        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

        total_time += (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0);
        //std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << "ms\n";
    }
    //system("python3 nc.py stop")
    system("python3 nc.py STOP");
    float avg_time = (total_time / (iterations * batch_size));
    float avg_fps = 1 / (avg_time / 1000);
    std::cout << "\nTotal time: " << total_time / 1000 << "s\n";
    std::cout << "Average time (per image): " << avg_time << "ms\n";
    std::cout << "Average FPS: " << avg_fps << " FPS\n\n\n";
}

int main(int argc, char *argv[]) {
    int imgSize = std::stoi(argv[2]);
    int batchSize = std::stoi(argv[3]);
    int iterations = 10000;
    // If the number of iterations is explicitly set i.e. argument 4
    if(argc == 5){
        iterations = std::stoi(argv[4]);
        std::cout << "-> custom number of iterations: " << iterations << std::endl;
        
    }
    
    benchmark(argv[1], imgSize, batchSize, iterations);
}   

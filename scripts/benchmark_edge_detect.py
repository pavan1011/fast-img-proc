import fast_image_processing as fip
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import fmean
import os
from os import listdir
from os.path import isfile, join
import sys

# If running only one trial per image, populate the directory with this script
# with a sample image called 'trial.jpg'. It can be any random image. 
# This helps getting consistent results for all images as the first run
# always takes much longer than normal on GPU since the caches/register are cold.
TRIALS_PER_IMAGE = 1

def benchmark_edge_detection(image_list, kernel_sizes, output_dir):
    results = []

    if TRIALS_PER_IMAGE == 1 and len(image_list)>0:
        # If running only one trial per image
        #Load a random image to wake up GPU cache
        # Otherwise first image will always run slower on GPU
        # run edge detect with last image
        trial_img = fip.Image("trial.jpg")
        _ = fip.edge_detect(trial_img, 1, 1, 7, fip.Hardware.GPU)

        
    for img_file in image_list:
        # Create a random image of specified size
        input_image = fip.Image(img_file)
        width = input_image.width
        height = input_image.height
        
        for kernel_size in kernel_sizes:
            print(f"Testing image size: {width}x{height}, kernel size: {kernel_size}")
            
            cpu_time_list = []
            gpu_time_list = []

            for trial in range(TRIALS_PER_IMAGE):
                # CPU Benchmark
                start_time = time.time()
                cpu_result = fip.edge_detect(input_image, 1, 1, kernel_size, fip.Hardware.CPU)
                cpu_time = time.time() - start_time
                cpu_time_list.append(cpu_time)

                # GPU Benchmark
                start_time = time.time()
                gpu_result = fip.edge_detect(input_image, 1, 1, kernel_size, fip.Hardware.GPU)
                gpu_time = time.time() - start_time
                gpu_time_list.append(gpu_time)

            
                results.append({
                    "height": height,
                    "width": width,
                    "image_size": width*height,
                    "kernel_size": kernel_size,
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time/gpu_time
                })

            print(f"took CPU (best): {round(min(cpu_time_list), 6)} s, GPU (best): {round(min(gpu_time_list), 6)} s")
            print(f"took CPU (avg): {round(fmean(cpu_time_list), 6)} s, GPU (avg): {round(fmean(gpu_time_list), 6)} s")

            out_img_file, out_img_ext= os.path.splitext(os.path.basename(img_file))

            print(f"Saving {output_dir}/cpu_output_{out_img_file}_{kernel_size}{out_img_ext}")
            cpu_result.save(f"{output_dir}/cpu_output_{out_img_file}_{kernel_size}{out_img_ext}")

            print(f"Saving {output_dir}/cpu_output_{out_img_file}_{kernel_size}{out_img_ext}")
            gpu_result.save(f"{output_dir}/gpu_output_{out_img_file}_{kernel_size}{out_img_ext}")
    
    return pd.DataFrame(results)

def plot_results(df):
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot execution times
    for k in df['kernel_size'].unique():
        mask = df['kernel_size'] == k
        ax1.plot(df[mask]['image_size'], df[mask]['avg_cpu'], 
                marker='o', label=f'CPU (k={k})')
        ax1.plot(df[mask]['image_size'], df[mask]['avg_gpu'], 
                marker='s', label=f'GPU (k={k})')
    
    ax1.set_xlabel('Image Size (pixels)')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('CPU vs GPU Execution Time')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot speedup
    for k in df['kernel_size'].unique():
        mask = df['kernel_size'] == k
        ax2.plot(df[mask]['image_size'], df[mask]['speedup'], 
                marker='o', label=f'k={k}')
    
    ax2.set_xlabel('Image Size (pixels)')
    ax2.set_ylabel('Speedup (CPU Time / GPU Time)')
    ax2.set_title('GPU Speedup Factor')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(f'edge_detection_benchmark.png')
    plt.close()

def main():
    # Test parameters
    if len(sys.argv) != 3:
        print("Pass 2 command line argument: <path to input images dir> <path to output>")
    
    test_image_path = sys.argv[1]
    output_dir = os.path.abspath(sys.argv[2])
    # Check if output dir exists, create if not
    if not os.path.exists(output_dir):
        print(f"Output directory: {output_dir} doesn't exist. Creating it.")
        os.makedirs(output_dir)
    print(f"Reading images from {test_image_path}")  
    image_list = []
    for f in listdir(test_image_path):
        if(isfile(join(test_image_path, f)) and \
            (f.endswith('.png') or f.endswith('.jpg'))):
            print(f"Found: {f}")
            image_list.append(join(test_image_path, f))
    
    kernel_sizes = [3, 5, 7]
    print(image_list)
    
    if len(image_list) <= 0:
        print("No images found in the input dir")
        return
    
    # Run benchmarks
    results = benchmark_edge_detection(image_list, kernel_sizes, output_dir)
    
    # Save results to CSV
    results.to_csv('benchmark_results.csv', index=False)
    
    # Plot results
    # TODO: Fix plotting
    # plot_results(results)
    
    # Print summary statistics
    print("\nBenchmark Summary:")

    print("\nMean Speedup by Kernel Size:")
    print(results.groupby('kernel_size')['speedup'].mean())

    print("\nMean Speedup by Image Size:")
    print(results.groupby('image_size')['speedup'].mean())

    print("\nMean Speedup by Image Size and Kernel Size:")
    print(results.groupby(['image_size', 'kernel_size'])['speedup'].mean())


    print("\nBest Speedup by Kernel Size:")
    print(results.groupby('kernel_size')['speedup'].max())

    print("\nBest Speedup by Image Size:")
    print(results.groupby('image_size')['speedup'].max())

    print("\nBest Speedup by Image Size and Kernel Size:")
    print(results.groupby(['image_size', 'kernel_size'])['speedup'].max())        

if __name__ == "__main__":
    main()
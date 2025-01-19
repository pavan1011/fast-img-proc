import fast_image_processing as fip
import time

def main():

    start_time = time.time()
    # Load an RGB image (PNG or JPG supported)
    input_image = fip.Image("input.png")
    print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_3 = fip.edge_detect(input_image, 1, 0, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))    
    cpu_edge_1_0_3.save("edge_1_0_3_cpu.png")

    start_time = time.time()
    # Smoothing on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_0_1_3 = fip.edge_detect(input_image, 0, 1, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_0_1_3.save("edge_0_1_3_cpu.png")

    start_time = time.time()
    # Derivative on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_1_1_3 = fip.edge_detect(input_image, 1, 1, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_1_3.save("edge_1_1_3_cpu.png")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_0_5.save("edge_1_0_5_cpu.png")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_0_1_5.save("edge_0_1_5_cpu.png")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_1_1_5 = fip.edge_detect(input_image, 1, 1, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_1_5.save("edge_1_1_5_cpu.png")

    # Chess BOARD
    # Load an RGB image (PNG or JPG supported)
    input_image = fip.Image("input_tiger.jpg")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_3 = fip.edge_detect(input_image, 1, 0, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_0_3.save("edge_1_0_3_cpu_tiger.jpg")

    start_time = time.time()
    # Smoothing on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_0_1_3 = fip.edge_detect(input_image, 0, 1, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_0_1_3.save("edge_0_1_3_cpu_tiger.jpg")

    start_time = time.time()
    # Derivative on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_1_1_3 = fip.edge_detect(input_image, 1, 1, 3, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_1_3.save("edge_1_1_3_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_0_5.save("edge_1_0_5_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_0_1_5.save("edge_0_1_5_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_1_1_5 = fip.edge_detect(input_image, 1, 1, 5, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_1_5.save("edge_1_1_5_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_7 = fip.edge_detect(input_image, 1, 0, 7, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_0_7.save("edge_1_0_7_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_7 = fip.edge_detect(input_image, 0, 1, 7, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_0_1_7.save("edge_0_1_7_cpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_1_1_7 = fip.edge_detect(input_image, 1, 1, 7, fip.Hardware.CPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    cpu_edge_1_1_7.save("edge_1_1_7_cpu_tiger.jpg")

    print("===== GPU ========")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    gpu_edge_1_0_3 = fip.edge_detect(input_image, 1, 0, 3, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_0_3.save("edge_1_0_3_gpu_tiger.jpg")

    start_time = time.time()
    # Smoothing on x-axis, Derivative on y-axis, kernel_size = 5
    gpu_edge_0_1_3 = fip.edge_detect(input_image, 0, 1, 3, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_0_1_3.save("edge_0_1_3_gpu_tiger.jpg")

    start_time = time.time()
    # Derivative on x-axis, Derivative on y-axis, kernel_size = 5
    gpu_edge_1_1_3 = fip.edge_detect(input_image, 1, 1, 3, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_1_3.save("edge_1_1_3_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    gpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_0_5.save("edge_1_0_5_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    gpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_0_1_5.save("edge_0_1_5_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    gpu_edge_1_1_5 = fip.edge_detect(input_image, 1, 1, 5, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_1_5.save("edge_1_1_5_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    gpu_edge_1_0_7 = fip.edge_detect(input_image, 1, 0, 7, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_0_7.save("edge_1_0_7_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    gpu_edge_0_1_7 = fip.edge_detect(input_image, 0, 1, 7, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_0_1_7.save("edge_0_1_7_gpu_tiger.jpg")

    start_time = time.time()
    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    gpu_edge_1_1_7 = fip.edge_detect(input_image, 1, 1, 7, fip.Hardware.GPU)
    print("--- %s seconds ---" % (time.time() - start_time))
    gpu_edge_1_1_7.save("edge_1_1_7_gpu_tiger.jpg")

if __name__ == "__main__":
    main()

import fast_image_processing as fip

def main():
    # Load an RGB image (PNG or JPG supported)
    input_image = fip.Image("input.png")

    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_3 = fip.edge_detect(input_image, 1, 0, 3, fip.Hardware.CPU)
    cpu_edge_1_0_3.save("edge_1_0_3_cpu.png")

    # Smoothing on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_0_1_3 = fip.edge_detect(input_image, 0, 1, 3, fip.Hardware.CPU)
    cpu_edge_0_1_3.save("edge_0_1_3_cpu.png")

    # Derivative on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_1_1_3 = fip.edge_detect(input_image, 1, 1, 3, fip.Hardware.CPU)
    cpu_edge_1_1_3.save("edge_1_1_3_cpu.png")


    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    cpu_edge_1_0_5.save("edge_1_0_5_cpu.png")

    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    cpu_edge_0_1_5.save("edge_0_1_5_cpu.png")

    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_1_1_5 = fip.edge_detect(input_image, 1, 1, 5, fip.Hardware.CPU)
    cpu_edge_1_1_5.save("edge_1_1_5_cpu.png")

    # Chess BOARD
    # Load an RGB image (PNG or JPG supported)
    input_image = fip.Image("input_chess.png")

    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_3 = fip.edge_detect(input_image, 1, 0, 3, fip.Hardware.CPU)
    cpu_edge_1_0_3.save("edge_1_0_3_chess.png")

    # Smoothing on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_0_1_3 = fip.edge_detect(input_image, 0, 1, 3, fip.Hardware.CPU)
    cpu_edge_0_1_3.save("edge_0_1_3_chess.png")

    # Derivative on x-axis, Derivative on y-axis, kernel_size = 5
    cpu_edge_1_1_3 = fip.edge_detect(input_image, 1, 1, 3, fip.Hardware.CPU)
    cpu_edge_1_1_3.save("edge_1_1_3_chess.png")


    # Derivatives on x-axis, smoothing on y-axis, kernel_size = 5
    cpu_edge_1_0_5 = fip.edge_detect(input_image, 1, 0, 5, fip.Hardware.CPU)
    cpu_edge_1_0_5.save("edge_1_0_5_chess.png")

    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_0_1_5 = fip.edge_detect(input_image, 0, 1, 5, fip.Hardware.CPU)
    cpu_edge_0_1_5.save("edge_0_1_5_chess.png")

    # Derivatives on y-axis, smoothing on x-axis, kernel_size = 5
    cpu_edge_1_1_5 = fip.edge_detect(input_image, 1, 1, 5, fip.Hardware.CPU)
    cpu_edge_1_1_5.save("edge_1_1_5_chess.png")

if __name__ == "__main__":
    main()

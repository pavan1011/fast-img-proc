import fast_image_processing as fip
import time
import os
from os import listdir
from os.path import isfile, join, isdir
import sys

def print_help():
    # If output_dir is not passed as cmd line arg, a new dir called "output_check_sbl_edge is created."
    print("Correct usage: python3 check_sbl_edge.py <input_dir or input_image> <optional: output_dir>")

def run_sbl_edge(img_file, output_dir):
    
    test_ksizes = [3, 5, 7]
    test_dx_dy_list = [[0, 1], [1, 0], [1, 1]]
    test_hardware = {"CPU" : fip.Hardware.CPU, 
                     "GPU" : fip.Hardware.GPU}
    
    input_filename, input_file_ext = os.path.splitext(os.path.basename(img_file))
    input_image = fip.Image(img_file)

    for hw in test_hardware.keys():
        print(f"============  {hw}  ============")
        for ksize in test_ksizes:
            for dx_dy in test_dx_dy_list:
                print(f"--- {input_filename}{input_file_ext} ---")
                output_filename = "output_" + \
                                    f"{input_filename}_" + \
                                    f"{dx_dy[0]}_" + \
                                    f"{dx_dy[1]}_" + \
                                    f"{ksize}_" + \
                                    f"{hw}{input_file_ext}"
                
                start_time = time.time()
                cpu_result = fip.edge_detect(input_image, 
                                                dx_dy[0], 
                                                dx_dy[1], 
                                                ksize, 
                                                test_hardware[hw])
                end_time = time.time()

                if (cpu_result.save(f"{output_dir}/{output_filename}")):
                    print(f"Saved {output_filename}")
                else:
                    print(f"Error saving {output_filename}")
                print("--- %s seconds ---" % (time.time() - start_time))

def main():

    len_cmd_args = len(sys.argv)

    if len_cmd_args > 1 and len_cmd_args <= 3:
        output_dir = "output_check_sbl_edge"
        if len_cmd_args == 3:
            output_dir = os.path.abspath(sys.argv[2])
        else: 
            print("No output dir provided." + \
                  "Creating one called output_check_sbl_edge in current dir")
        
        if not os.path.exists(output_dir):
            print(f"Output directory: {output_dir} doesn't exist. Creating it.")
            os.makedirs(output_dir)

        if isdir(os.path.abspath(sys.argv[1])):
            input_dir = os.path.abspath(sys.argv[1])
            image_list = []
            
            for f in listdir(input_dir):
                if(isfile(join(input_dir, f)) and \
                    (f.endswith('.png') or f.endswith('.jpg'))):
                        print(f"Found: {f}")
                        image_list.append(join(input_dir, f))
            
            print("======================================================")
            print(f"Calling check_sbl_edge with: ")
            print(f"input_dir: {input_dir}")
            print(f"output_dir: {output_dir}")
            print("======================================================")
            
            for img_file in image_list:
                run_sbl_edge(img_file, output_dir)
        
        else:
            input_file = os.path.abspath(sys.argv[1])
            if(isfile(input_file) and \
                    (input_file.endswith('.png') or input_file.endswith('.jpg'))):
                run_sbl_edge(input_file, output_dir)
            
            else:
                print_help()
    
    else:
        print_help()

if __name__ == "__main__":
    main()

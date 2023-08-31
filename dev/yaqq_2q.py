########################################################################################################################################################################################################

if __name__ == "__main__":

    print("\n_________________________________________________________________")
    print("\n         Welcome to YAQQ: Yet Another Quantum Quantizer.")
    print("\n         Copyright © 2023 Aritra Sarkar; AGPL v3 License")
    print("\nCode repository: https://github.com/Advanced-Research-Centre/YAQQ")
    print("_________________________________________________________________")

    devmode = input("\n  ===> Run Default Configuration? [Y/N] (def.: Y): ") or 'Y'

    if devmode == 'Y':
        
        print("\n  Option not available in PyPI version. Use the source code from GitHub dev directory.")

    else:

        print("\n  YAQQ has 4 configuration options:")
        print("     1. Data Dimension, Size and Type")
        print("     2. Cost Function")
        print("     3. Gate Decomposition Method")
        print("     4. Operation Mode and Search Technique")

        yaqq_ds_dim = int(input("\n  ===> Enter Data Set Dimension (def.: 2): ") or 2)        
        
        # yaqq_ds_size = int(input("\n  ===> Enter Data Set Size (def.: 500): ") or 500)
            
        # print("\n  ===> Choose Data Set:")
        # print("   Data Set 1 - Evenly distributed states (using golden mean)")
        # print("   Data Set 2 - Haar random states")
        # print("   Data Set 3 - Haar random unitaries")      # https://iopscience.iop.org/article/10.1088/1367-2630/ac37c8/meta
        # yaqq_ds_type = int(input("   Option (def.: 1): ") or 1)
        # match yaqq_ds_type:
        #     case 1: 
        #         ds = gen_ds_fiboS(samples=yaqq_ds_size)  # Returns list of unitary gate objects as the preparation for the state vectors from |0>
        #     case 2: 
        #         ds = gen_ds_randS(samples=yaqq_ds_size)  # Returns list of unitary gate objects as the preparation for the state vectors from |0>
        #     case 3: 
        #         ds = gen_ds_randU(samples=yaqq_ds_size)  # Returns list of unitary gate objects
        #     case _ : 
        #         print("Invalid option")
        #         exit(1)   

        # yaqq_ds_show = input("\n  ===> Visualize Data Set? [Y/N] (def.: N): ") or 'N'
        # if yaqq_ds_show == 'Y':
        #     vis_ds_randU(ds)    # Plots the states when the unitaries are applied to |0> state    

        # print("\n  ===> Choose Gate Decomposition Method:")
        # print("   Method 1 - Solovay-Kitaev Decomposition")
        # print("   Method 2 - Random Decomposition")
        # yaqq_dcmp = int(input("   Option (def.: 2): ") or 2)

        # print("\n  ===> Choose YAQQ Mode:")
        # print("   Mode 1 - Compare GS2 (in code) w.r.t. GS1 (in code)")
        # print("   Mode 2 - Generative novel GS2 w.r.t. GS1 (in code)")
        # yaqq_mode = int(input("   Option (def.: 2): ")) or 2
        # match yaqq_mode:
        #     case 1: 
        #         compare_gs(ds, yaqq_dcmp)
        #     case 2: 
        #         print("\n  ===> Choose Search Method:")
        #         print("   Method 1 - Random Gate Set Search")
        #         print("   Method 2 - U3 Angles Optimize with Multiple Random Initialization")
        #         yaqq_search = int(input("   Option (def.: 1): ")) or 1
        #         print()

        #         match yaqq_search:
        #             case 1: 
        #                 generate_gs_random(ds, yaqq_dcmp)
        #             case 2: 
        #                 generate_gs_optimize(ds, yaqq_dcmp)
        #             case _: 
        #                 print("Invalid option")
        #                 exit(1)   
        #     case _ : 
        #         print("Invalid option")
        #         exit(1)   
    
    print("\n_________________________________________________________________")
    print("\n------------------- Thank you for using YAQQ. -------------------")
    print("_________________________________________________________________")

########################################################################################################################################################################################################
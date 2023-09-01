from yaqq_ds import GenerateDataSet

########################################################################################################################################################################################################

if __name__ == "__main__":

    print("\n_____________________________________________________________________")
    print("\n           Welcome to YAQQ: Yet Another Quantum Quantizer.           ")
    print("\nCopyright © 2023 Quantum Intelligence Research Group; AGPL v3 License")
    print("\n  Code repository: https://github.com/Advanced-Research-Centre/YAQQ  ")
    print("_____________________________________________________________________")

    devmode = input("\n  ===> Run Default Configuration? [Y/N] (def.: Y): ") or 'Y'

    if devmode == 'Y':
        
        yaqq_ds_dim = 2
        yaqq_ds_type = 4
        yaqq_ds_size = 10

    elif devmode == 'N':

        print("\n  YAQQ has 4 configuration options:")
        print("     1. Data Dimension, Type and Size")
        print("     2. Cost Function")
        print("     3. Gate Decomposition Method")
        print("     4. Operation Mode and Search Technique")

        yaqq_ds_dim = int(input("\n  ===> Enter Data Set Dimension (def.: 2): ") or 2)        
        
        if yaqq_ds_dim == 1:
            print("\n Data Set Types for Dimension = 1:")
            print("     1. Haar Random 1-qubit pure States")
            print("     2. Haar Random 2x2 Unitaries")
            print("     3. Equispaced States on Bloch Sphere using Golden mean")
            yaqq_ds_type = int(input("\n  ===> Enter Data Set Type (def.: 3): ") or 3)
        elif yaqq_ds_dim == 2:
            print("\n Data Set Types for Dimension = 2:")
            print("     1. Haar Random 2-qubit pure States")
            print("     2. Haar Random 4x4 Unitaries")
            print("     3. Random Non-local Unitaries on Weyl chamber")
            print("     4. Equispaced Non-local Unitaries on Weyl chamber")
            yaqq_ds_type = int(input("\n  ===> Enter Data Set Type (def.: 4): ") or 4)
        else:
            print("\n Data Set Types for Dimension = 3+:")
            print("     1. Haar Random n-qubit pure States")
            print("     2. Haar Random 2^nx2^n Unitaries")
            yaqq_ds_type = int(input("\n  ===> Enter Data Set Type (def.: 2): ") or 2)

    gds = GenerateDataSet()
    if yaqq_ds_dim == 2 and yaqq_ds_type == 4:
        yaqq_ds_reso = int(input("\n  ===> Enter Weyl Chamber cx Spacing (def.: 23, 508 points): ") or 23)
        yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, None, yaqq_ds_reso)
    else:
        yaqq_ds_size = int(input("\n  ===> Enter Data Set Size (def.: 500): ") or 500)
        yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, yaqq_ds_size, None)

    if yaqq_ds_dim <= 2:
        yaqq_ds_show = input("\n  ===> Visualize Data Set? [Y/N] (def.: N): ") or 'N'
        if yaqq_ds_show == 'Y':
            pass


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
    
    print("\n_____________________________________________________________________")
    print("\n--------------------- Thank you for using YAQQ. ---------------------")
    print("_____________________________________________________________________")

########################################################################################################################################################################################################
from yaqq_ds import GenerateDataSet, VisualizeDataSet, ResultsPlotSave
from yaqq_nus import NovelUniversalitySearch
import configparser   # other choices: json, yaml, csv, xml
import json

########################################################################################################################################################################################################

if __name__ == "__main__":

    print("\n_____________________________________________________________________")
    print("\n           Welcome to YAQQ: Yet Another Quantum Quantizer.           ")
    print("\nCopyright © 2023 Quantum Intelligence Research Group; AGPL v3 License")
    print("\n  Code repository: https://github.com/Advanced-Research-Centre/YAQQ  ")
    print("_____________________________________________________________________")

    gds = GenerateDataSet()
    vds = VisualizeDataSet()
    nsa = NovelUniversalitySearch()
    rps = ResultsPlotSave()

    devmode = input("\n  ===> Run Default Configuration? [Y/N] (def.: Y): ") or 'Y'

    if devmode == 'Y':
        autocfg = True

        Config = configparser.ConfigParser()
        cfg_fname = input("\n  ===> Enter Configuration Filename (def.: QART_eid-0001): ") or 'HQEC_eid-0001'
        Config.read("configs/"+cfg_fname+".cfg")
        
        yaqq_mode = int(Config['experiment']['yaqq_mode'])

        # [ 1q:'rand','skt' | 2q:'rand','kak' | 3+q:'rand','qsd' ]
        # All possible options for testing: 
        #   3+q: [x,x,1], [1,1,2], [1,2,2], [2,1,2], [2,2,2]
        #    2q: [x,1,-], [1,2,-], [2,2,-]
        #    1q: [1,-,-], [2,-,-]
        # Most analytical, best fidelity, slow = [2,2,2] : 'skt','kak','qsd'   
        # Least analytical, bad fidelity, fast = [1,1,1] : 'rand','rand','rand' 
        yaqq_cf_dcmp_gs1 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs1'])
        yaqq_cf_dcmp_gs2 = json.loads(Config['experiment']['yaqq_cf_dcmp_gs2'])
        nsa.cnfg_dcmp(yaqq_cf_dcmp_gs1,yaqq_cf_dcmp_gs2)

        if yaqq_mode == 1 or yaqq_mode == 2:
            yaqq_ds_dim = int(Config['mode2']['yaqq_ds_dim'])
            yaqq_ds_type = int(Config['mode2']['yaqq_ds_type'])                  
            if yaqq_ds_dim == 1 and yaqq_ds_type == 4:
                yaqq_ds_reso = int(Config['mode2']['yaqq_ds_reso'])
                yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, None, yaqq_ds_reso)
            elif yaqq_ds_dim == 2 and yaqq_ds_type == 4:
                yaqq_ds_reso = int(Config['mode2']['yaqq_ds_reso'])
                yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, None, yaqq_ds_reso)
            else:
                yaqq_ds_size = int(Config['mode2']['yaqq_ds_size'])
                yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, yaqq_ds_size, None)
            yaqq_ds_show = Config['mode2']['yaqq_ds_show']
            if yaqq_ds_show == 'Y':
                if yaqq_ds_dim == 1:
                    vds.vis_ds_Bloch(yaqq_ds)
                else:
                    vds.vis_ds_Weyl(yaqq_ds)

            if yaqq_mode == 1:
                yaqq_cf_wgts = json.loads(Config['experiment']['yaqq_cf_wgts'])
                nsa.cnfg_wgts(yaqq_cf_wgts)
                yaqq_cf_ngs = Config['mode1']['yaqq_cf_ngs'].split(',')
                yaqq_ngs_search = Config['mode2']['optimize']
                gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db, opt_params = nsa.nusa(yaqq_ds,yaqq_cf_ngs,yaqq_ngs_search, autocfg, Config)
                for i in gs2:
                    print(i, gs2[i])
                print(opt_params)
            else:       
                gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db = nsa.compare_gs(yaqq_ds, autocfg, Config)   
            
            rps.plot_compare_gs(gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db, pfivt = True) 

        else:
            nsa.decompose_u(autocfg, Config)

        print("\n_____________________________________________________________________")
        print("\n--------------------- Thank you for using YAQQ. ---------------------")
        print("_____________________________________________________________________")

    elif devmode == 'N':
        autocfg = False

        print("\n  YAQQ has 4 configuration options:")
        print("     1. Operation Mode")
        print("     2. Gate Decomposition Method")
        print("     3. Cost Function")
        print("     4. Data Dimension, Type and Size")

        # TBD: Checks for invalid configurations
        print("\n Operation Mode:")
        print("   1. Generative novel GS2 w.r.t. GS1 (in code)")
        print("   2. Compare GS2 (in code) w.r.t. GS1 (in code)")
        print("   3. Decompose a n-qubit U (in code) w.r.t. GS1 (in code)")
        yaqq_mode = int(input("\n  ===> Enter YAQQ Mode (def.: 1): ") or 1)

        yaqq_cf_dcmp_gs1 = []
        yaqq_cf_dcmp_gs2 = []
        print("\n Gate Decomposition Method for Dimension = 1:")
        print("     1. Random Decomposition (trails auto-scale w.r.t. dimension)")
        print("     2. Solovay-Kitaev Decomposition")
        yaqq_cf_dcmp_gs1.append(int(input("\n  ===> Enter Gate Set 1 Decomposition Method for Dimension = 1 (def.: 1): ") or 1))
        if yaqq_mode != 3:
            yaqq_cf_dcmp_gs2.append(int(input("\n  ===> Enter Gate Set 2 Decomposition Method for Dimension = 1 (def.: 1): ") or 1))
        print("\n Gate Decomposition Method for Dimension = 2:")
        print("     1. Random Decomposition (trails auto-scale w.r.t. dimension)")
        print("     2. Cartan Decomposition")
        yaqq_cf_dcmp_gs1.append(int(input("\n  ===> Enter Gate Set 1 Decomposition Method for Dimension = 2 (def.: 2): ") or 2))
        if yaqq_mode != 3:
            yaqq_cf_dcmp_gs2.append(int(input("\n  ===> Enter Gate Set 2 Decomposition Method for Dimension = 2 (def.: 2): ") or 2))
        print("\n Gate Decomposition Method for Dimension = 3+:")
        print("     1. Random Decomposition (trails auto-scale w.r.t. dimension)")
        print("     2. Quantum Shannon Decomposition")
        yaqq_cf_dcmp_gs1.append(int(input("\n  ===> Enter Gate Set 1 Decomposition Method for Dimension = 3+ (def.: 2): ") or 2))
        if yaqq_mode != 3:
            yaqq_cf_dcmp_gs2.append(int(input("\n  ===> Enter Gate Set 2 Decomposition Method for Dimension = 3+ (def.: 2): ") or 2))
        nsa.cnfg_dcmp(yaqq_cf_dcmp_gs1,yaqq_cf_dcmp_gs2)

        if yaqq_mode == 3:
            nsa.decompose_u(autocfg)    # Dataset and costfunction selection not required for this mode
            print("\n_____________________________________________________________________")
            print("\n--------------------- Thank you for using YAQQ. ---------------------")
            print("_____________________________________________________________________")
            exit()

        yaqq_ds_dim = int(input("\n  ===> Enter Data Set Dimension (def.: 2): ") or 2)        
        if yaqq_ds_dim == 1:
            print("\n Data Set Types for Dimension = 1:")
            print("     1. Haar Random 1-qubit pure States")
            print("     2. Haar Random 2x2 Unitaries")
            print("     3. Equispaced States on Bloch Sphere using Golden mean")
            print("     4. Equispaced Angles on Bloch Sphere")
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

        if yaqq_ds_dim == 1 and yaqq_ds_type == 4:
            yaqq_ds_reso = int(input("\n  ===> Enter Bloch Sphere a_rz Spacing (def.: 16, 512 points): ") or 16)
            yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, None, yaqq_ds_reso)
        elif yaqq_ds_dim == 2 and yaqq_ds_type == 4:
            yaqq_ds_reso = int(input("\n  ===> Enter Weyl Chamber cx Spacing (def.: 23, 508 points): ") or 23)
            yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, None, yaqq_ds_reso)
        else:
            yaqq_ds_size = int(input("\n  ===> Enter Data Set Size (def.: 508): ") or 508)
            yaqq_ds = gds.yaqq_gen_ds(yaqq_ds_dim, yaqq_ds_type, yaqq_ds_size, None)

        if yaqq_ds_dim <= 2:
            yaqq_ds_show = input("\n  ===> Visualize Data Set? [Y/N] (def.: N): ") or 'N'
            if yaqq_ds_show == 'Y':
                if yaqq_ds_dim == 1:
                    vds.vis_ds_Bloch(yaqq_ds)
                else:
                    vds.vis_ds_Weyl(yaqq_ds)

        if yaqq_mode == 2:
            gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db = nsa.compare_gs(yaqq_ds, autocfg)     # Costfunction selection not required for this mode
            rps.plot_compare_gs(gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db, pfivt = True) 
            print("\n_____________________________________________________________________")
            print("\n--------------------- Thank you for using YAQQ. ---------------------")
            print("_____________________________________________________________________")
            exit()

        yaqq_cf_wgts = [int(i) for i in (input("\n  ===> Enter Cost Function Weights (def.: [1,1,1,1,0]): ") or '1,1,1,1,0').split(',')]
        print("\n  ===> Cost Function Weights: ", yaqq_cf_wgts)
        nsa.cnfg_wgts(yaqq_cf_wgts)

        print("\n Novel Gate Set Composition:")
        print("   R1: Haar Random 1-qubit Unitary")                         # Search: random
        print("   P1: Parametric 1-qubit Unitary (IBM U3)")                 # Search: parametric, random
        print("   G1: Golden 1-qubit Unitary")                              # TBD
        print("   SG1: Super Golden 1-qubit Unitary")                       # TBD
        print("   T1: T Gate 1-qubit Unitary")                              # Constant
        print("   TD1: T-dagger Gate 1-qubit Unitary")                      # Constant
        print("   H1: H (Hadamard) Gate 1-qubit Unitary")                   # Constant
        print("   F1: Load 1-qubit Unitary Gate definition from File")      # Constant TBD Extension
        if yaqq_ds_dim >= 2:
            print("   R2: Haar Random 2-qubit Unitary")                     # Search: random
            print("   NL2: Non-local 2-qubit Unitary")                      # Search: parametric, random
            print("   CX2: CNOT Gate 2-qubit Unitary")                      # Constant
            print("   B2: B (Berkeley) Gate 2-qubit Unitary")               # Constant
            print("   PE2: Perfect Entangler 2-qubit Unitary")              # TBD
            print("   SPE2: Special Perfect Entangler 2-qubit Unitary")     # Search: parametric, random
            print("   F2: Load 2-qubit Unitary Gate definition from File")  # Constant TBD Extension
        yaqq_cf_ngs = (input("\n  ===> Enter Gate Set (def.: [R1,R1,R1]): ") or 'R1,R1,R1').split(',')

        print("\n Search Method:")
        print("   1: Random Search for non-constant gates")
        print("   2: Parametric Search (SciPy) for non-constant gates")
        yaqq_ngs_search = input("\n  ===> Search Method as Optimize? (def.: Y): ") or 'Y'

        gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db, opt_params = nsa.nusa(yaqq_ds,yaqq_cf_ngs,yaqq_ngs_search, autocfg)
        print("Novel Parameters for GS2: ", opt_params)
        rps.plot_compare_gs(gs1, gs1_gates, pf01_db, cd01_db, gs2, gs2_gates, pf02_db, cd02_db, pfivt = True)  
        print("\n_____________________________________________________________________")
        print("\n--------------------- Thank you for using YAQQ. ---------------------")
        print("_____________________________________________________________________")

########################################################################################################################################################################################################
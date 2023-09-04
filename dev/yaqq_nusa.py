class NovelUniversalitySearchAgent:

    # ------------------------------------------------------------------------------------------------ #

    """
    Configure the weights of the elements of the cost function
    """

    def cnfg_cfn(self, wgts):
        self.w_apf = wgts[0]   # Weight of average process fidelity
        self.w_npf = wgts[1]   # Weight of novelty of process fidelity
        self.w_acd = wgts[2]   # Weight of average circuit depth
        self.w_ncd = wgts[3]   # Weight of novelty of circuit depth
        self.w_agf = wgts[4]   # Weight of average gate fabrication
        return
    
    # ------------------------------------------------------------------------------------------------ #

    """
    Calculate cost function based on distribution of process fidelity differences and gate depths of two gate sets
    """

    def cfn_calc(pf01_db,cd01_db,pf02_db,cd02_db):
        ivt_pf_gs01 = np.subtract(1,pf01_db)
        dist_pf_novelty = np.mean(abs(np.subtract(ivt_pf_gs01,pf02_db)))
        ivt_cd_gs01 = np.subtract(max(cd01_db),cd01_db)
        dist_cd_novelty = np.mean(abs(np.subtract(ivt_cd_gs01,cd02_db)))
        dist_pf_avg = - np.mean(pf02_db)
        dist_cd_avg = np.mean(cd02_db)
        w_pf_trend, w_cd_trend, w_pf_avg, w_cd_avg = 100, 100, 1000, 1
        cfn = w_pf_trend*dist_pf_novelty + w_cd_trend*dist_cd_novelty + w_pf_avg*dist_pf_avg + w_cd_avg*dist_cd_avg
        return cfn

    def nusa():
        return
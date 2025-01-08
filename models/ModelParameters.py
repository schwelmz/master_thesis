class ModelParameters:
    """Container for model parameters to avoid global variables"""
    def __init__(self, model, params_dict, is_dimensionless):
        if model == "NL":
            self._init_NL(params_dict)
            if is_dimensionless:
                self._compute_NL_dimensionless()
        elif model == "GM":
            print("ja")
            self._init_GM(params_dict)
            
    def _init_NL(self, params):
        # Nodal-Lefty parameters
        self.alpha_N = float(params.get('alpha_N', 0))
        self.alpha_L = float(params.get('alpha_L', 0))
        self.n_N = float(params.get('n_N', 0))
        self.n_L = float(params.get('n_L', 0))
        self.K_N = float(params.get('K_N', 0))
        self.K_L = float(params.get('K_L', 0))
        self.gamma_N = float(params.get('gamma_N', 0))
        self.gamma_L = float(params.get('gamma_L', 0))
        self.D_N = float(params.get('D_N', 0))
        self.D_L = float(params.get('D_L', 0))
        
    def _compute_NL_dimensionless(self):
        # Compute dimensionless parameters
        self.alpha_N_dimless = self.alpha_N / (self.gamma_N * self.K_N)
        self.alpha_L_dimless = self.alpha_L / (self.gamma_N * self.K_L)
        self.gamma_dimless = self.gamma_L / self.gamma_N
        self.d = self.D_L / self.D_N
    
    def _init_GM(self, params):
        # Gierer-Meinhardt parameters
        self.D_u = float(params.get('D_u', 0))
        self.D_v = float(params.get('D_v', 0))
        self.mu = float(params.get('mu', 0))
        self.a = float(params.get('a', 0))
        self.c = float(params.get('c', 0))
        self.r = float(params.get('r', 0))
    
    def print(self):
        print('Model parameters:')
        print('\n'.join("\t%s: %s" % (key,value) for key,value in vars(self).items()))
    
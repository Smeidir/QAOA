supported_qaoa_variants = ['vanilla', 'multiangle','recursive']
supported_param_inits = ['uniform','gaussian','static']
supported_optimizers = [
    'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B', 'TNC', 
    'COBYLA', 'COBYQA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-ncg', 
    'trust-exact', 'trust-krylov'
]
supported_backends = ['statevector', 'density_matrix_simulation', 'noisy_sampling', 'quantum_backend']
api_key = '6f1e0cb3bbbbdb6720ed8a755437ee0b3085a54f33274090146083fc7cc0dc9b5aea059a954e1618466f0d290dd5d8f18e8856e7539e5ec7ed38fd167efa0338'
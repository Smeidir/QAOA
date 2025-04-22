import numpy as np

def get_init_params(strategy: str , depth: int, cost_length : int = 1, mixer_length :int = 1)-> np.ndarray:  #trying to learn good code quality from chat :))) god help me

    """
    Initializes parameters for QAOA variants.

    Args:
        strategy: 'uniform', 'gaussian', or 'static'
        depth: QAOA depth p
        cost_length: number of qubits
        mixer_length: number of Pauli terms in cost Hamiltonian
        
    Returns:
        np.ndarray of shape (num_params,)
    """
    if strategy == "uniform":
        return _init_uniform(depth, cost_length, mixer_length)
    elif strategy == "gaussian":
        return _init_gaussian(depth, cost_length, mixer_length)
    elif strategy == "static":
        return _init_static(depth, cost_length, mixer_length)
    else:
        raise ValueError(f"Unsupported init strategy: {strategy}")

def _init_uniform(depth, cost_length, mixer_length):
    init_params = np.concatenate([
        np.concatenate([np.random.uniform(0, 2*np.pi, cost_length), 
                        np.random.uniform(0, np.pi, mixer_length)])
        for _ in range(depth)
    ])
    init_params =init_params.flatten()
    return init_params
def _init_gaussian(depth, cost_length, mixer_length):

    init_params = np.concatenate([
        np.concatenate([np.random.normal(np.pi,0.2,cost_length), 
                        (np.random.normal(np.pi/2,0.1,mixer_length))])
        for _ in range(depth)
    ])
    init_params =init_params.flatten()
    return init_params
def _init_static(depth, cost_length, mixer_length):
    optimized_params_dict = {1: [-np.pi/6, -np.pi/8],
    2: [-np.pi/6, -np.pi/8, -np.pi/4, -np.pi/13],
    3: [-np.pi*0.12641,np.pi*0.15244,np.pi*-0.24101,np.pi*-0.10299,np.pi*-0.27459,np.pi*-0.06517 ]} #https://arxiv.org/pdf/2102.06813
    if depth <= 3:
        init_params = optimized_params_dict[depth]
    else:
        step_size = 0.8/(depth-1)
        #final_values = 0.9/0.1
        init_params = np.array([[0.1 + n*step_size, 0.9-n*step_size] for n in range(0,depth)])
        init_params = init_params.flatten()

        
    new_params = [] #This does nothing if cost and mixer length are 1. 
    for i, param in enumerate(init_params):
        if i%2 == 0: #cost parameter
            new_params.extend([param]*cost_length)
        else: #mixer parameter
            new_params.extend([param]*mixer_length)
    init_params = np.array(new_params)
    return init_params

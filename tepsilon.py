modified_solution = [0,1,0,1,1,0]
print(modified_solution)
import numpy as np
modified_solution = [x + (1-2*x)*0.3 for x in modified_solution]
print(modified_solution)
print(2*np.arcsin(np.sqrt(modified_solution)))
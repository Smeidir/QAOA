import numpy as np
#helper functions. I don't want to risk read/write queues if they're tied to an object.

def to_bitstring(integer, num_bits): #helper function
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]
def to_bitstring_str(integer, num_bits): #helper function
    result = np.binary_repr(integer, width=num_bits)
    return result
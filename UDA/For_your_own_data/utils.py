import numpy as np

def recommended_radius(num_class):
    if num_class <= 5:
        return 5.0
    elif num_class <=20:
        return 8.5
    elif num_class <=50:
        return 10.0
    else:
        return 20.0

def recommended_bottleneck_dim(num_class):
    j = 8
    while True:
        if 3*num_class <= 256:
            dim = 256
            break
        elif 3*num_class > 2**j and 3*num_class <= 2**(j+1):
            dim = 2**(j+1)
            break
        j += 1
    return dim
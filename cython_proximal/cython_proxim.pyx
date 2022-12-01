
from libcpp cimport bool

import numpy as np
cimport numpy as np
cimport openmp

#DTYPE = np.int
#ctypedef np.int DTYPE_t

from cython.parallel import prange
cimport cython
from libc.math cimport tanh


# https://stackoverflow.com/questions/21851985/difference-between-np-int-np-int-int-and-np-int-t-in-cython


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # for modulo being fast


def cython_compute_proximal(np.int32_t max_delay,
                            np.int32_t inf_val,
                            np.int32_t states_dim,
                            np.ndarray[np.float64_t, ndim=2] events_history,
                            np.ndarray[np.float64_t, ndim=2] right_proximal,
                            np.ndarray[np.float64_t, ndim=2] left_proximal):

    cdef np.int32_t r, c
    cdef np.float64_t counter

    # for r in range(states_dim):
    for r in prange(states_dim,  nogil=True, schedule='static', num_threads=1):
        counter = inf_val
        for c in range(max_delay):
            if events_history[r, c] == 1:
                counter = 0
            elif counter < inf_val:
                counter = counter + 1

            right_proximal[r, c] = counter

        counter = inf_val
        for c in range(max_delay-1, -1, -1):
            if events_history[r, c] == 1:
                counter = 0
            elif counter < inf_val:
                counter = counter + 1

            left_proximal[r, c] = counter
    return

































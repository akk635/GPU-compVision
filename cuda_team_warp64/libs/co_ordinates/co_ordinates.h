#ifndef CO_ORDINATES_H
#define CO_ORDINATES_H

#include <stdlib.h>

// linearize a given coordinate in a dimension
__device__ size_t linearize_coords(dim3 coords, dim3 dims);

#endif
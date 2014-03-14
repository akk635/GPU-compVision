#include "co_ordinates.h"

__device__ size_t linearize_coords(dim3 coords, dim3 dims) {
	return (size_t) coords.z * dims.x * dims.y + (size_t) coords.y * dims.x + coords.x;
}
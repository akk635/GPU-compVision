

__device__ linearize(dim3 coord, dim3 dims) {
	return coord.z * dims.x * dims.y + coord.y * dims.x + coord.x;
}
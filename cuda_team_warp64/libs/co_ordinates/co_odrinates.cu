

__device__ linearize_coords(dim3 coords, dim3 dims) {
	return coords.z * dims.x * dims.y + coords.y * dims.x + coords.x;
}
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi; 

__global__ void
PageRank(const float *outedge, const float *inedge)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  float sum = 0;
  sum += inedge[i];

}



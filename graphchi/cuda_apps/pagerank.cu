extern "C" __global__ void
pagerank_kernel(const float *d_vertices, const int *d_edge_index, const float *d_edges, const int *d_edge_num)
{
  float sum = 0;
  float pagerank = 0;
  float pagerankcount = 0;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int num_in = d_edge_num[i * 2];
  int num_out = d_edge_num[i * 2 + 1];
  int pos_in = d_edge_index[i * 2];
  int pos_out = d_edge_index[i * 2 +1];
  for (int j=pos_in; j < pos_in+num_in; j++) {
    sum = sum + d_edges[j];
  }
  pagerank = 0.15 + (1 - 0.15) * sum;
  if (num_out > 0) {
    pagerankcount = pagerank / num_out;
    for (int j=pos_out; j < pos_out+num_out; j++) {
      d_edges[j] = pagerankcount;
    }
  }

}


extern "C" __global__ void
pagerank_kernel(float *d_vertices, const int *d_edge_index, float *d_edges, const int *d_edge_num, int size)
{
  float sum = 0;
  float pagerank = 0;
  float pagerankcount = 0;
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= size) return;
  int num_in = d_edge_num[i * 2];
  int num_out = d_edge_num[i * 2 + 1];
  int pos_in = d_edge_index[i * 2];
  int pos_out = d_edge_index[i * 2 +1];
  for (int j=pos_in; j < pos_in+num_in; j++) {
    sum = sum + d_edges[j];
  }
  pagerank = (float) 0.15 + (float) 0.85  * sum;
  if (num_out > 0) {
    pagerankcount = pagerank / num_out;
    for (int j=pos_out; j < pos_out+num_out; j++) {
      d_edges[j] = pagerankcount;
    }
  }
  d_vertices[i] = pagerank;
}


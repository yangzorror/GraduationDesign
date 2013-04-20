#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>

#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

void PageRank(std::vector<svertex_t> &vertices, int iter_num)
{
  cudaError_t err = cudaSuccess;
  size_t size = vertices.size();
  float *vertices = (float *)malloc(size * sizeof(float));
  int *edge_index = (int *)malloc(size * 2 * sizeof(float));
  int tot_edges = 0;

  for (int i = 0; i < size; i++)
  {
    tot_edges = tot_edges + vertices[i].num_outedges() + vertices[i].num_inedges();
    edge_index[i * 2 + 1] = vertices[i].num_outedges();
    edge_index[i*2] = vertices[i].num_inedges();
    vertices[i] = vertices[i].get_data();
  }

  float *edges = (float *)malloc(tot_edges * sizeof(float));
  int j = 0;
  for (int i =0; i < size; i++)
  {
    for (int k = 0; k < vertices[i].num_inedges(); k++)
    {
      edges[j] = vertices[i].inedges[k]->get_data();
      j++;
    }
    for (int k = 0; k < vertices[i].num_outedges(); k++)
    {
      edges[j] = vertices[i].outedges[k]->get_data();
      j++;
    }
  }
  float *d_vertices = NULL;
  err = cudaMalloc((void **)&d_vertices, size * sizeof(float));
  err = cudaMemcpy(d_vertices, vertices, size * sizeof(float), cudaMemcpyHostToDevice);
  float *d_edge_index = NULL;
  err = cudaMalloc((void **)&d_edge_index, size * 2 * sizeof(float));
  err = cudaMemcpy(d_edge_index, edge_index, size * 2 * sizeof(float), cudaMemcpyHostToDevice);
  float *d_edges = NULL;
  err = cudaMalloc((void **)&d_edges, tot_edges *  sizeof(float));
  err = cudaMemcpy(d_edges, edges, tot_edges *  sizeof(float), cudaMemcpyHostToDevice);
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  KernelPageRank<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, d_edge_index, d_edges, size);

}  

__global__ void
KernelPageRank(const int num, const float *d_vertices, const float *d_edge_index, const float *d_edges)
{
  float sum = 0;
  float pagerank = 0;
  float pagerankcount = 0;

}

                                  

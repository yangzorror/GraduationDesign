#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <cuda_runtime.h>

#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

using namespace graphchi;

struct cuda_vertex
{
  
}


void PageRank(std::vector<svertex_t> &vertices)
{
  size_t size = vertices.size();

}

__global__ void
KernelPageRank(const int num, std::vector<svertex_t> &vertices)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  verices[i] 
}



int main(int argc, const char ** argv) {
   graphchi_init(argc, argv);
   metrics m("pagerank");
 
   /* Parameters */
   std::string filename    = get_option_string("file"); // Base filename
   int niters              = get_option_int("niters", 4);
   bool scheduler          = false;                    // Non-dynamic version of pagerank.
   int ntop                = get_option_int("top", 20);

   /* Process input file - if not already preprocessed */
   int nshards             = convert_if_notexists<EdgeDataType>(filename, get_option_string("nshards", "auto"));

   /* Run */
   graphchi_engine<float, float> engine(filename, nshards, scheduler, m);
   engine.set_modifies_inedges(false); // Improves I/O performance.
   PagerankProgram program;
   engine.run(program, niters);
   
   /* Output top ranked vertices */
   std::vector< vertex_value<float> > top = get_top_vertices<float>(filename, ntop);
   std::cout << "Print top " << ntop << " vertices:" << std::endl;
   for(int i=0; i < (int)top.size(); i++) {
     std::cout << (i+1) << ". " << top[i].vertex << "\t" << top[i].value << std::endl;
   }

   metrics_report(m);
   return 0;
}

                                  

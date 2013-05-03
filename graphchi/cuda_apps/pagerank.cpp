#include <stdio.h>
#include <string>
#include <iostream>
#include <cstring>

#include <cmath>
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>
#include <helper_functions.h>

#include "graphchi_basic_includes.hpp"
#include "util/toplist.hpp"

//using namespace std;
using namespace graphchi;


CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction pagerank_kernel;
CUdeviceptr d_vertices;
CUdeviceptr d_edge_index;
CUdeviceptr d_edges;
CUdeviceptr d_edge_num;
bool noprompt = true;

// Functions
void Cleanup(bool);
CUresult CleanupNoFailure();
void RandomInit(float *, int);
bool findModulePath(const char *, std::string &, char **, std::string &);

int *pArgc = NULL;
char **pArgv = NULL;

//define input ptx file for different platforms
#define PTX_FILE "pagerank_kernel.ptx"

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
	std::fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
	exit(EXIT_FAILURE);
    }
}

inline int cudaDeviceInit(int ARGC, char **ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (CUDA_SUCCESS == err) checkCudaErrors(cuDeviceGetCount(&deviceCount));
    if (deviceCount == 0)
    {
	std::fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");
    if (dev < 0) dev = 0;
    if (dev > deviceCount-1)
    {
	std::fprintf(stderr, "\n");
	std::fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
	std::fprintf(stderr, ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
	std::fprintf(stderr, "\n");
        return -dev;
    }
    checkCudaErrors(cuDeviceGet(&cuDevice, dev));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);
    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
      std::printf("> Using CUDA Device [%d]: %s\n", dev, name);
    }
    return dev;
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
    return nGpuArchCoresPerSM[7].Cores;
}
// end of GPU Architecture definitions


// This function returns the best GPU based on performance
inline int getMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    checkCudaErrors(cuDeviceGetCount(&device_count));

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major > 0 && major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, major);
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceGetAttribute(&multiProcessorCount,
                                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             current_device));
        checkCudaErrors(cuDeviceGetAttribute(&clockRate,
                                             CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             current_device));
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major == 9999 && minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
        }

        int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

        if (compute_perf  > max_compute_perf)
        {
            // If we find GPU with SM major > 2, search only these
            if (best_SM_arch > 2)
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }

        ++current_device;
    }

    return max_perf_device;
}

// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDevice(int argc, char **argv, int *p_devID)
{
    CUdevice cuDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = cudaDeviceInit(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        char name[100];
        devID = getMaxGflopsDeviceId();
        checkCudaErrors(cuDeviceGet(&cuDevice, devID));
        cuDeviceGetName(name, 100, cuDevice);
        printf("> Using CUDA Device [%d]: %s\n", devID, name);
    }

    cuDeviceGet(&cuDevice, devID);

    if (p_devID)
    {
        *p_devID = devID;
    }

    return cuDevice;
}

CUresult CleanupNoFailure()
{
    CUresult error;
    // Free device memory
    if (d_vertices) error = cuMemFree(d_vertices);
    if (d_edge_index) error = cuMemFree(d_edge_index);
    if (d_edges) error = cuMemFree(d_edges);
    if (d_edge_num) error = cuMemFree(d_edge_num);
    // Free host memory
    //if (h_A) free(h_A);
    //if (h_B) free(h_B);
    //if (h_C) free(h_C);
    error = cuCtxDetach(cuContext);
    return error;
}

void Cleanup(bool noError)
{
    CUresult error;
    error = CleanupNoFailure();
    if (!noError || error != CUDA_SUCCESS)
    {
        printf("Function call failed\nFAILED\n");
        exit(EXIT_FAILURE);
    }
    if (!noprompt)
    {
        printf("\nPress ENTER to exit...\n");
        fflush(stdout);
        fflush(stderr);
        getchar();
    }
}

bool inline
findModulePath(const char *module_file, std::string &module_path, char **argv, std::string &ptx_source)
{
    //char *actual_path = sdkFindFilePath(module_file, argv[0]);
    std::string actual_path = "./pagerank_kernel.ptx";

   // if (actual_path != NULL)
   // {
        module_path = actual_path;
   // }
    /*else
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }*/

    if (module_path.empty())
    {
        printf("> findModulePath could not find file: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath found file at <%s>\n", module_path.c_str());

        if (module_path.rfind(".ptx") != std::string::npos)
        {
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }

        return true;
    }
}
// end of CUDA Helper Functions

// Host code

typedef float VertexDataType;
typedef float EdgeDataType;

struct PagerankProgram : public GraphChiProgram<VertexDataType, EdgeDataType> {

  void PageRank(float *h_vertices, int *edge_num, int *edge_index, float *edges, size_t size, int tot_edges)
  {
    logstream(LOG_INFO) << tot_edges  << std::endl;
    int argc = NULL;
    char **argv = NULL;
    pArgc = &argc;
    pArgv = argv;
    int devID = 0;
    CUresult error;
    // Initialize
    checkCudaErrors(cuInit(0));
    // Get number of devices supporting CUDA
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (error != CUDA_SUCCESS) Cleanup(false);
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        Cleanup(false);
    }
    if (devID < 0)
    {
        devID = 0;
    }
    if (devID > deviceCount-1)
    {
        fprintf(stderr, "(Device=%d) invalid GPU device.  %d GPU device(s) detected.\nexiting...\n", devID, deviceCount);
        CleanupNoFailure();
        exit(EXIT_SUCCESS);
    }
    else
    {
        int major, minor;
        char deviceName[100];
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, devID));
        checkCudaErrors(cuDeviceGetName(deviceName, 256, devID));
        printf("> Using Device %d: \"%s\" with Compute %d.%d capability\n", devID, deviceName, major, minor);
    }
    // pick up device with zero ordinal (default, or devID)
    error = cuDeviceGet(&cuDevice, devID);
    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }
    // Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }
    // first search for the module path before we load the results
    std::string module_path, ptx_source;

    if (!findModulePath(PTX_FILE, module_path, argv, ptx_source))
    {
        if (!findModulePath("pagerank_kernel.cubin", module_path, argv, ptx_source))
        {
            printf("> findModulePath could not find <vectorAdd> ptx or cubin\n");
            Cleanup(false);
        }
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    // Create module from binary file (PTX or CUBIN)
    if (module_path.rfind("ptx") != std::string::npos)
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void*[jitNumOptions];
        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;
        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;
        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;
        error = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
    }
    else
    {
        error = cuModuleLoad(&cuModule, module_path.c_str());
    }
    if (error != CUDA_SUCCESS) Cleanup(false);
    // Get function handle from module
    error = cuModuleGetFunction(&pagerank_kernel, cuModule, "pagerank_kernel");
    if (error != CUDA_SUCCESS) Cleanup(false);
    // Allocate vectors in device memory
    error = cuMemAlloc(&d_vertices, size * sizeof(float));
    error = cuMemAlloc(&d_edge_num, size * 2 * sizeof(int));
    error = cuMemAlloc(&d_edge_index, size * 2 * sizeof(int));
    error = cuMemAlloc(&d_edges, tot_edges *  sizeof(float));
    if (error != CUDA_SUCCESS) Cleanup(false);
    
    // Copy vectors from host memory to device memory
    error = cuMemcpyHtoD(d_vertices, h_vertices, size * sizeof(float));
    error = cuMemcpyHtoD(d_edge_num, edge_num, size * 2 * sizeof(int));
    error = cuMemcpyHtoD(d_edge_index, edge_index, size * 2 * sizeof(int));
    error = cuMemcpyHtoD(d_edges, edges, tot_edges * sizeof(float));
    if (error != CUDA_SUCCESS) Cleanup(false);
    
    /*cudaError_t err = cudaSuccess;
    float *d_vertices = NULL;
    err = cudaMalloc((void **)&d_vertices, size * sizeof(float));
    err = cudaMemcpy(d_vertices, h_vertices, size * sizeof(float), cudaMemcpyHostToDevice);
    float *d_edge_index = NULL;
    err = cudaMalloc((void **)&d_edge_index, size * 2 * sizeof(float));
    err = cudaMemcpy(d_edge_index, edge_index, size * 2 * sizeof(float), cudaMemcpyHostToDevice);
    float *d_edges = NULL;
    err = cudaMalloc((void **)&d_edges, tot_edges *  sizeof(float));
    err = cudaMemcpy(d_edges, edges, tot_edges *  sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    KernelPageRank<<<blocksPerGrid, threadsPerBlock>>>(d_vertices, d_edge_index, d_edges, size);*/

    int threadsPerBlock = 256;
    int blocksPerGrid   = (size + threadsPerBlock - 1) / threadsPerBlock;
    void *args[] = { &d_vertices, &d_edge_index, &d_edges, &d_edge_num , &size };
    // Launch the CUDA kernel
    error = cuLaunchKernel(pagerank_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0,
                               NULL, args, NULL);
    if (error != CUDA_SUCCESS) Cleanup(false);
    error = cuCtxSynchronize();

    error = cuMemcpyDtoH(h_vertices, d_vertices, size * sizeof(float));
    error = cuMemcpyDtoH(edges, d_edges, tot_edges * sizeof(float));
    if (error != CUDA_SUCCESS) Cleanup(false);
    error = CleanupNoFailure();
    logstream(LOG_INFO)<<"Cu Work Done.\n";
  } 

  void before_iteration(int iteration, graphchi_context &info) {
  }
/**
  * Called after an iteration has finished. Not implemented.
  */
  void after_iteration(int iteration, graphchi_context &ginfo) {
  }
/**
  * Called before an execution interval is started. Not implemented.
  */
  void before_exec_interval(vid_t window_st, vid_t window_en, graphchi_context &ginfo) {
  }
  void update(graphchi_vertex<VertexDataType, EdgeDataType> &v, graphchi_context &ginfo) {
  }
};
  
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

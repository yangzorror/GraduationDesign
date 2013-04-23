# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])
# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
  OS_SIZE = 32
  OS_ARCH = i686
endif

ifeq ($(x86_64),1)
  OS_SIZE = 64
  OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
CUDA_PATH       ?= /usr/local/cuda-5.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

# Extra user flags
EXTRA_NVCCFLAGS ?=
EXTRA_LiwDFLAGS ?=

ifneq ($(DARWIN),)
  LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -framework CUDA
  CCFLAGS   := -arch $(OS_ARCH)
else
  ifeq ($(OS_SIZE),32)
    LDFLAGS   := -L$(CUDA_LIB_PATH) -lcuda
    CCFLAGS   := -m32
  else
    LDFLAGS   := -L$(CUDA_LIB_PATH) -lcuda
    CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
  NVCCFLAGS := -m32
  PTX_FILE      := vectorAdd_kernel32.ptx
else
  NVCCFLAGS := -m64
  PTX_FILE  := vectorAdd_kernel64.ptx
  ifeq ($(dbg),1)
    CCFLAGS   += -g
    NVCCFLAGS += -g -G
    TARGET    := debug
  else
    TARGET    := release
  endif
endif

# Common includes and paths for CUDA
INCLUDES      := -I$(CUDA_INC_PATH) -I.  -I./inc


#######################################################################



INCFLAGS = -I/usr/local/include/ -I./src/

CPP = g++
CPPFLAGS = -g -ggdb $(INCFLAGS)  -fopenmp -Wall -Wno-strict-aliasing 
LINKERFLAGS = -lz
DEBUGFLAGS = -g -ggdb $(INCFLAGS)
HEADERS=$(wildcard *.h**)


#all: apps tests

all: build 
apps: example_apps/connectedcomponents example_apps/pagerank example_apps/pagerank_functional example_apps/communitydetection example_apps/trianglecounting example_apps/randomwalks
als: example_apps/matrix_factorization/als_edgefactors  example_apps/matrix_factorization/als_vertices_inmem
tests: tests/basic_smoketest tests/bulksync_functional_test tests/dynamicdata_smoketest tests/test_dynamicedata_loader

build: pagerank pagerank_kernel.ptx

pagerank_kernel.ptx: cuda_apps/pagerank.cu
	@mkdir -p cuda_bin
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -o $@ -ptx $<
	@cp $@ cuda_bin

pagerank.o: cuda_apps/pagerank.cpp $(HEADERS)
	$(CPP) $(CPPFLAGS) $(INCLUDES) -o $@ -c $<
	#cp $@ ./data

pagerank: pagerank.o $(HEADERS)
	$(CPP) $(CCFLAGS) $(CPPFLAGS) -o $@ pagerank.o $(LDFLAGS) $(EXTRA_LDFLAGS) $(LINKERFLAGS)
	@cp $@ cuda_bin
clean:
	@rm -rf bin/*
	cd toolkits/collaborative_filtering/; make clean; cd ../../
	cd toolkits/parsers/; make clean; cd ../../
	cd toolkits/graph_analytics/; make clean; cd ../../

blocksplitter: src/preprocessing/blocksplitter.cpp $(HEADERS)
	$(CPP) $(CPPFLAGS) src/preprocessing/blocksplitter.cpp -o bin/blocksplitter $(LINKERFLAGS)

sharder_basic: src/preprocessing/sharder_basic.cpp $(HEADERS)
	@mkdir -p bin
	$(CPP) $(CPPFLAGS) src/preprocessing/sharder_basic.cpp -o bin/sharder_basic $(LINKERFLAGS)

example_apps/% : example_apps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Iexample_apps/ $@.cpp -o bin/$@ $(LINKERFLAGS) 



myapps/% : myapps/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) -Imyapps/ $@.cpp -o bin/$@ $(LINKERFLAGS)

tests/%: src/tests/%.cpp $(HEADERS)
	@mkdir -p bin/$(@D)
	$(CPP) $(CPPFLAGS) src/$@.cpp -o bin/$@	$(LINKERFLAGS)


graphlab_als: example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp
	$(CPP) $(CPPFLAGS) example_apps/matrix_factorization/graphlab_gas/als_graphlab.cpp -o bin/graphlab_als $(LINKERFLAGS)

cf:
	cd toolkits/collaborative_filtering/; bash ./test_eigen.sh; 
	if [ $$? -ne 0 ]; then exit 1; fi
	cd toolkits/collaborative_filtering/; make 
cf_test:
	cd toolkits/collaborative_filtering/; make test; 
cfd:
	cd toolkits/collaborative_filtering/; make -f Makefile.debug

parsers:
	cd toolkits/parsers/; make
ga:
	cd toolkits/graph_analytics/; make

docs: */**
	doxygen conf/doxygen/doxygen.config


	

	

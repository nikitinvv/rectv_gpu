CUDA_PATH ?= "/usr/local/cuda-8.0/"

NVCC          := $(CUDA_PATH)/bin/nvcc

# internal flags
NVCCFLAGS   := -m64 -Xcompiler -fopenmp -O3
LDFLAGS     := -lcufft -lgomp


# Gencode arguments
SMS ?= 35 50 52 60 61 62 

ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

# Target rules
all: build 

build: obj rectv

obj/main.o:src/main.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
obj/rectv.o:src/rectv.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
obj/radonusfft.o:src/radonusfft.cu
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<
obj:  
	mkdir -p obj
rectv: 	obj/main.o obj/rectv.o obj/radonusfft.o
	$(NVCC) $(GENCODE_FLAGS) -o $@ $+ $(LDFLAGS)
clean:
	rm -rf rectv obj


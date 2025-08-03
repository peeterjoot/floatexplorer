CXXFLAGS += -g
CXXFLAGS += -std=c++20

OS := $(shell uname -s)
MACHINE := $(shell uname -m)

# see README for OS dependencies:
ifeq ($(OS),Darwin)
CXX := g++-15
LOADLIBES += -lquadmath
CXXFLAGS += -fext-numeric-literals
CXXFLAGS += -DUSE_QUADMATH
#CXXFLAGS += -gdwarf-4
endif

CUDA_VERSION := cuda-12.9
NVCC := /usr/local/$(CUDA_VERSION)/bin/nvcc
NVCC_EXISTS := $(wildcard $(NVCC))

ifneq ($(NVCC_EXISTS),)
CXX := $(NVCC)
CXXFLAGS += -arch=sm_86
CXXFLAGS += -I/usr/local/$(CUDA_VERSION)/include
CXXFLAGS += -DHAVE_CUDA
LDFLAGS += -L/usr/local/$(CUDA_VERSION)/lib64
LOADLIBES += -lcudart

ifeq ($(OS),Linux)
ifeq ($(MACHINE),x86_64)
LOADLIBES += -lquadmath
CXXFLAGS += -Xcompiler -fext-numeric-literals
CXXFLAGS += -DUSE_QUADMATH
endif
endif
else
ifeq ($(OS),Linux)
ifeq ($(MACHINE),x86_64)
LOADLIBES += -lquadmath
CXXFLAGS += -fext-numeric-literals
CXXFLAGS += -DUSE_QUADMATH
endif
endif
endif

all : floatexplorer

clean:
	rm -f floatexplorer
	rm -rf *.dSYM

test:
ifneq ($(NVCC_EXISTS),)
	./floatexplorer --e4m3 3 | diff -up expected/e4m3.txt -
	./floatexplorer --e5m2 3 | diff -up expected/e5m2.txt -
	./floatexplorer --spe --e4m3 | diff -up expected/e4m3.special.txt -
	./floatexplorer --spe --e5m2 | diff -up expected/e5m2.special.txt -
	./floatexplorer --spe --fp16 | diff -up expected/fp16.special.txt -
	./floatexplorer --spe --bf16 | diff -up expected/bf16.special.txt -
endif
	./floatexplorer --spe | diff -up expected/float.special.txt -
	./floatexplorer --spe --double | diff -up expected/double.special.txt -
#ifneq ($(OS),Darwin)
#	./floatexplorer --spe --f128 | diff -up - expected/f128.special.txt
#endif

# vim: noet ts=8 sw=8

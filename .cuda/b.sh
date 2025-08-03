set -x

/usr/local/cuda-12.9/bin/nvcc -g -std=c++20 -arch=sm_86 -I/usr/local/cuda-12.9/include -DHAVE_CUDA -Xcompiler -fext-numeric-literals -DUSE_QUADMATH  -L/usr/local/cuda-12.9/lib64  f.cc -lcudart -lquadmath  -o f

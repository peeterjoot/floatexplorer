## floatexplorer

This is a little bit of code to dump the representation of some floating point types, including:

- e5m2,
- e4m3,
- fp16 (e5m10)
- bf16 (e8m7),
- ieee 32-bit (C float: e8m23),
- ieee 64-bit (C double, and long double on some platforms: e11m52),
- Intel 80-bit long double (e15m64: unlike most, this one doesn't use a IEEE like representation, and has an explicit leading mantissa bit)
- IEEE 128-bit (e15m112: Linux ARM long double, GCC libquadmath)

## TODO

- See if the HIP API can also do the float type conversions as a cross check.
- Implement zArch mainframe "HEXFLOAT" types.
- Review outputs for E4M3 for:

    ./floatexplorer --e4m3 0x78
    ./floatexplorer --e4m3 0xF8
    ./floatexplorer --e4m3 0xF9
    ./floatexplorer --e4m3 0xFA
    ./floatexplorer --e4m3 0xFB
    ./floatexplorer --e4m3 0xFC
    ./floatexplorer --e4m3 0xFD
    ./floatexplorer --e4m3 0xFE
    ./floatexplorer --e4m3 0xFF

(testbin/e4m3.sh: get different results for these in CUDA vs. non-CUDA configurations.)

- Same thing for:

    ./floatexplorer --e5m2 0xFD
    ./floatexplorer --e5m2 0xFE
    ./floatexplorer --e5m2 0xFF

(testbin/e5m2.sh)

- Review outputs from:

./testbin/bf16.sh | tee expected/bf16.all.txt
./testbin/fp16.sh | tee expected/fp16.all.txt

(those were generated with CUDA conversion api -- compare to non-CUDA)

## Discussion

An early version of the code is described here:

https://peeterjoot.com/2025/04/23/a-little-float-explorer-program/

## Build
Requires C++20 for `<format>`. Run:

```bash
make
```

## Known to build on

* MacOS
* ARM Linux (fedora 42)
* x64 Linux (fedora 42)
* x64 Linux (fedora 41)
* x64 Linux (debian Ubuntu 24.04 -- WSL2)

## Dependencies:

Linux x64:

```
sudo dnf -y install make
sudo dnf -y install g++
sudo dnf -y install gcc
sudo dnf -y install libquadmath
sudo dnf -y install libquadmath-devel
```

MacOs:

```
brew install gcc
```

# CUDA dependencies

Support for GPU types (e5m2, e4m3, fp16, bf16) has been implemented.  For string <> float conversions for these types, CUDA support is used if available (rudimentary auto-detection in the makefile.)  If using Fedora, note that Fedora 42 (latest) is not currently supported by the cuda toolkit.

The fedora41 installation sequence was something like:

```
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/$(uname -m)/cuda-fedora41.repo
sudo dnf module disable nvidia-driver
sudo dnf config-manager setopt cuda-fedora41-$(uname -m).exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig
sudo dnf -y install cuda-toolkit
sudo dnf config-manager setopt cuda-fedora41-x86_64.exclude=
sudo dnf install -y nvidia-driver-cuda --refresh
```

... with some reboots in the mix.

## CUDA bugs:

Note that the CUDA 12.9 type conversion seems to misidentify some E4M3 NaNs, such as 0x7E.  Using:

     __half half = __nv_cvt_fp8_to_halfraw( s, __NV_E4M3 );
     float output = __half2float( half );

we get 448 instead of NaN (and not even 240, which would be the big-value if NaNs weren't supported.)

i.e.: 240:
```
FromDigits["1111", 2]*2^(7 - 3)
```

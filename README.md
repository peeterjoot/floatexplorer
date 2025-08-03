## floatexplorer

This is a little bit of code to dump the representation of some floating point types, including:

- e5m2 format,
- e4m3 format,
- 32-bit (C float), 
- 64-bit (C double),
- 128-bit (Linux ARM long double, GCC libquadmath)
- 80-bit Intel long double (unlike the above, this one doesn't use the IEEE representation, and has an explicit leading mantissa bit.)

fp16, bf16: TODO.

## TODO

The e5m2, e4m3, float32, and float64 print code should be merged (which would make fp16 and fp16 easier too.)

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

# CUDA dependencies (WIP.)

Support for two GPU types is implemented (e5m2, e4m3).  For string <> float conversions for these types, CUDA support is required (sort of 
auto-detected in the makefile.)  If using Fedora, note that Fedora 42 (latest) is not currently supported by the cuda toolkit.

BF16, and FP16 TODO.

```
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/$(uname -m)/cuda-fedora41.repo
sudo dnf module disable nvidia-driver
sudo dnf config-manager setopt cuda-fedora41-$(uname -m).exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig
sudo dnf -y install cuda-toolkit
sudo dnf config-manager setopt cuda-fedora41-x86_64.exclude=
sudo dnf install -y nvidia-driver-cuda --refresh
```


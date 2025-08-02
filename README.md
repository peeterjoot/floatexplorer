## floatexplorer

This is a little bit of code to dump the representation of some floating point types, including:
- 32-bit (C float), 
- 64-bit (C double),
- 128-bit (Linux ARM long double, GCC libquadmath)
- 80-bit Intel long double (unlike the above, this one doesn't use the IEEE representation, and has an explicit leading mantissa bit.)

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
* ARM Linux (fedora)
* x64 Linux (fedora)

Have not tried on debian yet (any arch), but will do so on my WSL2 instance when I get around to it.

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

# CUDA dependencies (optional:WIP.)

I'm running fedora42, but nvidia only has a fedora41 repo (at the time of writing):

```
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora41/$(uname -m)/cuda-fedora41.repo
sudo dnf module disable nvidia-driver
sudo dnf config-manager setopt cuda-fedora41-$(uname -m).exclude=nvidia-driver,nvidia-modprobe,nvidia-persistenced,nvidia-settings,nvidia-libXNVCtrl,nvidia-xconfig
sudo dnf -y install cuda-toolkit
sudo dnf -y install gcc14 gcc14-c++
```


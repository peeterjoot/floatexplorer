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
sudo dnf -y install libquadmath
sudo dnf -y install libquadmath-devel
```

MacOs:

```
brew install gcc
```

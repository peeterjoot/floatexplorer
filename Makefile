CXXFLAGS += -g
CXXFLAGS += -std=c++20

OS := $(shell uname -s)
MACHINE := $(shell uname -m)

# see README for OS dependencies:
ifeq ($(OS),Darwin)
CXX := g++-15
LDFLAGS += -lquadmath
CXXFLAGS += -fext-numeric-literals
#CXXFLAGS += -gdwarf-4
endif

ifeq ($(OS),Linux)
ifeq ($(MACHINE),x86_64)
LDFLAGS += -lquadmath
CXXFLAGS += -fext-numeric-literals
endif
endif

all : floatexplorer

clean:
	rm -f floatexplorer
	rm -rf *.dSYM

test:
	./floatexplorer --spe | diff -up - expected/float.special.txt
	./floatexplorer --spe --double | diff -up - expected/double.special.txt
#ifneq ($(OS),Darwin)
#	./floatexplorer --spe --f128 | diff -up - expected/f128.special.txt
#endif

# vim: noet ts=8 sw=8

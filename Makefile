CXXFLAGS += -g
CXXFLAGS += -std=c++20

OS := $(shell uname -s)

# requires: brew install gcc
ifeq ($(OS),Darwin)
CXX := g++-15
LDFLAGS += -lquadmath
CXXFLAGS += -fext-numeric-literals
CXXFLAGS += -gdwarf-4
endif

all : floatexplorer

clean:
	rm -f floatexplorer
	rm -rf *.dSYM

test:
	./floatexplorer --spe | diff -up - expected/float.special.txt
	./floatexplorer --spe --double | diff -up - expected/double.special.txt
	./floatexplorer --spe --longdouble | diff -up - expected/longdouble.special.txt

# vim: noet ts=8 sw=8

CXXFLAGS += -g
CXXFLAGS += -std=c++20

OS := $(shell uname -s)

# requires: brew install gcc
ifeq ($(OS),Darwin)
CXX := g++-15
LDFLAGS += -lquadmath
CXXFLAGS += -fext-numeric-literals
endif

all : floatexplorer

clean:
	rm -f floatexplorer

test:
	./floatexplorer --spe | diff -up - expected/float.special.txt
	./floatexplorer --spe --double | diff -up - expected/double.special.txt

# vim: noet ts=8 sw=8

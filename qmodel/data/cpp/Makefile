CC = g++
CFLAGS = -O3 -Wall -shared -std=c++11 -fPIC -fopenmp
INCLUDES = $(shell python3 -m pybind11 --includes)
SUFFIX = $(shell python3-config --extension-suffix)

all: permute${SUFFIX}

permute${SUFFIX}: permute.cpp permute.pyi
	${CC} ${CFLAGS} ${INCLUDES} permute.cpp -o permute${SUFFIX}
	python3 -m doctest permute.pyi

clean:
	rm *.so

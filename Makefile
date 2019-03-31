VEC_INFO=VecInfoAVX2
ARCH=haswell

CXXFLAGS=-std=c++14 -O3 -march=$(ARCH) -DVEC_INFO=$(VEC_INFO) -Wall -Wextra

miroslav: main.cpp miroslav.h
	$(CXX) $(CXXFLAGS) -o miroslav main.cpp

ROOTlibs=$(shell root-config --libs) -lUnfold
ROOTflags=$(shell root-config --cflags)

fastUnfold: fastUnfold.cc
	g++ -g -O2 $(ROOTflags) -I/usr/include/mkl/  -I/usr/include/eigen3/ -o $@  $^  $(ROOTlibs)  -L${MKLROOT}/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl

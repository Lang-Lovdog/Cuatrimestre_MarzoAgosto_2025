CXX=g++.exe -Wall
#OCV_I1PATH=/opencv/build/install/include
#OCV_I2PATH=$(OCV_I1PATH)/opencv2
#OCV_lPATH=/opencv/build/install/x64/mingw/bin
#CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH)
#CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
CXXFLAGS= -lm -std=c++11
SRCDIR=src
BUILDDIR=build
BINDIR=$(BUILDDIR)/bin
CXXFILES=$(wildcard $(SRCDIR)/*.cxx)

genetics:
	cd $(BUILDDIR) && $(CXX) $(subst $(SRCDIR),../$(SRCDIR),$(CXXFILES)) $(CXXFLAGS) ../$(SRCDIR)/main.cpp -c
	$(CXX) $(wildcard $(BUILDDIR)/*.o) -o $(BINDIR)/Genetics


CXX=g++

ifeq ($(OS),Windows_NT)
OCV_I1PATH=/opencv/build/install/include
OCV_I2PATH=$(OCV_I1PATH)/opencv2
OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH)
CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
else
CXXFLAGS = `pkg-config --cflags --libs opencv4`
endif

SRCDIR=src
BUILDDIR=build
BINDIR=$(BUILDDIR)/bin
CXXFILES=$(wildcard $(SRCDIR)/*.cxx)

analisis:
#	cd $(BUILDDIR) && $(CXX) ../$(SRCDIR)/OCV_Course-main.cpp $(subst $(SRCDIR),../$(SRCDIR),$(CXXFILES)) $(CXXFLAGS) -c
#	$(CXX) $(wildcard $(BULDDIR)/*.o) $(CXXLINKS) -o $(BINDIR)/OCVCourse
	$(CXX) $(SRCDIR)/OCV_main.cpp $(CXXFILES)  $(CXXLINKS) $(CXXFLAGS) -o $(BINDIR)/Tarea02

takahiro:
	$(CXX) $(SRCDIR)/takahiro_colortransfer.cpp $(CXXLINKS) $(CXXFLAGS) -o $(BINDIR)/Tarea02

CXX=g++.exe
OCV_I1PATH=/opencv/build/install/include
OCV_I2PATH=$(OCV_I1PATH)/opencv2
OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH)
CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
SRCDIR=src
BUILDDIR=build
BINDIR=$(BUILDDIR)/bin
CXXFILES=$(SRCDIR)/dibujo.cxx $(SRCDIR)/main.cpp

analisis:
#	cd $(BUILDDIR) && $(CXX) ../$(SRCDIR)/OCV_Course-main.cpp $(subst $(SRCDIR),../$(SRCDIR),$(CXXFILES)) $(CXXFLAGS) -c
#	$(CXX) $(wildcard $(BULDDIR)/*.o) $(CXXLINKS) -o $(BINDIR)/OCVCourse
	$(CXX) $(CXXFILES) $(CXXLINKS) $(CXXFLAGS) -o $(BINDIR)/dibujo
	$(BINDIR)/dibujo imagen.png

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
CXXFILES=$(SRCDIR)/dibujo.cxx $(SRCDIR)/main.cpp

analisis:
#	cd $(BUILDDIR) && $(CXX) ../$(SRCDIR)/OCV_Course-main.cpp $(subst $(SRCDIR),../$(SRCDIR),$(CXXFILES)) $(CXXFLAGS) -c
#	$(CXX) $(wildcard $(BULDDIR)/*.o) $(CXXLINKS) -o $(BINDIR)/OCVCourse
	$(CXX) $(CXXFILES) $(CXXLINKS) $(CXXFLAGS) -o $(BINDIR)/dibujo

run:
	cd res; \
	for img in *.png ; do \
		../$(BINDIR)/dibujo "$$img" ; \
	done

	cd res; \
	for img in *.jpg ; do \
		../$(BINDIR)/dibujo "$$img" ; \
	done

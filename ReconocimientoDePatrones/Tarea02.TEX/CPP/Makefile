CXX=g++
ifeq ($(OS),Windows_NT)
###########################################################
## WINDOWS SECTION EVERY NAME HERE WILL USE WIN_ PREFIX
WIN_OCV_IPATH=-I/opencv/build/install/include
WIN_OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=$(WIN_OCV_IPATH) $(WIN_OCV_IPATH)/opencv2 $(WIN_OCV_IPATH)/opencv2/core
CXXLINKS=$(wildcard $(WIN_OCV_lPATH)/*.dll)
###########################################################
## Linux Section Every Name Here Will Use Linux_ Prefix
else
CXXFLAGS=`pkg-config --cflags opencv4`
CXXLINKS=`pkg-config --libs opencv4`
endif
CXXFLAGS+= -g -Wall
SRCDIR=src
BUILDDIR=build
BINDIR=$(BUILDDIR)/bin
CXXFILES=$(wildcard $(SRCDIR)/*.cxx)

MAIN_GLCM=$(SRCDIR)/MatrizCaracteristicas_main.cpp
MAIN_CROP=$(SRCDIR)/Crop_main.cpp

BINTARGET_GLCM=$(BINDIR)/matcar
BINTARGET_CROP=$(BINDIR)/matcrop

build: buildGLCM buildCrop

buildCrop:
	$(CXX) $(CXXFILES) $(MAIN_CROP) $(CXXLINKS) $(CXXFLAGS) -o $(BINTARGET_CROP)

buildGLCM:
	$(CXX) $(CXXFILES) $(MAIN_GLCM) $(CXXLINKS) $(CXXFLAGS) -o $(BINTARGET_GLCM)

.PHONY: build

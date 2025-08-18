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

MAIN_SDH=$(SRCDIR)/SDHmain.cpp

BINTARGET_SDH=$(BINDIR)/matsdh

build:
	$(CXX) $(CXXFILES) $(MAIN_GLCM) $(CXXLINKS) $(CXXFLAGS) -o $(BINTARGET_SDH)

.PHONY: build

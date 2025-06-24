CXX=g++
###########################################################
## WINDOWS SECTION EVERY NAME HERE WILL USE WIN_ PREFIX
WIN_OCV_IPATH=-I/opencv/build/install/include
WIN_OCV_lPATH=/opencv/build/install/x64/mingw/bin
WIN_CXXFLAGS=$(WIN_OCV_IPATH) $(WIN_OCV_IPATH)/opencv2 $(WIN_OCV_IPATH)/opencv2/core
WIN_CXXLINKS=$(wildcard $(WIN_OCV_lPATH)/*.dll)
###########################################################
SRCDIR=src
BUILDDIR=build
BINDIR=$(BUILDDIR)/bin
CXXFILES=$(wildcard $(SRCDIR)/*.cxx) $(wildcard $(SRCDIR)/*.cpp)
BINTARGET=$(BINDIR)/matcar

windowsMain:
	echo $(WIN_CXXFLAGS)
	$(CXX) $(CXXFILES) $(WIN_CXXLINKS) $(WIN_CXXFLAGS) -o $(BINTARGET)

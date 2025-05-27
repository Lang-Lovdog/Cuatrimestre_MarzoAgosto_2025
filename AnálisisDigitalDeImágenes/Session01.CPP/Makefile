CXX=g++.exe
OCV_I1PATH=/opencv/build/install/include
OCV_I2PATH=$(OCV_I1PATH)/opencv2
OCV_lPATH=/opencv/build/install/x64/mingw/bin
CXXFLAGS=-I$(OCV_I1PATH) -I$(OCV_I2PATH)
CXXLINKS=$(wildcard $(OCV_lPATH)/*.dll)
CXXFILES=$(wildcard ./*.cxx)

build:
	$(CXX) OCV_Course.cpp $(CXXFILES) $(CXXLINKS) $(CXXFLAGS) -o OCVCourse
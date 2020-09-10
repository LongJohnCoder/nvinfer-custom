################################################################################
# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#################################################################################

CUDA_VER=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX:= g++
SRCS:= gstnvinfer.cpp  gstnvinfer_allocator.cpp gstnvinfer_property_parser.cpp \
       gstnvinfer_meta_utils.cpp gstnvinfer_impl.cpp aligner.cpp
INCS:= $(wildcard *.h)
LIB:=libnvdsgst_infercustom.so

NVDS_VERSION:=5.0

CFLAGS+= -fPIC -std=c++11 -DDS_VERSION=\"5.0.0\" \
	 -I /usr/local/cuda-$(CUDA_VER)/include \
	 -I /usr/include/gstreamer-1.0 \
	 -I /opt/nvidia/deepstream/deepstream-5.0/sources/includes \
     -I /usr/include/glib-2.0 \
	 -I ./includes \
	 -I ./libs/nvdsinfer -DNDEBUG

GST_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/gst-plugins/
LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

LIBS := -shared -Wl,-no-undefined \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_helper -lnvdsgst_meta -lnvds_meta \
       -lnvds_infer -lnvbufsurface -lnvbufsurftransform -ldl -lpthread \
       -Wl,-rpath,$(LIB_INSTALL_DIR)

OBJS:= $(SRCS:.cpp=.o)

PKGS:= gstreamer-1.0 gstreamer-base-1.0 gstreamer-video-1.0 opencv
CFLAGS+=$(shell pkg-config --cflags $(PKGS))
LIBS+=$(shell pkg-config --libs $(PKGS))

ifeq ($(shell uname -m), aarch64)
CFLAGS+=-DIS_TEGRA
LIBS+=-lcuda
endif

all: $(LIB)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CFLAGS) $<

$(LIB): $(OBJS) $(DEP) Makefile
	$(CXX) -o $@ $(OBJS) $(LIBS)

install: $(LIB)
	cp -rv $(LIB) $(GST_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(LIB)

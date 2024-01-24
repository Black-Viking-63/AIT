FROM ubuntu:22.04
WORKDIR /

# Update and upgrade
RUN apt update && apt -y upgrade

# Create dir
RUN mkdir /usr/local/Dev

# Python 3
RUN apt install -y curl python3-testresources python3-dev wget gnupg2 software-properties-common
WORKDIR /usr/local/Dev/
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# OpenCV x.x.x with non free modules
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections

# To prevent interactive configuration of tzdata
ENV TZ=Europe/Samara
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

## GStreamer

RUN apt -y install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
RUN apt -y install ubuntu-restricted-extras libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev libgstreamer-plugins-bad1.0-0 libgstreamer-plugins-base1.0-0 libgstreamer-plugins-base1.0-dev

## OpenCV build dependencies

RUN apt -y install build-essential cmake unzip git pkg-config libgtk2.0-dev libavcodec-dev libavformat-dev \
                   libswscale-dev libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libv4l-dev \
                   libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran \
                   python3-dev python3-pip python3-numpy

## OpenCV

RUN apt update
RUN apt install python3-opencv

RUN echo $(python3 -c "import cv2 as cv; print(cv.__version__)")

### Update numpy

RUN pip3 install -U numpy
RUN pip3 install numba
RUN pip3 install Pillow

CMD ["/bin/bash"]
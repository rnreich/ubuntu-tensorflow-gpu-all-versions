## How to really install tensorflow-gpu from source on a clean Ubuntu

**level**: beginners

the correct versions of the components must be compatible with each other, and they have to be installed in the correct order. i chose versions of the components that were released around the same period of time. also, tensorflow-gpu should be built from source.

**os**: Ubuntu - all versions (anything above 16.0.0 - main releases only)

**gpu type**: nvidia

**python version**: 3

**components**: cuda toolkit 9.0, cudnn 7, nccl 2.3.7, bazel 0.8.0, gcc 6, g++ 6, tensorflow r1.5 sources

### install a new Ubuntu instance and nvidia drivers

sudo apt-get update && sudo apt-get upgrade

**sudo ubuntu-drivers autoinstall**

sudo reboot

sudo apt-get update && sudo apt-get upgrade

### install cuda toolkit 9.0 - Linux, x86_64, Ubuntu, 17.04, deb (local)

https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal (1.2 GB)

sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub

sudo dpkg -i cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb

### install cudnn 7.0 - cuDNN v7.4.2 (Dec 14, 2018), for CUDA 9.0:

*runtime*:

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.2/prod/9.0_20181213/Ubuntu16_04-x64/libcudnn7_7.4.2.24-1%2Bcuda9.0_amd64.deb (121 MB)

sudo apt-get update && sudo apt-get upgrade

**sudo dpkg -i libcudnn7_7.4.2.24-1+cuda9.0_amd64.deb**

*developer*:

https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.4.2/prod/9.0_20181213/Ubuntu16_04-x64/libcudnn7-dev_7.4.2.24-1%2Bcuda9.0_amd64.deb (112 MB)

sudo apt-get update && sudo apt-get upgrade

**sudo dpkg -i libcudnn7-dev_7.4.2.24-1+cuda9.0_amd64.deb**

sudo apt-get install aptitude

sudo aptitude install cuda

*accept all aptitude solutions*

sudo apt-get install libcupti-dev

export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

### install nccl - NCCL v2.3.7, for CUDA 9.0, Nov 8 & Dec 14, 2018:

https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.3/prod3/nccl-repo-ubuntu1604-2.3.7-ga-cuda9.0_1-1_amd64.deb (51.8 MB)

sudo apt-get update && sudo apt-get upgrade

**sudo dpkg -i nccl-repo-ubuntu1604-2.3.7-ga-cuda9.0_1-1_amd64.deb**

sudo apt-get update && sudo apt-get upgrade

**sudo apt install libnccl2 libnccl-dev**

### install bazel 0.8.0: https://github.com/bazelbuild/bazel/releases/download/0.8.0/bazel-0.8.0-installer-linux-x86_64.sh (158 MB):**

sudo apt-get update && sudo apt-get upgrade

sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python

sudo chmod +x bazel-0.8.0-installer-linux-x86_64.sh

sudo ./bazel-0.8.0-installer-linux-x86_64.sh

export PATH="$PATH:/usr/local/bin"

### install gcc 6:

sudo apt-get update && sudo apt-get upgrade

**sudo apt-get install gcc-6 g++-6**

### build tensorflow-gpu from source:

sudo apt-get update && sudo apt-get upgrade

**sudo apt-get install python3-dev python3-pip git**

**git clone https://github.com/tensorflow/tensorflow.git**

cd tensorflow

git checkout r1.5

bazel shutdown

bazel clean

git clean -xdf

./configure

- Please specify the location of python. [Default is /usr/bin/python]: **/usr/bin/python3**
- Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]
- Do you wish to build TensorFlow with jemalloc as malloc support? [Y/n]: n
- Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
- Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
- Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
- Do you wish to build TensorFlow with XLA JIT support? [y/N]: N
- Do you wish to build TensorFlow with GDR support? [y/N]: N
- Do you wish to build TensorFlow with VERBS support? [y/N]: N
- Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
- Do you wish to build TensorFlow with CUDA support? [y/N]: N
- Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 
- Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
- Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 
- Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
- Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 5.0]
- Do you want to use clang as CUDA compiler? [y/N]: N
- Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: **/usr/bin/gcc-6**
- Do you wish to build TensorFlow with MPI support? [y/N]: N
- Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 

**sudo bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package --verbose_failures**

INFO: Build completed successfully, 5984 total actions

### build the package:

*"The bazel build command creates an executable named build_pip_package—this is the program that builds the pip package. For example, the following builds a .whl package in the /tmp/tensorflow_pkg directory:"*

sudo ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

Fri Jan 11 06:51:40 IST 2019 : === Output wheel file is in: /tmp/tensorflow_pkg

ls /tmp/tensorflow_pkg

*tensorflow-1.5.1-cp36-cp36m-linux_x86_64.whl*

### install the package:

sudo python3 -m pip uninstall pip && sudo apt install python3-pip --reinstall

sudo pip3 install /tmp/tensorflow_pkg/*tensorflow-1.5.1-cp36-cp36m-linux_x86_64*.whl

### verify installation:

cd ..

python3

import tensorflow

quit()

*if everything is ok you should see nothing. type quit()*

### install useful libraries

sudo pip3 install pandas keras

### make a snapshot or image of the machine

### *fast route - installing compiled packages rather than building from source:*

sudo pip3 install --upgrade tensorflow-gpu

sudo pip3 uninstall tensorflow-gpu

sudo pip3 install tensorflow-gpu==0.0

0.12.1, 1.0.0, 1.0.1, 1.1.0rc1, 1.1.0rc2, 1.1.0, 1.2.0rc0, 1.2.0rc1, 1.2.0rc2, 1.2.0, 1.2.1, 1.3.0rc0, 1.3.0rc1, 1.3.0rc2, 1.3.0, 1.4.0rc0, 1.4.0rc1, 1.4.0, 1.4.1, 1.5.0rc0, 1.5.0rc1, 1.5.0, 1.5.1, 1.6.0rc0, 1.6.0rc1, 1.6.0, 1.7.0rc0, 1.7.0rc1, 1.7.0, 1.7.1, 1.8.0rc0, 1.8.0rc1, 1.8.0, 1.9.0rc0, 1.9.0rc1, 1.9.0rc2, 1.9.0, 1.10.0rc0, 1.10.0rc1, 1.10.0, 1.10.1, 1.11.0rc0, 1.11.0rc1, 1.11.0rc2, 1.11.0, 1.12.0rc0, 1.12.0rc1, 1.12.0rc2, 1.12.0

### if this saved you:

![Seahorse](https://images.pexels.com/photos/221420/pexels-photo-221420.jpeg?auto=compress&cs=tinysrgb&h=750&w=1260)

1KJ1SNVoMT9zXRJxrRC8y8awuxRfmUd3Vb

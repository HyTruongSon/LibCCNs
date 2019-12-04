# g++ -std=c++11 -shared ccn1d_lib.cc -o ccn1d_lib.so -fPIC -I/home/hytruongson/.local/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/hytruongson/.local/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -O2

CC=g++
# PYTHON=python
PYTHON=python3

TF_CFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

$CC -std=c++11 -shared ccn1d_lib.cc -o ccn1d_lib.so -fPIC $TF_CFLAGS $TF_LFLAGS -O2
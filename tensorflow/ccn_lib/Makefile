CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_INC=-I$(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

ccn1d_lib.so:
	g++ -std=c++11 -shared ccn1d_lib.cc -o ccn1d_lib.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 $(TF_INC)

all: ccn1d_lib.so

clean:
	rm -rf ccn1d_lib.so

anew: clean all
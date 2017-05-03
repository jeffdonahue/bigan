# export THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32
# CNMem enabled:
export THEANO_FLAGS=mode=FAST_RUN,device=gpu
export THEANO_FLAGS=${THEANO_FLAGS},dnn.enabled=True
# export THEANO_FLAGS=${THEANO_FLAGS},lib.cnmem=1
# export THEANO_FLAGS=${THEANO_FLAGS},lib.cnmem=0.1
export THEANO_FLAGS=${THEANO_FLAGS},lib.cnmem=0.45

export THEANO_FLAGS=${THEANO_FLAGS},floatX=float32
# export THEANO_FLAGS=${THEANO_FLAGS},floatX=float16

# export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic
# export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_fwd=small,dnn.conv.algo_bwd_filter=none,dnn.conv.algo_bwd_data=none
# export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_fwd=guess_once,dnn.conv.algo_bwd_filter=guess_once,dnn.conv.algo_bwd_data=guess_once
# export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_fwd=guess_on_shape_change,dnn.conv.algo_bwd_filter=guess_on_shape_change,dnn.conv.algo_bwd_data=guess_on_shape_change
export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_fwd=time_once,dnn.conv.algo_bwd_filter=time_once,dnn.conv.algo_bwd_data=time_once
# export THEANO_FLAGS=${THEANO_FLAGS},dnn.conv.algo_fwd=time_on_shape_change,dnn.conv.algo_bwd_filter=time_on_shape_change,dnn.conv.algo_bwd_data=time_on_shape_change

# debugging mode (1st is less extreme, 2nd is if 1st doesn't work)
# export THEANO_FLAGS=${THEANO_FLAGS},optimizer=fast_compile,exception_verbosity=high
# export THEANO_FLAGS=${THEANO_FLAGS},optimizer=None,exception_verbosity=high

# suggestions from http://deeplearning.net/software/theano/faq.html
# export THEANO_FLAGS=${THEANO_FLAGS},gcc.cxxflags="-O3 -ffast-math -ftree-loop-distribution -funroll-loops -ftracer"
# export THEANO_FLAGS=${THEANO_FLAGS},optimizer_including=local_remove_all_assert
# export THEANO_FLAGS=${THEANO_FLAGS},optimizer_excluding=low_memory

# debugging NaNs
# export THEANO_FLAGS=${THEANO_FLAGS},mode=DebugMode,DebugMode.check_py=False
# export THEANO_FLAGS=${THEANO_FLAGS},mode=DebugMode,DebugMode.check_py=False,tensor.cmp_sloppy=1
# export THEANO_FLAGS=${THEANO_FLAGS},mode=NanGuardMode,NanGuardMode.nan_is_error=True

# CUDNN=/home/jdonahue/cudnn3/cuda
# CUDNN=/home/jdonahue/cudnn4/cuda
CUDNN=/home/jdonahue/cudnn5/cuda
export CPATH=${CUDNN}/include
export LD_LIBRARY_PATH=${CUDNN}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LD_LIBRARY_PATH
export PATH=${PATH}:/usr/local/cuda/bin

if [ -z "${CUDA_HOME}" ]; then
  echo "CUDA_HOME is not defined!"
  exit 1
else
  echo "CUDA_HOME is defined as ${CUDA_HOME}"
fi
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
python -m pip install -e . --no-build-isolation --no-deps -v \
	  --config-settings editable_mode=compat \
	    --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES="$ARCHS"

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# These policies are related to SWIG python libraries naming
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
find_package(SWIG REQUIRED)
include(${SWIG_USE_FILE})

# SWIG requires the Python header
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_PATH})
# TODO: Is NumPy required?
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})

find_package(OpenMP)

# Must tell CMake that SWIG interface files are CXX files
set_source_files_properties(rectv.i PROPERTIES CPLUSPLUS ON)
swig_add_library(rectv LANGUAGE python SOURCES rectv.i rectv.cu radonusfft.cu admm.cu operators.cu)
swig_link_libraries(rectv cudart cufft cublas OpenMP::OpenMP_CXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -fopenmp")

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/rectv.py" DESTINATION rectv_gpu)
install(TARGETS rectv LIBRARY DESTINATION rectv_gpu)
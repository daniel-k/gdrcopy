cmake_minimum_required(VERSION 3.7)

project(gdrcopy
	VERSION 1.0
	DESCRIPTION "gdrcopy"
	LANGUAGES C CXX)

# fail if CUDA not found
find_package(CUDA QUIET REQUIRED)

add_library(gdrapi SHARED
	gdrapi.c
	memcpy_avx.c
	memcpy_sse.c
	memcpy_sse41.c)

set_source_files_properties(memcpy_avx.c PROPERTIES COMPILE_FLAGS -mavx)
set_source_files_properties(memcpy_sse.c PROPERTIES COMPILE_FLAGS -msse)
set_source_files_properties(memcpy_sse41.c PROPERTIES COMPILE_FLAGS -msse4.1)

target_include_directories(gdrapi
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gdrdrv)


cuda_add_executable(basic basic.cpp)
target_link_libraries(basic gdrapi cuda)

cuda_add_executable(validate validate.cpp)
target_link_libraries(validate gdrapi)

cuda_add_executable(copybw copybw.cpp)
target_link_libraries(copybw gdrapi cuda)

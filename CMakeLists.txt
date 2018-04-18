cmake_minimum_required(VERSION 3.8)

project(gdrcopy
	VERSION 1.0
	DESCRIPTION "gdrcopy"
	LANGUAGES C CXX CUDA)

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


add_executable(basic basic.cpp)
set_source_files_properties(basic.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(basic gdrapi cuda)

add_executable(validate validate.cpp)
set_source_files_properties(validate.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(validate gdrapi cuda)

add_executable(copybw copybw.cpp)
set_source_files_properties(copybw.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(copybw gdrapi cuda)

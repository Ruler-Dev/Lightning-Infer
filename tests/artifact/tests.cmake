infernix_add_test(infernix_artifact_reader_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reader.cpp"
  LIBRARIES infernix_artifact)

infernix_add_test(infernix_artifact_materialization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materialization.cpp" "${CMAKE_CURRENT_LIST_DIR}/materialization_cuda_errors.cpp"
  LIBRARIES infernix_artifact)

if(NOT WIN32)
target_link_options(infernix_artifact_materialization_test PRIVATE
  "LINKER:--wrap=cudaMalloc"
  "LINKER:--wrap=cudaMallocHost"
  "LINKER:--wrap=cudaFree"
  "LINKER:--wrap=cudaFreeHost"
  "LINKER:--wrap=cudaEventCreateWithFlags"
  "LINKER:--wrap=cudaEventRecord"
  "LINKER:--wrap=cudaMemcpyAsync"
  "LINKER:--wrap=cudaStreamSynchronize")
endif()

add_test(NAME infernix_artifact_writer_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/writer_interop.py"
    $<TARGET_FILE:infernix_artifact_materialization_test>)

set_tests_properties(
  infernix_artifact_writer_interop_test
  PROPERTIES SKIP_RETURN_CODE 77)

set_tests_properties(
  infernix_artifact_materialization_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Flash-Next expert banks and block-scaled FP8 written by Python and accepted by the C++ reader.
add_test(NAME infernix_artifact_flash_next_interop_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/flash_next_interop.py"
    $<TARGET_FILE:infernix_artifact_reader_test>)

# Device segments (layer split): Binder segment tags and per-segment layout, then per-segment
# device arenas and pinned chunks on two contexts of device 0 and, with two GPUs, on devices 0 and 1.
infernix_add_test(infernix_artifact_materializer_segments_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_materializer_segments.cpp"
  LIBRARIES infernix_artifact)
set_tests_properties(infernix_artifact_materializer_segments_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu")

# The same on the real artifact (INFERNIX_QWEN4_ARTIFACT): every dense object loaded as segments
# {24, 48} on two contexts of device 0 equals a one-segment load byte for byte.
add_test(NAME infernix_artifact_materializer_segments_real_test
  COMMAND infernix_artifact_materializer_segments_test --real)
set_tests_properties(infernix_artifact_materializer_segments_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600)

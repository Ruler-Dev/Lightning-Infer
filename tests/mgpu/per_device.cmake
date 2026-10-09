# Per-device CUDA host state (src/core/per_device.h): PerDevice, once_per_device and the numerics
# SM count, host side only (no kernel, no device allocation), on every visible device. One entry
# per INFERNIX_NUMERICS_SMS mode: unset, 0, pinned, malformed and above the devices' SM count.
infernix_add_test(infernix_per_device_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_per_device.cpp"
  LIBRARIES infernix_core
  ARGS --unpinned)
set_tests_properties(infernix_per_device_test PROPERTIES
  SKIP_RETURN_CODE 77
  ENVIRONMENT_MODIFICATION "INFERNIX_NUMERICS_SMS=unset:")
set(per_device_modes zero pinned malformed excessive)
set(per_device_values 0 1 7O 100000)
set(per_device_expectations --unpinned --pinned --invalid --invalid)
foreach(mode value expectation IN ZIP_LISTS per_device_modes per_device_values per_device_expectations)
  if(expectation STREQUAL "--pinned")
    add_test(NAME infernix_per_device_${mode}_test COMMAND infernix_per_device_test --pinned ${value})
  else()
    add_test(NAME infernix_per_device_${mode}_test COMMAND infernix_per_device_test ${expectation})
  endif()
  set_tests_properties(infernix_per_device_${mode}_test PROPERTIES
    SKIP_RETURN_CODE 77
    ENVIRONMENT "INFERNIX_NUMERICS_SMS=${value}")
endforeach()

# The Ops of a layer split run on device a, device b and device a again, bit for bit, with
# INFERNIX_NUMERICS_SMS pinned (to the smaller SM count unless set); skipped with fewer than two
# GPUs (--devices 0,0 runs two contexts on one). The streams entry needs one GPU: eager TMA
# launches of two streams of one device, raced, against their isolated references.
infernix_add_op_test(infernix_multidevice_ops_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_multidevice_ops.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_multidevice_ops_test PROPERTIES TIMEOUT 900 RUN_SERIAL TRUE)
add_test(NAME infernix_multidevice_ops_streams_test
  COMMAND infernix_multidevice_ops_test --streams)
set_tests_properties(infernix_multidevice_ops_streams_test PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 600 RUN_SERIAL TRUE)

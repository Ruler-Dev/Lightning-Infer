# Multi-device core primitives (multi-GPU WP4). The device guard: nested guards, two contexts of one
# device, and DeviceContext::free_bytes, the VRAM budget source and the link probe reading the
# device they name with another one current (INFERNIX_CUDA_SYNC unset: both contexts spin).
infernix_add_test(infernix_device_guard_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_device_guard.cpp"
  LIBRARIES infernix_core)
set_tests_properties(infernix_device_guard_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu"
  ENVIRONMENT_MODIFICATION "INFERNIX_CUDA_SYNC=unset:")

# The handoffs: HandoffChannel rounds in Event and Gate mode between graphs of two streams (100,000
# Gate rounds of 16 B to 1.3 MiB), the bounded gate's timeout, BulkHandoff of 80 MiB; with two GPUs
# also across them in both directions, and one-way latency histograms of both modes.
infernix_add_test(infernix_handoff_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_handoff.cu"
  LIBRARIES infernix_core)
set_tests_properties(infernix_handoff_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu"
  TIMEOUT 1800
  RUN_SERIAL TRUE)

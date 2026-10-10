# The Engine's device set (multi-GPU WP5): CUDA_DEVICE_ORDER=PCI_BUS_ID selected before the first
# CUDA call unless already set, and the contexts of --devices created in order (one entry: one
# context and no log; {0,0}: two contexts of one GPU; {1,0} with two GPUs), all with the spin
# schedule. Both variables are unset for the run. Also the layer split's segments from
# --layer-split K and the layer count (layer_split_ends), checked without a device.
infernix_add_test(infernix_engine_devices_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_devices.cpp"
  LIBRARIES infernix_runtime_support infernix_core)
set_tests_properties(infernix_engine_devices_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu"
  ENVIRONMENT_MODIFICATION "INFERNIX_CUDA_SYNC=unset:;CUDA_DEVICE_ORDER=unset:")

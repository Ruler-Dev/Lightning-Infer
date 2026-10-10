# The CPU miss service's ports (multi-GPU WP9): one service thread and worker team answering the
# moe_experts calls of several streams or devices, one port each. The default entry checks the
# options and runs two streams of one GPU on two ports; the two-GPU entry puts port 0 on device a
# and port 1 on device b (--devices a,b; skipped with fewer than two GPUs), with layer-split rounds
# chained across the devices and with both devices at once. Every output equals the CPU engine's
# bit for bit, every published request is answered and every port's heartbeat advances.
infernix_add_op_test(infernix_miss_service_ports_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_miss_service_ports.cu"
  LIBRARIES infernix_ops)
set_tests_properties(infernix_miss_service_ports_test PROPERTIES
  LABELS "gpu"
  TIMEOUT 600
  RUN_SERIAL TRUE)
add_test(NAME infernix_miss_service_ports_two_gpu_test
  COMMAND infernix_miss_service_ports_test --two-gpus)
set_tests_properties(infernix_miss_service_ports_two_gpu_test PROPERTIES
  SKIP_RETURN_CODE 77
  LABELS "gpu"
  TIMEOUT 600
  RUN_SERIAL TRUE)

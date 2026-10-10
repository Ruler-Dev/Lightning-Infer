# Segment-aware load bindings and per-segment Parameters (multi-GPU WP7) on the real artifact
# (INFERNIX_QWEN4_ARTIFACT): the load plan of a split at layer 24 places every dense weight in its
# segment's arena (the head, MTP and proposal head in the last, the Vision tower in the first) and
# keeps the pinned space; materialized on two contexts of device 0, each segment's Parameters hold
# exactly its layers (and the head on the last), their operands in its arena and their expert scales
# in its own buffer. Skips without the artifact; about 7 GiB of device 0 and the pinned embedding.
infernix_add_test(infernix_qwen4_exp_load_segments_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_load_segments.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_load_segments_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 1800)
# The same with segment 1 on device 1 (materialize_model on two GPUs, each segment's Parameters
# built under its device); skips with fewer than two GPUs. The same 7 GiB, split between them.
add_test(NAME infernix_qwen4_exp_load_segments_two_gpu_test
  COMMAND infernix_qwen4_exp_load_segments_test --second-device 1)
set_tests_properties(infernix_qwen4_exp_load_segments_two_gpu_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 1800)

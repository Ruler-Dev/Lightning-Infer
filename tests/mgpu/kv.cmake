# Layer-split KV (multi-GPU WP10): a segment's page and window geometry over a subset of the
# attention layers and its global plane order (host only), then block tables published by page
# index to one table pool per segment, on two streams of one GPU and on two GPUs when present
# (skips after the geometry checks without a GPU).
infernix_add_test(infernix_qwen4_exp_kv_segments_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_segments.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_kv_segments_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

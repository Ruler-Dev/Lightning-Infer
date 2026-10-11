# The expert residency of a layer-split segment (multi-GPU WP11): a residency over the banks of
# layers 24-47 of a synthetic 48-layer model serves 24 local layers, promotes from banks[local] and
# warm-starts from its part of a whole-model saved state; a trickle (a slow link's segment) promotes
# in a rotating window of layers and defers its copies until issue_deferred; with two GPUs, a
# residency for device 1 constructed while device 0 is current reserves device 1's whole memory and
# allocates there. The saved state's split and merge are host-only checks in
# infernix_qwen4_exp_expert_cache_test.
infernix_add_test(infernix_qwen4_exp_expert_residency_split_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_expert_residency_split.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_qwen4_exp_expert_residency_split_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

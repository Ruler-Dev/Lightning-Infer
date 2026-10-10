# A layer split through the Engine on the real artifact (multi-GPU WP13, WP14): split at K = 1, 2,
# 24 and 47 on two contexts of device 0, each prompt's prefill calls, the first token, forced
# tokens and every decode and verification round give the unsplit Engine's logits bit for bit
# (FNV-1a per call or round) and its ids, for prompts of 1, 255, 256 and 4113 tokens, a 48-token
# decode with MTP, forced tokens and a cancelled prompt. Skips without INFERNIX_QWEN4_ARTIFACT;
# five Engines in turn, each pinning the expert banks (about 64 GiB of Host RAM).
infernix_add_test(infernix_qwen4_exp_split_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_split_prefill_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime infernix_core)
set_tests_properties(infernix_qwen4_exp_split_prefill_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600)
# The same with plain decode rounds (no drafter) at K = 2 and 24 (the PLE on segment 0).
add_test(NAME infernix_qwen4_exp_split_plain_real_test
  COMMAND infernix_qwen4_exp_split_prefill_real_test --plain --splits 2,24 --prompts 1,255 --no-cancel)
set_tests_properties(infernix_qwen4_exp_split_plain_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600)
# Two lanes, two requests decoding together (C = 2 pair rounds with MTP) at K = 24 on device 0.
add_test(NAME infernix_qwen4_exp_split_pair_real_test
  COMMAND infernix_qwen4_exp_split_prefill_real_test --pair --splits 24 --prompts 1 --no-forced --no-cancel)
set_tests_properties(infernix_qwen4_exp_split_pair_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600)
# The same across two GPUs (segment 0 on device 0, segment 1 on device 1) at K = 24, against the
# unsplit Engine on device 0, both with the launch decompositions of a 70-SM GPU (the smaller of
# this machine's two; set INFERNIX_NUMERICS_SMS to the smaller GPU's count elsewhere). Skips with
# fewer than two GPUs.
add_test(NAME infernix_qwen4_exp_split_prefill_two_gpu_real_test
  COMMAND infernix_qwen4_exp_split_prefill_real_test --devices 0,1 --reference 0 --splits 24)
set_tests_properties(infernix_qwen4_exp_split_prefill_two_gpu_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600
  ENVIRONMENT "INFERNIX_NUMERICS_SMS=70")
# Two GPUs with the gate handoff, and a round delayed past the gate's bound: that request fails with
# a recoverable handoff timeout and the next one gives the unsplit run's bits.
add_test(NAME infernix_qwen4_exp_split_gate_two_gpu_real_test
  COMMAND infernix_qwen4_exp_split_prefill_real_test --devices 0,1 --reference 0 --splits 24 --handoff gate
          --timeout --prompts 1,255 --no-cancel)
set_tests_properties(infernix_qwen4_exp_split_gate_two_gpu_real_test PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real"
  TIMEOUT 3600
  ENVIRONMENT "INFERNIX_NUMERICS_SMS=70")

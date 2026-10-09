# The layer split's pure interfaces (host only, header-only, no GPU): the segment map of a
# contiguous split and the per-segment link policies.
infernix_add_test(infernix_qwen4_exp_segment_map_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_segment_map.cpp")

infernix_add_test(infernix_qwen4_exp_link_policy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_link_policy.cpp")

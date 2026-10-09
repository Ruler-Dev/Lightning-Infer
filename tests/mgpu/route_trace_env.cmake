# The route trace's INFERNIX_ROUTE_TRACE fallback (host only, no GPU): baseline runs record traces
# for the layer-split planner without the test seam.
infernix_add_test(infernix_qwen4_exp_route_trace_env_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_route_trace_env.cpp"
  LIBRARIES infernix_model_runtime)

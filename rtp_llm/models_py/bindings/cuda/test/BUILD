load("//:def.bzl", "cc_test_wrapper", "copts", "cuda_copts")
load("@arch_config//:arch_select.bzl", "torch_deps")

cc_test = cc_test_wrapper

cc_import(
    name = "cuda13_torch_nvshmem",
    shared_library = "@pip_gpu_cuda13_torch_torch//:site-packages/torch/lib/libtorch_nvshmem.so",
)

cuda13_torch_link_deps = select({
    "@//:using_cuda13_x86": [
        ":cuda13_torch_nvshmem",
    ],
    "//conditions:default": [],
})

cc_test(
    name = "cuda_speculative_sampling_test",
    srcs = ["CudaSpeculativeSamplingTest.cc"],
    copts = copts(),
    deps = [
        "//rtp_llm/cpp/testing:device_test_utils",
        "//rtp_llm/models_py/bindings/common/kernels:vocab_prune_kernels",
        "//rtp_llm/models_py/bindings/cuda/kernels:kernels_speculative_sampling",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
    exec_properties = {'gpu':'A10'},
)

# Opt-in performance/correctness reproduction, separate from the regular CRC UT.
CUDA13_ONLY = select({
    "//:using_cuda13_x86": [],
    "//:using_cuda13_arm": [],
    "//conditions:default": ["@platforms//:incompatible"],
})

cc_library(
    name = "crc_copy_benchmark_support",
    testonly = True,
    srcs = ["CrcCopyBenchmarkSupport.cc"],
    hdrs = ["CrcCopyBenchmarkSupport.h"],
    copts = copts(),
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        "//rtp_llm/cpp/cache/block_tree_cache",
        "//rtp_llm/cpp/cache:cache_config",
        "//rtp_llm/cpp/cache:cache_core",
        "//rtp_llm/cpp/cache:device_block_pool_config_helper",
        "//rtp_llm/cpp/cache/test:cache_config_test_utils",
        "//rtp_llm/models_py/bindings:no_block_copy_hdr",
        "//rtp_llm/models_py/bindings/core:exec_ops_test_lib",
        "//rtp_llm/models_py/bindings/cuda/ops:cuda_impl",
        "//rtp_llm/models_py/bindings/cuda:no_block_copy",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ] + torch_deps() + cuda13_torch_link_deps,
)

cc_binary(
    name = "crc_copy_benchmark",
    testonly = True,
    srcs = ["CrcCopyBenchmark.cu"],
    copts = cuda_copts() + ["-O3"],
    tags = ["manual"],
    target_compatible_with = CUDA13_ONLY,
    deps = [
        ":crc_copy_benchmark_support",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

py_test(
    name = "crc_copy_benchmark_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "run_crc_copy_benchmark.py",
    ],
    main = "run_crc_copy_benchmark.py",
    data = [":crc_copy_benchmark"],
    args = ["--binary=$(location :crc_copy_benchmark)"],
    size = "large",
    timeout = "long",
    tags = ["manual", "gpu", "local", "exclusive", "no-remote"],
    target_compatible_with = CUDA13_ONLY,
)

py_test(
    name = "crc_copy_benchmark_stats_test",
    srcs = [
        "crc_copy_benchmark_stats.py",
        "crc_copy_benchmark_stats_test.py",
    ],
    main = "crc_copy_benchmark_stats_test.py",
)

#llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    install(TARGETS ${TEST_TARGET} RUNTIME)
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_target_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    install(TARGETS ${TEST_TARGET} RUNTIME)
    target_link_libraries(${TEST_TARGET} PRIVATE common)
    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

llama_build(test-sampling-history.cpp)
llama_test(test-sampling-history)

# build test-tokenizer-0 target once and add many tests
add_executable(test-tokenizer-0 test-tokenizer-0.cpp)
target_link_libraries(test-tokenizer-0 PRIVATE common)
install(TARGETS test-tokenizer-0 RUNTIME)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

if (LLAMA_LLGUIDANCE)
    llama_target_and_test(test-grammar-llguidance.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API
    #llama_target_and_test(test-sampling.cpp)
    #llama_target_and_test(test-grammar-parser.cpp)
    #llama_target_and_test(test-grammar-integration.cpp)
    #llama_target_and_test(test-llama-grammar.cpp)
    #llama_target_and_test(test-chat.cpp)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    #if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
    #    llama_target_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)
    #    target_include_directories(test-json-schema-to-grammar PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../examples/server)
    #endif()


    # build test-tokenizer-1-bpe target once and add many tests
    # add_executable(test-tokenizer-1-bpe test-tokenizer-1-bpe.cpp)
    # target_link_libraries(test-tokenizer-1-bpe PRIVATE common)
    # install(TARGETS test-tokenizer-1-bpe RUNTIME)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    # add_executable(test-tokenizer-1-spm test-tokenizer-1-spm.cpp)
    # target_link_libraries(test-tokenizer-1-spm PRIVATE common)
    # install(TARGETS test-tokenizer-1-spm RUNTIME)

    # llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${CMAKE_CURRENT_SOURCE_DIR}/../models/ggml-vocab-baichuan.gguf)

    # llama_target_and_test(test-double-float.cpp) # SLOW
endif()

#llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp )
llama_build_and_test(test-json-partial.cpp )
llama_build_and_test(test-cpu-affinity.cpp )
llama_build_and_test(test-nanbeige-loop.cpp )
#llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

llama_build_and_test(test-iq4-ks-kt-decode.cpp)
target_include_directories(test-iq4-ks-kt-decode PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)

# llama_target_and_test(test-opt.cpp) # SLOW

llama_build_and_test(test-cohort-prefix-sharing.cpp)
target_include_directories(test-cohort-prefix-sharing PRIVATE ${PROJECT_SOURCE_DIR}/examples/server)

llama_build_and_test(test-numa.cpp)
target_include_directories(test-numa PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/common)

llama_build_and_test(test-io-uring.cpp)
target_include_directories(test-io-uring PRIVATE ${PROJECT_SOURCE_DIR}/src)

llama_build_and_test(test-disruptor-ring.cpp)
target_include_directories(test-disruptor-ring PRIVATE ${PROJECT_SOURCE_DIR}/common)

llama_build_and_test(test-mpsc-ring.cpp)
target_include_directories(test-mpsc-ring PRIVATE ${PROJECT_SOURCE_DIR}/common)

llama_build_and_test(test-bench-server-queue.cpp)
target_include_directories(test-bench-server-queue PRIVATE ${PROJECT_SOURCE_DIR}/common)
llama_target_and_test(test-model-load-cancel.cpp  LABEL "model")
llama_target_and_test(test-autorelease.cpp        LABEL "model")
llama_target_and_test(test-mxfp8-compute.cpp      LABEL "quant")
# Second registration with the compute gate enabled: GGML_MXFP8_COMPUTE=1 must
# accept MXFP8 weights for MUL_MAT and the graph result must match the decoded
# reference. The default entry above runs with the gate OFF (gather-only parity).
add_test(NAME test-mxfp8-compute-enabled COMMAND $<TARGET_FILE:test-mxfp8-compute> --compute)

# test-quantize-fns.cpp was orphaned in this fork (never registered). It is kept
# buildable for targeted inspection, but not CI-gating: the fork's IQK vec_dot
# entry points delegate to iqk_mul_mat(nrc, nrc, ...) and produce NaN/abort when
# called with the test's single-row convention (bx = by = 0, nrc = 1) for many
# pre-existing types (q4_0, q5_0, q4_K, ...), independently of MXFP8.
llama_target_and_test(test-quantize-fns.cpp       LABEL "quant")
set_tests_properties(test-quantize-fns PROPERTIES DISABLED TRUE)


# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

add_executable(test-moe-cache-shares test-moe-cache-shares.cpp)
target_include_directories(test-moe-cache-shares PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-moe-cache-shares COMMAND $<TARGET_FILE:test-moe-cache-shares>)
add_executable(test-kv-zdd test-kv-zdd.cpp)
target_include_directories(test-kv-zdd PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_test(NAME test-kv-zdd COMMAND $<TARGET_FILE:test-kv-zdd>)
add_executable(test-kv-zdd-bench test-kv-zdd-bench.cpp)
target_include_directories(test-kv-zdd-bench PRIVATE ${PROJECT_SOURCE_DIR}/src)

if (GGML_CUDA)
    llama_build_and_test(test-cuda-l2-persistence.cpp)
    target_include_directories(test-cuda-l2-persistence PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
endif()

add_executable(test-npmc-structures test-npmc-structures.cpp)
target_include_directories(test-npmc-structures PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-npmc-structures PRIVATE common)
add_test(NAME test-npmc-structures COMMAND $<TARGET_FILE:test-npmc-structures>)

add_executable(test-ccd-suffix-trie test-ccd-suffix-trie.cpp)
target_include_directories(test-ccd-suffix-trie PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-ccd-suffix-trie PRIVATE common)
add_test(NAME test-ccd-suffix-trie COMMAND $<TARGET_FILE:test-ccd-suffix-trie>)

add_executable(test-paged-kv-cache test-paged-kv-cache.cpp)
target_include_directories(test-paged-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-paged-kv-cache PRIVATE common)
add_test(NAME test-paged-kv-cache COMMAND $<TARGET_FILE:test-paged-kv-cache>)

add_executable(test-tiered-kv-cache test-tiered-kv-cache.cpp)
target_include_directories(test-tiered-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-tiered-kv-cache PRIVATE common)
add_test(NAME test-tiered-kv-cache COMMAND $<TARGET_FILE:test-tiered-kv-cache>)

add_executable(test-superchip-pipeline test-superchip-pipeline.cpp)
target_include_directories(test-superchip-pipeline PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-superchip-pipeline PRIVATE common)
add_test(NAME test-superchip-pipeline COMMAND $<TARGET_FILE:test-superchip-pipeline>)

add_executable(test-model-loader-npmc test-model-loader-npmc.cpp)
target_include_directories(test-model-loader-npmc PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/ggml/include)
target_link_libraries(test-model-loader-npmc PRIVATE llama common)
add_test(NAME test-model-loader-npmc COMMAND $<TARGET_FILE:test-model-loader-npmc>)

add_executable(test-cluster-pipeline test-cluster-pipeline.cpp)
target_include_directories(test-cluster-pipeline PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-cluster-pipeline PRIVATE common)
add_test(NAME test-cluster-pipeline COMMAND $<TARGET_FILE:test-cluster-pipeline>)

add_executable(test-veb-layout test-veb-layout.cpp)
target_include_directories(test-veb-layout PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-veb-layout PRIVATE common)
add_test(NAME test-veb-layout COMMAND $<TARGET_FILE:test-veb-layout>)

add_executable(test-unreal-selector test-unreal-selector.cpp)
target_include_directories(test-unreal-selector PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test-unreal-selector PRIVATE common llama)
add_test(NAME test-unreal-selector COMMAND $<TARGET_FILE:test-unreal-selector>)

add_executable(test-bitwise-veb test-bitwise-veb.cpp)
target_include_directories(test-bitwise-veb PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-bitwise-veb PRIVATE common)
add_test(NAME test-bitwise-veb COMMAND $<TARGET_FILE:test-bitwise-veb>)

add_executable(test-npmc-superblock test-npmc-superblock.cpp)
target_include_directories(test-npmc-superblock PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-npmc-superblock PRIVATE common)
add_test(NAME test-npmc-superblock COMMAND $<TARGET_FILE:test-npmc-superblock>)

add_executable(test-spec-counterfactual test-spec-counterfactual.cpp)
target_include_directories(test-spec-counterfactual PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-spec-counterfactual PRIVATE common)
add_test(NAME test-spec-counterfactual COMMAND $<TARGET_FILE:test-spec-counterfactual>)

add_executable(test-spec-arbiter test-spec-arbiter.cpp)
target_include_directories(test-spec-arbiter PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-spec-arbiter PRIVATE common)
add_test(NAME test-spec-arbiter COMMAND $<TARGET_FILE:test-spec-arbiter>)

add_executable(test-spec-integration test-spec-integration.cpp)
target_include_directories(test-spec-integration PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-spec-integration PRIVATE common)
add_test(NAME test-spec-integration COMMAND $<TARGET_FILE:test-spec-integration>)

add_executable(test-polygraph-plan test-polygraph-plan.cpp)
target_include_directories(test-polygraph-plan PRIVATE ${PROJECT_SOURCE_DIR}/common ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(test-polygraph-plan PRIVATE common)
add_test(NAME test-polygraph-plan COMMAND $<TARGET_FILE:test-polygraph-plan>)

add_executable(test-parallel-split-failure test-parallel-split-failure.cpp)
target_include_directories(test-parallel-split-failure PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(test-parallel-split-failure PRIVATE ggml)
# Match the actual production directory definition, not only the requested option.
get_directory_property(_ggml_test_definitions DIRECTORY "${PROJECT_SOURCE_DIR}/ggml/src" COMPILE_DEFINITIONS)
if (GGML_OPENMP AND "GGML_USE_OPENMP" IN_LIST _ggml_test_definitions)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(test-parallel-split-failure PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(test-parallel-split-failure PRIVATE TEST_GGML_OPENMP)
        # OMP_THREAD_LIMIT was introduced in OpenMP 3.0. Older runtimes can
        # ignore it and correctly supply a full team, so do not expect fallback.
        if (OpenMP_CXX_VERSION VERSION_GREATER_EQUAL "3.0")
            foreach(limit IN ITEMS 1 2)
                add_test(NAME test-parallel-split-limit-${limit}
                    COMMAND $<TARGET_FILE:test-parallel-split-failure> --expect-native)
                set_tests_properties(test-parallel-split-limit-${limit} PROPERTIES TIMEOUT 30
                    ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1;OMP_DYNAMIC=FALSE;OMP_THREAD_LIMIT=${limit}")
            endforeach()
        endif()
        add_test(NAME test-parallel-split-nested
            COMMAND $<TARGET_FILE:test-parallel-split-failure> --nested-serialized)
        set_tests_properties(test-parallel-split-nested PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77
            ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1;OMP_DYNAMIC=FALSE;OMP_THREAD_LIMIT=3")
        # Make the configured Clang resource headers explicit in compile_commands
        # as well: a different clangd version may otherwise miss omp.h.
        if (CMAKE_CXX_COMPILER_CLANG_RESOURCE_DIR AND NOT MSVC)
            target_compile_options(test-parallel-split-failure PRIVATE
                "-resource-dir=${CMAKE_CXX_COMPILER_CLANG_RESOURCE_DIR}")
        endif()
    endif()
endif()
add_test(NAME test-parallel-split-failure COMMAND $<TARGET_FILE:test-parallel-split-failure>)
set_tests_properties(test-parallel-split-failure PROPERTIES TIMEOUT 30 ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1;OMP_DYNAMIC=FALSE;OMP_THREAD_LIMIT=3")

if (NOT WIN32)
    add_executable(test-eval-failure test-eval-failure.cpp)
    target_include_directories(test-eval-failure PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/examples/server)
    target_link_libraries(test-eval-failure PRIVATE common)
    add_test(NAME test-eval-failure COMMAND $<TARGET_FILE:test-eval-failure>)
    set_tests_properties(test-eval-failure PROPERTIES TIMEOUT 60 ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1")
endif()

add_executable(test-checkpoint-spill test-checkpoint-spill.cpp)
target_include_directories(test-checkpoint-spill PRIVATE ${PROJECT_SOURCE_DIR}/common)
add_test(NAME test-checkpoint-spill COMMAND $<TARGET_FILE:test-checkpoint-spill>)

# Host-only resource/cache regression: mock CUDA calls, use production owners.
find_package(Threads REQUIRED)
add_executable(test-cuda-reduction-lifecycle test-cuda-reduction-lifecycle.cpp)
target_include_directories(test-cuda-reduction-lifecycle PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
add_test(NAME test-cuda-reduction-lifecycle COMMAND $<TARGET_FILE:test-cuda-reduction-lifecycle>)
set_tests_properties(test-cuda-reduction-lifecycle PROPERTIES TIMEOUT 30)

add_executable(test-cuda-graph-memory test-cuda-graph-memory.cpp)
target_include_directories(test-cuda-graph-memory PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(test-cuda-graph-memory PRIVATE Threads::Threads)
add_test(NAME test-cuda-graph-memory COMMAND $<TARGET_FILE:test-cuda-graph-memory>)
set_tests_properties(test-cuda-graph-memory PROPERTIES TIMEOUT 30)

option(POLYGRAPH_MICROBENCH "Build test-cuda-graph-pinned (model-free CUDA backend is broken in this fork; see the .cpp header)" OFF)
if (GGML_CUDA AND POLYGRAPH_MICROBENCH)
    add_executable(test-cuda-graph-pinned test-cuda-graph-pinned.cpp)
    target_include_directories(test-cuda-graph-pinned PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/ggml/src)
    target_link_libraries(test-cuda-graph-pinned PRIVATE ggml)
    add_test(NAME test-cuda-graph-pinned COMMAND $<TARGET_FILE:test-cuda-graph-pinned>)
endif()



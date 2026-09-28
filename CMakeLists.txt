cmake_minimum_required(VERSION 3.16)
project(velum VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(VELUM_ENABLE_CUDA "Enable the CUDA backend (reserved for GGML kernels)" ON)

# --- CUDA (GGML backend) -----------------------------------------------------
# Phase 1 has no CUDA sources; this only reserves the language so later phases
# can drop in GGML's CUDA kernels without restructuring the build. GGML itself
# is integrated in a later phase (its own CMake subdirectory).
if(VELUM_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER} "
                   "(no CUDA sources in Phase 1)")
  else()
    message(STATUS "CUDA not found; building CPU-only (Phase 1 needs no CUDA)")
  endif()
endif()

# --- GGML (tensor library backend, vendored as a git submodule) --------------
# One compute path for the Flow decoder (and later LLM/HiFT): GGML's own CUDA
# kernels. Build the CPU backend always; add CUDA when VELUM_ENABLE_CUDA and a
# CUDA toolchain exist. See docs/ARCHITECTURE.md and docs/adr/0001.
#
# GGML is statically linked into velum and its tools so the built artifacts are
# self-contained: BUILD_SHARED_LIBS=OFF makes ggml-base/ggml-cpu/ggml-cuda static
# archives instead of shared libraries (ldd must not show libggml-*.so). This is
# distinct from GGML_STATIC, which adds a full `-static` system link and is NOT
# what we want here. See docs/adr/0002.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "build ggml as static libraries" FORCE)

set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(VELUM_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
endif()

# --- Apply vendored-GGML patches (idempotent) --------------------------------
# GGML is pinned as a git submodule, but we carry local source patches that fix
# upstream behavior we cannot accept as-is (see docs/adr/0002). Patches are
# applied here at configure time, before ggml compiles. Idempotent: a patch whose
# change is already present is skipped; a patch that applies neither forward nor
# reverse (upstream diverged) aborts configure with a clear message rather than
# silently building wrong code.
set(VELUM_GGML_SRC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml")
file(GLOB VELUM_GGML_PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
list(SORT VELUM_GGML_PATCHES)
foreach(_patch IN LISTS VELUM_GGML_PATCHES)
  get_filename_component(_patch_name "${_patch}" NAME)

  # Forward check: would this patch apply onto the current tree?
  execute_process(
    COMMAND git apply --check "${_patch}"
    WORKING_DIRECTORY "${VELUM_GGML_SRC}"
    RESULT_VARIABLE _fwd
    OUTPUT_VARIABLE _fwd_out ERROR_VARIABLE _fwd_err)
  if(_fwd EQUAL 0)
    execute_process(
      COMMAND git apply "${_patch}"
      WORKING_DIRECTORY "${VELUM_GGML_SRC}"
      RESULT_VARIABLE _apply
      OUTPUT_VARIABLE _apply_out ERROR_VARIABLE _apply_err)
    if(NOT _apply EQUAL 0)
      message(FATAL_ERROR "Failed to apply ggml patch ${_patch_name}:\n${_apply_err}")
    endif()
    message(STATUS "Applied ggml patch: ${_patch_name}")
    continue()
  endif()

  # Reverse check: is the change already present?
  execute_process(
    COMMAND git apply --check --reverse "${_patch}"
    WORKING_DIRECTORY "${VELUM_GGML_SRC}"
    RESULT_VARIABLE _rev
    OUTPUT_VARIABLE _rev_out ERROR_VARIABLE _rev_err)
  if(_rev EQUAL 0)
    message(STATUS "ggml patch already applied (skipping): ${_patch_name}")
    continue()
  endif()

  # Neither applies -> upstream source changed; do not build silently.
  message(FATAL_ERROR
    "patch 应用失败，可能是 GGML 版本升级导致代码变化，需要手动检查更新 patch 内容: ${_patch_name}\n"
    "  forward: ${_fwd_err}\n  reverse: ${_rev_err}")
endforeach()

add_subdirectory(third_party/ggml)

# --- ICU (Unicode NFC + character properties for the text tokenizer) ---------
# The Qwen2Tokenizer must be bit-exact with the Rust tokenizers crate, so we use
# ICU's UCD tables for NFC (unorm2) and \p{L}/\p{N}/\s (uchar) instead of
# hand-rolled Unicode data.
find_package(ICU COMPONENTS uc i18n REQUIRED)

# --- DSP frontend (pure C++, no dependencies) --------------------------------
add_library(velum_dsp STATIC
  src/dsp/log_mel.cpp
  src/dsp/fbank.cpp
)
target_include_directories(velum_dsp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- Placeholder stage modules (shells, to be implemented in later phases) ---
add_library(velum_frontend STATIC src/frontend/frontend.cpp)
target_include_directories(velum_frontend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(velum_llm STATIC
  src/llm/llm.cpp
  src/llm/ops.cpp
  src/llm/gguf.cpp
  src/llm/tokenizer.cpp
  src/llm/sampling.cpp
)
target_include_directories(velum_llm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(velum_llm PUBLIC ICU::uc ICU::i18n ggml)

add_library(velum_flow STATIC
  src/flow/flow.cpp
  src/flow/ops.cpp
  src/flow/gguf.cpp
  src/flow/prelookahead.cpp
  src/flow/dit.cpp
  src/flow/flow_matching.cpp
)
target_include_directories(velum_flow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(velum_flow PUBLIC ggml)

add_library(velum_hift STATIC
  src/hift/hift.cpp
  src/hift/ops.cpp
  src/hift/gguf.cpp
  src/hift/source.cpp
)
target_include_directories(velum_hift PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(velum_hift PUBLIC ggml)

add_library(velum_pipeline STATIC src/pipeline/pipeline.cpp)
target_include_directories(velum_pipeline PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(velum_pipeline PUBLIC velum_llm velum_flow velum_hift)

# --- Executables -------------------------------------------------------------
add_executable(velum src/cli/main.cpp)
target_link_libraries(velum PRIVATE
  velum_pipeline
  velum_dsp
  velum_frontend
  velum_llm
  velum_flow
  velum_hift
)

# --- Committed tokenizer data -> build tree ---------------------------------
# Copy the committed tokenizer data into the build tree so `velum`'s default
# --tokenizer-dir (build/tokenizer) works without running any Python. The data
# lives in data/tokenizer/ (see its README for provenance and licensing).
#
# TODO(packaging): when Velum is installed, add
#   install(FILES data/tokenizer/vocab.tsv data/tokenizer/merges.txt
#           data/tokenizer/added_tokens.tsv DESTINATION share/velum/tokenizer)
# and make the default --tokenizer-dir fall back to that installed path — a
# build/ directory will not exist next to an installed binary.
set(VELUM_TOKENIZER_DATA_OUT
  ${CMAKE_CURRENT_BINARY_DIR}/tokenizer/vocab.tsv
  ${CMAKE_CURRENT_BINARY_DIR}/tokenizer/merges.txt
  ${CMAKE_CURRENT_BINARY_DIR}/tokenizer/added_tokens.tsv)

add_custom_command(
  OUTPUT ${VELUM_TOKENIZER_DATA_OUT}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/tokenizer
  COMMAND ${CMAKE_COMMAND} -E copy
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/vocab.tsv
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/merges.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/added_tokens.tsv
          ${CMAKE_CURRENT_BINARY_DIR}/tokenizer
  DEPENDS
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/vocab.tsv
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/merges.txt
          ${CMAKE_CURRENT_SOURCE_DIR}/data/tokenizer/added_tokens.tsv
  COMMENT "Copying committed tokenizer data into build/tokenizer")

add_custom_target(velum_tokenizer_data DEPENDS ${VELUM_TOKENIZER_DATA_OUT})
add_dependencies(velum velum_tokenizer_data)

# Test utility used by tests/verify_dsp.py for the numerical comparison.
add_executable(velum_dump tests/dump_features.cpp)
target_link_libraries(velum_dump PRIVATE velum_dsp)

# Test utility used by tests/verify_flow.py: runs the Flow decoder and dumps
# per-stage tensors for comparison against the PyTorch reference.
add_executable(velum_flow_dump tests/flow_dump.cpp)
target_link_libraries(velum_flow_dump PRIVATE velum_flow)

# Test utility used by tests/verify_hift.py: runs the HiFT vocoder and dumps
# per-stage tensors for comparison against the PyTorch reference.
add_executable(velum_hift_dump tests/hift_dump.cpp)
target_link_libraries(velum_hift_dump PRIVATE velum_hift)

# Test utility used by tests/verify_hift_source.py: loads the SineGen2 fixed
# source asset (load_source) and verifies the round-trip + production vocode.
add_executable(velum_hift_source_dump tests/hift_source_dump.cpp)
target_link_libraries(velum_hift_source_dump PRIVATE velum_hift)

# Test utility used by tests/verify_tokenizer.py: runs the C++ text tokenizer
# and dumps token ids for bit-exact comparison against CosyVoice3Tokenizer.
add_executable(velum_tokenizer_dump tests/tokenizer_dump.cpp)
target_link_libraries(velum_tokenizer_dump PRIVATE velum_llm)

# Test utility used by tests/verify_llm.py: runs the Qwen2 prefill and dumps
# per-layer hidden states + final_norm + logits for comparison against the
# PyTorch reference.
add_executable(velum_llm_dump tests/llm_dump.cpp)
target_link_libraries(velum_llm_dump PRIVATE velum_llm)

# Test utility used by tests/verify_llm_build.py: assembles the CosyVoice3LM
# lm_input from raw token ids (embed_tokens + speech_embedding lookups + concat)
# and dumps it for bit-exact comparison against the zero-shot PyTorch reference.
add_executable(velum_llm_build_dump tests/llm_build_dump.cpp)
target_link_libraries(velum_llm_build_dump PRIVATE velum_llm)

# Test utility used by tests/verify_llm_decode.py: runs prefill + one KV-cache
# decode step and dumps the cache + decode logits/final_norm.
add_executable(velum_llm_decode_dump tests/llm_decode_dump.cpp)
target_link_libraries(velum_llm_decode_dump PRIVATE velum_llm)

# Test utility used by tests/verify_llm_decode_seq.py: runs prefill + the full
# autoregressive decode (injecting a captured token trajectory) and dumps the
# per-step llm_decoder logits.
add_executable(velum_llm_decode_seq_dump tests/llm_decode_seq_dump.cpp)
target_link_libraries(velum_llm_decode_seq_dump PRIVATE velum_llm)

# Test utility used by tests/verify_llm_sampling.py: reproduces the deterministic
# nucleus candidate set from raw per-step logits (sampling policy, no model run).
add_executable(velum_llm_sample_dump tests/llm_sample_dump.cpp)
target_link_libraries(velum_llm_sample_dump PRIVATE velum_llm)

# Test utility used by tests/verify_generate.py: runs the fully autonomous
# prefill -> sample -> KV-cache decode -> stop loop (own RNG, no injected history).
add_executable(velum_llm_generate_dump tests/llm_generate_dump.cpp)
target_link_libraries(velum_llm_generate_dump PRIVATE velum_llm)

# --- Tests (Python-driven numerical verification) ----------------------------
# The verify scripts need numpy (+ torch for the flow/hift ones). Use the repo's
# .venv python rather than the system `python3`, which has neither.
set(VELUM_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python" CACHE FILEPATH
    "Python interpreter (with numpy/torch) used to run the verify scripts")
enable_testing()
add_test(NAME dsp_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_DUMP=$<TARGET_FILE:velum_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_dsp.py)
add_test(NAME flow_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_FLOW_DUMP=$<TARGET_FILE:velum_flow_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_flow.py)
add_test(NAME hift_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_HIFT_DUMP=$<TARGET_FILE:velum_hift_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_hift.py)
add_test(NAME hift_source_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_HIFT_SOURCE_DUMP=$<TARGET_FILE:velum_hift_source_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_hift_source.py)
add_test(NAME tokenizer_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_TOKENIZER_DUMP=$<TARGET_FILE:velum_tokenizer_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_tokenizer.py)
add_test(NAME llm_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_DUMP=$<TARGET_FILE:velum_llm_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm.py)
add_test(NAME llm_build_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_BUILD_DUMP=$<TARGET_FILE:velum_llm_build_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_build.py)
add_test(NAME llm_decode_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_DECODE_DUMP=$<TARGET_FILE:velum_llm_decode_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_decode.py)
add_test(NAME llm_decode_seq_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_DECODE_SEQ_DUMP=$<TARGET_FILE:velum_llm_decode_seq_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_decode_seq.py)
add_test(NAME llm_sampling_numerical
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_SAMPLE_DUMP=$<TARGET_FILE:velum_llm_sample_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_llm_sampling.py)
add_test(NAME llm_generate
  COMMAND ${CMAKE_COMMAND} -E env
          VELUM_LLM_GENERATE_DUMP=$<TARGET_FILE:velum_llm_generate_dump>
          ${VELUM_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/tests/verify_generate.py)

set(TARGET llama-server)
option(LLAMA_SERVER_VERBOSE "Build verbose logging option for Server" ON)
option(LLAMA_SERVER_SSL     "Build SSL support for the server"        OFF)
option(LLAMA_SERVER_SQLITE3 "Build SQlite3 support for the server"    OFF)

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

if (MINGW)
    # fix: https://github.com/ggerganov/llama.cpp/actions/runs/9651004652/job/26617901362?pr=8006
    add_compile_definitions(_WIN32_WINNT=${GGML_WIN_VER})
endif()

set(TARGET_SRCS
    server.cpp
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-response-reader.h
    server-response-task.h
    server-completion-stream.h
    server-common.cpp
    server-common.h
    server-decision.cpp
    server-decision.h
    server-context.cpp
    server-context.h
)
set(PUBLIC_ASSETS
    index.html.gz

)

foreach(asset ${PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    message("TARGET_SRCS contains: ${input}")
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)

endforeach()

# include new llamacpp webui
set(ALT_PUBLIC_ASSETS
    index_llamacpp.html.gz
    loading.html
)

foreach(asset ${ALT_PUBLIC_ASSETS})
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/public_llamacpp/${asset}")
    set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
    list(APPEND TARGET_SRCS ${output})
    add_custom_command(
        DEPENDS "${input}"
        OUTPUT "${output}"
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
    )
    message("TARGET_SRCS contains: ${input}")
    set_source_files_properties(${output} PROPERTIES GENERATED TRUE)

endforeach()


add_executable(${TARGET} server-main.cpp server-app.h ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)
target_compile_definitions(${TARGET} PRIVATE
    SERVER_VERBOSE=$<BOOL:${LLAMA_SERVER_VERBOSE}>
)
if (MSVC)
    target_link_options(${TARGET} PRIVATE
     $<$<CONFIG:DEBUG>:/STACK:20971520,1048576 >
      $<$<CONFIG:RELEASE>:/STACK:20971520,1048576>
      )
endif()
# target_link_libraries(${TARGET} PRIVATE "/STACK:104857600")
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
#target_link_libraries(${TARGET} PRIVATE common ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_link_libraries(${TARGET} PRIVATE common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

if (LLAMA_SERVER_SSL)
    find_package(OpenSSL REQUIRED)
    target_link_libraries(${TARGET} PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(${TARGET} PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    target_compile_definitions(cpp-httplib PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    target_link_libraries(cpp-httplib PRIVATE OpenSSL::SSL OpenSSL::Crypto)
endif()

if (LLAMA_SERVER_SQLITE3)
    find_package(SQLite3  REQUIRED)
    target_link_libraries(${TARGET} PRIVATE SQLite::SQLite3)
    target_include_directories(${TARGET} PUBLIC ./sqlite_modern_cpp/hdr)
    target_compile_definitions(${TARGET} PRIVATE SQLITE3_MODERN_CPP_SUPPORT)
endif()

if (WIN32)
    TARGET_LINK_LIBRARIES(${TARGET} PRIVATE ws2_32)
endif()

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Result ownership against the production response queues; no model or HTTP loop.
if (LLAMA_BUILD_TESTS)
    add_executable(test-server-result-lifetime
        ${PROJECT_SOURCE_DIR}/tests/test-server-result-lifetime.cpp
        server-queue.cpp)
    target_include_directories(test-server-result-lifetime PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ../mtmd)
    target_link_libraries(test-server-result-lifetime PRIVATE common mtmd ${CMAKE_THREAD_LIBS_INIT})
    target_compile_definitions(test-server-result-lifetime PRIVATE SERVER_VERBOSE=0)
    target_compile_features(test-server-result-lifetime PRIVATE cxx_std_20)
    add_test(NAME test-server-result-lifetime COMMAND $<TARGET_FILE:test-server-result-lifetime>)
    set_tests_properties(test-server-result-lifetime PROPERTIES TIMEOUT 30
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1")
endif()

# Real server context/queue regression; fault injection lives only in the test.
# No production HTTP routes are imitated, and no GPU/model download is required.
if (LLAMA_BUILD_TESTS AND NOT WIN32)
    add_executable(test-server-wire ${PROJECT_SOURCE_DIR}/tests/test-server-wire.cpp
        ${PROJECT_SOURCE_DIR}/tests/response-calibration-probes.cpp ${TARGET_SRCS})
    target_include_directories(test-server-wire PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src ../mtmd)
    target_link_libraries(test-server-wire PRIVATE common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
    # Consistent across all separately compiled fixture server translation units;
    # the production server and shared libraries do not enable these hooks.
    target_compile_definitions(test-server-wire PRIVATE SERVER_VERBOSE=0 SERVER_RESPONSE_CALIBRATION=1)
    target_compile_features(test-server-wire PRIVATE cxx_std_20)
    if (LLAMA_SERVER_SSL)
        target_link_libraries(test-server-wire PRIVATE OpenSSL::SSL OpenSSL::Crypto)
        target_compile_definitions(test-server-wire PRIVATE CPPHTTPLIB_OPENSSL_SUPPORT)
    endif()
    if (LLAMA_SERVER_SQLITE3)
        target_link_libraries(test-server-wire PRIVATE SQLite::SQLite3)
        target_include_directories(test-server-wire PRIVATE ./sqlite_modern_cpp/hdr)
        target_compile_definitions(test-server-wire PRIVATE SQLITE3_MODERN_CPP_SUPPORT)
    endif()
    add_test(NAME test-server-wire COMMAND $<TARGET_FILE:test-server-wire>)
    set_tests_properties(test-server-wire PROPERTIES TIMEOUT 60
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1;TMPDIR=${CMAKE_CURRENT_BINARY_DIR}")

    add_executable(test-server-eval-failure
        ${PROJECT_SOURCE_DIR}/tests/test-server-eval-failure.cpp
        server-context.cpp server-task.cpp server-queue.cpp
        server-common.cpp server-chat.cpp server-decision.cpp)
    target_include_directories(test-server-eval-failure PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/ggml/src ../mtmd)
    target_link_libraries(test-server-eval-failure PRIVATE common mtmd cpp-httplib ${CMAKE_THREAD_LIBS_INIT})
    target_compile_definitions(test-server-eval-failure PRIVATE SERVER_VERBOSE=0)
    target_compile_features(test-server-eval-failure PRIVATE cxx_std_20)
    add_test(NAME test-server-eval-failure COMMAND $<TARGET_FILE:test-server-eval-failure>)
    set_tests_properties(test-server-eval-failure PROPERTIES TIMEOUT 60
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=;GGML_CUDA_NO_PINNED=1;TMPDIR=${CMAKE_CURRENT_BINARY_DIR}")
endif()

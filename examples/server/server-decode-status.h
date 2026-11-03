#pragma once

// Public llama_decode status, not ggml_status (whose +1 means cancellation).
// Only the documented KV-slot warning permits retrying with a smaller batch.
enum class server_decode_action {
    success,
    retry_kv,
    cancelled,
    allocation_failed,
    failed,
};

inline server_decode_action server_classify_decode(int ret) {
    switch (ret) {
        case 0:  return server_decode_action::success;
        case 1:  return server_decode_action::retry_kv;
        case -2: return server_decode_action::allocation_failed;
        case -3: return server_decode_action::cancelled;
        default: return server_decode_action::failed;
    }
}

inline const char * server_decode_error_message(server_decode_action action) {
    switch (action) {
        case server_decode_action::retry_kv:
            return "Input prompt is too big compared to KV size. Please try increasing KV size.";
        case server_decode_action::allocation_failed:
            return "Insufficient resources to evaluate the batch; request stopped without replay.";
        case server_decode_action::cancelled:
            return "Evaluation cancelled.";
        default:
            return "Backend evaluation failed; request stopped without replay.";
    }
}

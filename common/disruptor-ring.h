#pragma once

// Forwarding header for backward-compatibility.
// Implementation is consolidated into common/lockfree-ring.h
#include "lockfree-ring.h"

// Backward-compatible alias for disruptor_cpu_pause
inline void disruptor_cpu_pause() {
    lockfree_ring_cpu_pause();
}

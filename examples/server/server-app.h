#pragma once

#include <functional>

struct server_context;

// Single-instance application entry, shared by the CLI and loopback integration
// tests. Caller owns context until return. ready runs once on the owner thread
// after model/route initialization and before HTTP workers start; it receives the
// bound port (including ephemeral port zero). No request-visible fault controls.
// Process-global logging/signal state means this is not a reentrant server API.
int server_run(int argc, char ** argv, server_context & context,
               const std::function<void(int)> & ready = {});

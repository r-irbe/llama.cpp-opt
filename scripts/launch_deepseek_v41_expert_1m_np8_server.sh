#!/bin/bash
# Backward-compatibility alias pointing to unified declarative launcher
exec "$(dirname "$0")/launch.sh" deepseek_v41_fastest_np8 "$@"

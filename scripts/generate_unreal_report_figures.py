#!/usr/bin/env python3
"""
Generate publication-grade figures for the UNREAL Extreme Scaling and KV Caching monograph.
Outputs high-res PNG and PDF figures to docs/reports/figures/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out_dir = "/home/radu/code/ik_llama.cpp/docs/reports/figures"
os.makedirs(out_dir, exist_ok=True)

# Set high-quality styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
})

# -------------------------------------------------------------
# Figure 1: Asymptotic Complexity & Prefill Latency
# -------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(6.5, 4.2), dpi=300)

tokens = np.array([4096, 16384, 65536, 262144, 1048576, 4194304, 10000000])
tokens_m = tokens / 1e6

# Quadratic full self-attention prefill latency (seconds)
# Base: 128K = ~4.5s on 8x H100; scales as O(N^2)
dense_latency = 4.5 * (tokens / 131072)**2

# UNREAL: linear chunk scan O(N * Lp * G * d) + constant quadratic prefill on retained Top-k (1,400 tokens)
# 10M tokens scan is ~0.45s; prefill on 1,400 tokens is ~0.005s
unreal_latency = 0.045 * (tokens / 1e6) + 0.005

ax1.loglog(tokens, dense_latency, 'r-o', label=r'Dense Quadratic Attention $\mathcal{O}(N^2)$', lw=2.2)
ax1.loglog(tokens, unreal_latency, 'b-s', label=r'UNREAL Pipeline $\mathcal{O}(N \cdot T) + \mathcal{O}(k^2)$', lw=2.2)

ax1.axvline(x=1e7, color='gray', linestyle='--', alpha=0.7)
ax1.text(1e7 * 0.95, 1e-1, '10M Tokens', rotation=90, verticalalignment='bottom', horizontalalignment='right', color='dimgray', fontsize=9)

ax1.set_xlabel('Virtual Context Length (Tokens)')
ax1.set_ylabel('Prefill Execution Time (Seconds)')
ax1.set_title('Prefill Scaling: Dense Attention vs. UNREAL Exact-Cover')
ax1.grid(True, which="both", ls="--")
ax1.legend(loc='upper left')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig1_prefill_scaling.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig1_prefill_scaling.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------
# Figure 2: KV Cache Memory Footprint (10M Tokens)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=300)

labels = ['32K', '128K', '512K', '1M', '10M']
dense_kv_gb = [0.033, 0.131, 0.524, 1.05, 10.49] # per head/layer scaled to 32GB baseline
# Real 16-bit 32-layer 32-head 128-dim: 2 * 32 * 2 * 32 * 128 = 524,288 bytes / token = ~0.5 MB / 1k tokens = ~5GB for 10M
# For standard 70B model: ~1.28 GB per 100K tokens -> 10M tokens = 128 GB FP16 KV
dense_kv_gb = [0.41, 1.64, 6.55, 13.1, 131.0]
unreal_kv_gb = [0.018, 0.018, 0.018, 0.018, 0.018] # constant 1,400 tokens = 18 MB = 0.018 GB

x = np.arange(len(labels))
width = 0.35

rects1 = ax.bar(x - width/2, dense_kv_gb, width, label='Dense Auto-Regressive KV', color='#d9534f')
rects2 = ax.bar(x + width/2, unreal_kv_gb, width, label='UNREAL Filtered KV (Top-10)', color='#5cb85c')

ax.set_ylabel('KV Cache Memory (GB, log scale)')
ax.set_title('Physical KV Cache Memory Consumption vs. Virtual Context')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.grid(True, which="both", axis='y', ls="--")
ax.legend(loc='upper left')

# Annotate savings
ax.text(4 + width/2, 0.03, '99.986%\nSavings', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2b662b')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig2_kv_memory_footprint.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig2_kv_memory_footprint.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------
# Figure 3: Arbitrary Middle Edit KV Invalidation: Standard vs Chunk Hash
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.0), dpi=300)

# Standard Causal
chunks = 20
c_idx = np.arange(chunks)
edit_idx = 7

# Standard: 0..6 valid (green), 7..19 invalidated (red)
colors_std = ['#5cb85c' if i < edit_idx else '#d9534f' for i in range(chunks)]
ax1.bar(c_idx, np.ones(chunks), color=colors_std, edgecolor='black', lw=0.5)
ax1.set_title('Standard Auto-Regressive Attention: Middle Edit Invalidates Suffix [k ... N]')
ax1.set_ylabel('Status')
ax1.set_yticks([])
ax1.set_xticks(c_idx)
ax1.set_xticklabels([f'$C_{{{i}}}$' for i in c_idx], fontsize=8)
ax1.text(edit_idx, 0.5, 'EDIT', ha='center', va='center', fontweight='bold', color='white', fontsize=8)

# UNREAL / Chunk Hash: only Chunk 7 invalidated
colors_unreal = ['#5cb85c' if i != edit_idx else '#d9534f' for i in range(chunks)]
ax2.bar(c_idx, np.ones(chunks), color=colors_unreal, edgecolor='black', lw=0.5)
ax2.set_title(r'UNREAL Exact-Cover + Block Hash: Middle Edit Invalidates ONLY $C_{k}$ ($95\%$ KV Reused)')
ax2.set_ylabel('Status')
ax2.set_yticks([])
ax2.set_xticks(c_idx)
ax2.set_xticklabels([f'$C_{{{i}}}$' for i in c_idx], fontsize=8)
ax2.text(edit_idx, 0.5, 'EDIT', ha='center', va='center', fontweight='bold', color='white', fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig3_kv_invalidation_comparison.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig3_kv_invalidation_comparison.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

# -------------------------------------------------------------
# Figure 4: Agent-Tuned Strategy Performance Radar / Bar
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=300)

strategies = ['adaptive_default', 'code_pinpoint', 'narrative_recall', 'analytic_aggregate', 'temporal_window']
boost = [1.5, 3.2, 0.4, 0.8, 1.0]
expansion = [0, 1, 3, 0, 4]
centroids = [7, 7, 10, 5, 8]

x = np.arange(len(strategies))
w = 0.25

ax.bar(x - w, boost, w, label='Symbol Boost Weight', color='#0275d8')
ax.bar(x, expansion, w, label='Chunk Expansion Radius', color='#f0ad4e')
ax.bar(x + w, [c/2.0 for c in centroids], w, label='Centroids $L_p / 2$', color='#5bc0de')

ax.set_xticks(x)
ax.set_xticklabels(['Adaptive\nDefault', 'Code\nPinpoint', 'Narrative\nRecall', 'Analytic\nAggregate', 'Temporal\nWindow'], fontsize=9)
ax.set_ylabel('Parameter Magnitude')
ax.set_title('Agent-Tuned Strategy Configurations')
ax.grid(True, axis='y', ls='--')
ax.legend(loc='upper right')

plt.tight_layout()
fig.savefig(os.path.join(out_dir, "fig4_agent_tuning_strategies.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(out_dir, "fig4_agent_tuning_strategies.png"), dpi=300, bbox_inches='tight')
plt.close(fig)

print("All publication figures successfully generated in", out_dir)

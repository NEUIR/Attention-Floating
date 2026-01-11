#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def detect_model_type(npz_path: str) -> str:
    """Detect model type: autoregressive or non_autoregressive"""
    path_lower = npz_path.lower()
    if "llama" in path_lower:
        return "autoregressive"
    elif "llada" in path_lower:
        return "non_autoregressive"
    else:
        return "autoregressive"


def find_sinks_from_attention(attn: np.ndarray, epsilon: float = 3.0) -> np.ndarray:
    """Detect sink tokens from attention column means"""
    col_mean = attn.mean(axis=0)
    if col_mean.size == 0:
        return np.zeros(0, dtype=bool)
    
    global_mean = float(col_mean.mean())
    global_std = float(col_mean.std())
    
    sink_mask = np.zeros_like(col_mean, dtype=bool)
    if not np.isclose(global_std, 0.0):
        thr = global_mean + epsilon * global_std
        sink_mask = col_mean > thr
    
    if not sink_mask.any():
        max_idx = int(np.argmax(col_mean))
        sink_mask[max_idx] = True
    
    return sink_mask


def compute_qk_cos_normprod(q_h: np.ndarray, k_h: np.ndarray, eps: float = 1e-12):
    """Compute QK, Cos, NormProd matrices"""
    qk = q_h @ k_h.T
    q_norm = np.linalg.norm(q_h, axis=1, keepdims=True)
    k_norm = np.linalg.norm(k_h, axis=1, keepdims=True).T
    normp = q_norm @ k_norm
    cos = qk / np.maximum(normp, eps)
    return qk, cos, normp


def mask_upper_triangle(matrix: np.ndarray, causal: bool) -> np.ndarray:
    """Mask upper triangle for causal models"""
    if not causal:
        return matrix
    
    masked = matrix.copy()
    mask = np.triu(np.ones_like(masked, dtype=bool), k=1)
    masked[mask] = np.nan
    return masked


def plot_heatmap_single(
    data: np.ndarray,
    sink_mask: np.ndarray,
    matrix_type: str,
    output_path: str,
    start_token: int = 0
):
    """Plot and save a single heatmap as PDF"""
    T = data.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    sink_positions = np.where(sink_mask)[0]
    
    im = ax.imshow(data, aspect='auto', cmap='GnBu', origin='upper')
    
    ax.set_xlabel('Key Position', fontsize=28)
    ax.set_ylabel('Query Position', fontsize=28)
    
    ax.tick_params(axis='x', which='major', labelsize=28)
    ax.tick_params(axis='y', which='major', labelsize=28)
    
    tick_spacing = max(1, 10)
    tick_positions = list(range(0, T, tick_spacing))
    tick_labels = [str(start_token + i) for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    if len(sink_positions) > 0:
        ax.scatter(sink_positions, [0]*len(sink_positions), 
                  marker='^', s=150, 
                  color='red', edgecolors='darkred', 
                  linewidths=1.5, clip_on=False, zorder=10,
                  transform=ax.get_xaxis_transform())
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=28)
    
    if len(sink_positions) > 0:
        ax.text(0.98, 0.98, '▲ Floating Position', 
               transform=ax.transAxes,
               fontsize=28, 
               color='red',
               verticalalignment='top', 
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor='red', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_heatmap_trio(
    qk: np.ndarray,
    cos: np.ndarray,
    normp: np.ndarray,
    sink_mask: np.ndarray,
    layer_idx: int,
    head_idx,
    model_type: str,
    causal: bool,
    output_path: str,
    start_token: int = 0
):
    """Save three heatmaps as separate PDF files"""
    qk_plot = mask_upper_triangle(qk, causal)
    cos_plot = mask_upper_triangle(cos, causal)
    normp_plot = mask_upper_triangle(normp, causal)
    
    base_path = output_path.replace('_heatmaps.png', '')
    
    plot_heatmap_single(
        data=qk_plot,
        sink_mask=sink_mask,
        matrix_type='QK',
        output_path=f"{base_path}_QK.pdf",
        start_token=start_token
    )
    
    plot_heatmap_single(
        data=cos_plot,
        sink_mask=sink_mask,
        matrix_type='Cos',
        output_path=f"{base_path}_Cos.pdf",
        start_token=start_token
    )
    
    plot_heatmap_single(
        data=normp_plot,
        sink_mask=sink_mask,
        matrix_type='NormProd',
        output_path=f"{base_path}_NormProd.pdf",
        start_token=start_token
    )


def extract_and_plot_heatmaps(
    npz_path: str,
    output_dir: str,
    layer_idx: int = 0,
    head_idx: int = 0,
    use_head_avg: bool = False,
    epsilon: float = 3.0,
    max_tokens: Optional[int] = None,
    start_token: int = 0,
    end_token: Optional[int] = None,
    auto_detect: bool = True,
    force_causal: Optional[bool] = None
):
    """Extract data from npz and plot heatmaps"""
    data = np.load(npz_path, allow_pickle=True)
    
    if auto_detect:
        model_type = detect_model_type(npz_path)
        causal = (model_type == "autoregressive") if force_causal is None else force_causal
    else:
        model_type = "unknown"
        causal = False if force_causal is None else force_causal
    
    attn_key = f"layer_{layer_idx}_full_attention_heads"
    k_key = f"layer_{layer_idx}_key_states"
    q_key = f"layer_{layer_idx}_query_states"
    
    attn_heads = data[attn_key]
    k_states = data[k_key]
    q_states = data[q_key] if q_key in data else None
    
    H, T, _ = attn_heads.shape
    H_kv = k_states.shape[0]
    
    original_T = T
    
    if end_token is None:
        end_token = T
    
    if max_tokens is not None:
        end_token = start_token + max_tokens
    
    start_token = max(0, start_token)
    end_token = min(T, end_token)
    
    actual_T = end_token - start_token
    attn_heads = attn_heads[:, start_token:end_token, start_token:end_token]
    k_states = k_states[:, start_token:end_token, :]
    if q_states is not None:
        q_states = q_states[:, start_token:end_token, :]
    
    if use_head_avg:
        k_h = k_states.mean(axis=0)
        
        if q_states is not None:
            q_h = q_states.mean(axis=0)
        else:
            q_h = k_h
        
        A = attn_heads.mean(axis=0)[:actual_T, :actual_T]
        head_tag = "head_avg"
    else:
        kv_idx = head_idx % H_kv
        k_h = k_states[kv_idx, :actual_T, :]
        
        if q_states is not None:
            q_h = q_states[head_idx, :actual_T, :]
        else:
            q_h = k_h
        
        A = attn_heads[head_idx, :actual_T, :actual_T]
        head_tag = f"head_{head_idx}"
    
    qk, cos, normp = compute_qk_cos_normprod(q_h, k_h)
    
    sink_mask = find_sinks_from_attention(A, epsilon=epsilon)
    
    sample_id = os.path.basename(npz_path).replace("_attentions.npz", "")
    mask_tag = "causal" if causal else "full"
    
    if start_token > 0 or end_token < original_T:
        range_tag = f"tok{start_token}-{end_token}"
    else:
        range_tag = ""
    
    if range_tag:
        output_path = os.path.join(
            output_dir,
            f"{sample_id}_layer{layer_idx}_{head_tag}_{mask_tag}_{range_tag}_heatmaps.png"
        )
    else:
        output_path = os.path.join(
            output_dir,
            f"{sample_id}_layer{layer_idx}_{head_tag}_{mask_tag}_heatmaps.png"
        )
    
    plot_heatmap_trio(
        qk=qk,
        cos=cos,
        normp=normp,
        sink_mask=sink_mask,
        layer_idx=layer_idx,
        head_idx=head_tag,
        model_type=model_type,
        causal=causal,
        output_path=output_path,
        start_token=start_token
    )


if __name__ == "__main__":
    # Autoregressive model (LLaMA)
    npz_path_autoregressive = "/path/to/data.npz"
    output_dir_autoregressive = "/path/to/autoregressive_heatmaps"
    
    # Non-autoregressive model (LLaDA)
    npz_path_non_autoregressive = "/path/to/data.npz"
    output_dir_non_autoregressive = "/path/to/non_autoregressive_heatmaps"
    
    PROCESS_NON_AUTOREGRESSIVE = True
    PROCESS_AUTOREGRESSIVE = False
    
    PROCESS_ALL_LAYERS = False
    SPECIFIC_LAYERS = [1, 11, 28]
    
    USE_HEAD_AVG = True
    HEAD_IDX = 15
    EPSILON = 3.0
    
    START_TOKEN = 0
    END_TOKEN = 256
    MAX_TOKENS = None
    
    if PROCESS_ALL_LAYERS:
        if PROCESS_AUTOREGRESSIVE:
            test_data = np.load(npz_path_autoregressive, allow_pickle=True)
        elif PROCESS_NON_AUTOREGRESSIVE:
            test_data = np.load(npz_path_non_autoregressive, allow_pickle=True)
        
        all_layers = []
        for k in test_data.keys():
            if k.startswith("layer_") and k.endswith("_full_attention_heads"):
                try:
                    layer_num = int(k.split("_")[1])
                    all_layers.append(layer_num)
                except:
                    pass
        layers_to_process = sorted(all_layers)
    else:
        layers_to_process = SPECIFIC_LAYERS
    
    if PROCESS_AUTOREGRESSIVE:
        for layer_idx in layers_to_process:
            extract_and_plot_heatmaps(
                npz_path=npz_path_autoregressive,
                output_dir=output_dir_autoregressive,
                layer_idx=layer_idx,
                head_idx=HEAD_IDX,
                use_head_avg=USE_HEAD_AVG,
                epsilon=EPSILON,
                max_tokens=MAX_TOKENS,
                start_token=START_TOKEN,
                end_token=END_TOKEN,
                auto_detect=True,
            )
    
    if PROCESS_NON_AUTOREGRESSIVE:
        for layer_idx in layers_to_process:
            extract_and_plot_heatmaps(
                npz_path=npz_path_non_autoregressive,
                output_dir=output_dir_non_autoregressive,
                layer_idx=layer_idx,
                head_idx=HEAD_IDX,
                use_head_avg=USE_HEAD_AVG,
                epsilon=EPSILON,
                max_tokens=MAX_TOKENS,
                start_token=START_TOKEN,
                end_token=END_TOKEN,
                auto_detect=True,
            )
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from pathlib import Path
import sys
from matplotlib.patches import Rectangle


def create_temporal_heatmap(
    npz_path: str,
    output_dir: str,
    layer_idx: int = None,
    average_layers: bool = False,
    model_path: str = None,
    epsilon: float = 3.0
):
    
    # Validate parameters
    if not average_layers and layer_idx is None:
        layer_idx = 20
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    step_data_list = data['step_attention_data']
    
    # Load tokenizer if provided
    tokenizer = None
    if model_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            pass
    
    # Collect incoming attention across all steps
    all_steps = []
    all_incoming_attention = []
    sequence = None
    
    # Determine which layers to use
    if average_layers:
        sample_step = step_data_list[0]
        available_layers = []
        for key in sample_step.keys():
            if key.startswith('layer_') and key.endswith('_attention_avg'):
                layer_num = int(key.split('_')[1])
                available_layers.append(layer_num)
        available_layers.sort()
    else:
        available_layers = [layer_idx]
    
    for step_data in step_data_list:
        step_num = step_data['step']
        
        layer_attentions = []
        
        for layer_num in available_layers:
            attn_key = f'layer_{layer_num}_attention_avg'
            if attn_key not in step_data:
                continue
            
            attn_matrix = step_data[attn_key]
            layer_attentions.append(attn_matrix)
        
        if len(layer_attentions) == 0:
            return
        
        if len(layer_attentions) > 1:
            attn_matrix = np.mean(layer_attentions, axis=0)
        else:
            attn_matrix = layer_attentions[0]
        
        incoming_attention = attn_matrix.mean(axis=0)
        
        all_steps.append(step_num)
        all_incoming_attention.append(incoming_attention)
        
        if sequence is None and 'sequence' in step_data:
            sequence = step_data['sequence']
            if len(sequence.shape) == 2:
                sequence = sequence[0]
    
    heatmap_data = np.array(all_incoming_attention)
    num_steps, seq_len = heatmap_data.shape
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if average_layers:
        layer_suffix = "alllayers"
    else:
        layer_suffix = f"layer{layer_idx}"
    
    # Generate temporal heatmap with sinks
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(
        heatmap_data,
        aspect='auto',
        cmap='Purples',
        interpolation='nearest',
        origin='upper'
    )

    HATCH_PATTERN = "//"
    HATCH_EDGE_COLOR = "#FFFFFF"
    HATCH_ALPHA = 0.9

    for i, step_num in enumerate(all_steps):
        incoming = all_incoming_attention[i]
        mean_score = incoming.mean()
        std_score = incoming.std()
        threshold = mean_score + epsilon * std_score
        
        sink_positions = np.where(incoming > threshold)[0]
        if len(sink_positions) == 0:
            continue

        for pos in sink_positions:
            ax.add_patch(Rectangle(
                (pos - 0.5, i - 0.5), 1.0, 1.0,
                facecolor='none',
                edgecolor=HATCH_EDGE_COLOR,
                hatch=HATCH_PATTERN,
                linewidth=0.0,
                alpha=HATCH_ALPHA,
                zorder=6
            ))
    
    ax.set_xlabel('Token Position', fontsize=24)
    ax.set_ylabel('Denoising Step', fontsize=24)
    
    proxy = Rectangle(
        (0, 0), 1, 1,
        facecolor='none',
        edgecolor=HATCH_EDGE_COLOR,
        hatch=HATCH_PATTERN,
        linewidth=0.0,
        alpha=HATCH_ALPHA
    )

    leg = ax.legend([proxy], ['Attention Floating'],
                    loc='upper right', fontsize=20, frameon=True)

    leg.get_frame().set_facecolor("#C0B4DF")
    leg.get_frame().set_alpha(1)
    leg.get_frame().set_edgecolor('none')

    ax.set_xticks(np.arange(0, seq_len, max(1, seq_len//8)))
    ax.set_yticks(np.arange(0, num_steps, max(1, num_steps//5)))
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=24)

    plt.tight_layout()
    
    filename = f'temporal_heatmap_sinks_{layer_suffix}.pdf'
    save_path = output_path / filename
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create temporal heatmap visualization (PDF format). "
                    "All heads are automatically averaged. "
                    "Choose to use a specific layer, multiple layers, or average all layers."
    )
    parser.add_argument('--npz', type=str, required=True, 
                        help='Path to attention npz file')
    parser.add_argument('--output', type=str, required=True, 
                        help='Output directory (or base directory for --layers)')
    parser.add_argument('--layer', type=int, default=None,
                        help='Single layer index (default: 20 if neither --layers nor --all_layers specified)')
    parser.add_argument('--layers', type=str, default=None,
                        help='Comma-separated layer indices (e.g., "0,5,10" or "0,1,2,...,31"). '
                             'Each layer will be saved to output/layer_X/ subdirectory.')
    parser.add_argument('--all_layers', action='store_true',
                        help='Average all layers instead of using specific layer(s)')
    parser.add_argument('--model', type=str, default=None, 
                        help='Model path for tokenizer (optional)')
    parser.add_argument('--epsilon', type=float, default=3.0, 
                        help='Sink detection threshold (default: 3.0)')
    
    args = parser.parse_args()
    
    if args.layers:
        try:
            layer_list = [int(x.strip()) for x in args.layers.split(',')]
            if not layer_list:
                raise ValueError("No layers specified")
            layer_list = sorted(set(layer_list))
        except ValueError as e:
            print(f"Error parsing --layers: {e}")
            sys.exit(1)
        
        base_path = Path(args.output)
        base_path.mkdir(parents=True, exist_ok=True)
        
        for layer_idx in layer_list:
            layer_output = base_path / f"layer_{layer_idx}"
            
            try:
                create_temporal_heatmap(
                    npz_path=args.npz,
                    output_dir=str(layer_output),
                    layer_idx=layer_idx,
                    average_layers=False,
                    model_path=args.model,
                    epsilon=args.epsilon
                )
            except Exception as e:
                print(f"Layer {layer_idx} failed: {e}")
        
    else:
        layer_idx = args.layer if args.layer is not None else 20
        
        create_temporal_heatmap(
            npz_path=args.npz,
            output_dir=args.output,
            layer_idx=layer_idx,
            average_layers=args.all_layers,
            model_path=args.model,
            epsilon=args.epsilon
        )
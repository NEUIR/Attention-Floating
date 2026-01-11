import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from scipy import stats
import os
import glob


class AttentionSinkAnalyzer:
    
    def __init__(self, npz_path: str):
        self.data = np.load(npz_path, allow_pickle=True)
        
    def calculate_ar_bos_attention(self, layer_idx: int = 0) -> Optional[Dict]:
        key = f'layer_{layer_idx}_full_attention'
        if key not in self.data:
            return None
        
        attention_matrix = self.data[key]
        bos_idx = 0
        bos_attention = attention_matrix[:, bos_idx]
        total_attention = attention_matrix.sum()
        bos_total = bos_attention.sum()
        bos_percentage = (bos_total / total_attention) * 100
        
        return {
            'layer': layer_idx,
            'bos_attention_percentage': bos_percentage,
        }
    
    def get_all_layers(self) -> List[int]:
        layers = []
        for key in self.data.keys():
            if key.endswith('_full_attention'):
                layer_num = int(key.split('_')[1])
                layers.append(layer_num)
        return sorted(layers)


class LLaDAAttentionSinkAnalyzer:
    
    def __init__(self, npz_path: str):
        self.data = np.load(npz_path, allow_pickle=True)
    
    def identify_sinks_paper_method(self, attention_matrix: np.ndarray, 
                                    epsilon: float = 3.0,
                                    min_sinks: int = 1) -> List[int]:
        cumulative_attention = attention_matrix.mean(axis=0)
        sequence_length = len(cumulative_attention)
        
        sink_indices = []
        
        for j in range(sequence_length):
            other_scores = [cumulative_attention[k] for k in range(sequence_length) if k != j]
            other_mean = np.mean(other_scores)
            
            if cumulative_attention[j] > other_mean + epsilon:
                sink_indices.append(j)
        
        if len(sink_indices) < min_sinks:
            sorted_indices = np.argsort(cumulative_attention)[-min_sinks:].tolist()
            sink_indices = sorted(list(set(sink_indices + sorted_indices)))
        
        return sorted(sink_indices)
    
    def calculate_sink_attention(self, layer_idx: int = 0, 
                                epsilon: float = 3.0) -> Optional[Dict]:
        key = f'layer_{layer_idx}_full_attention'
        if key not in self.data:
            return None
        
        attention_matrix = self.data[key]
        sink_tokens = self.identify_sinks_paper_method(attention_matrix, epsilon=epsilon)
        
        sink_attention_total = 0
        for sink_idx in sink_tokens:
            sink_attn = attention_matrix[:, sink_idx].sum()
            sink_attention_total += sink_attn
        
        total_attention = attention_matrix.sum()
        sink_percentage = (sink_attention_total / total_attention) * 100
        
        return {
            'layer': layer_idx,
            'sink_attention_percentage': sink_percentage,
            'num_sinks': len(sink_tokens),
        }
    
    def get_all_layers(self) -> List[int]:
        layers = []
        for key in self.data.keys():
            if key.endswith('_full_attention'):
                layer_num = int(key.split('_')[1])
                layers.append(layer_num)
        return sorted(layers)


class DreamAttentionSinkAnalyzer:
    
    def __init__(self, npz_path: str):
        self.data = np.load(npz_path, allow_pickle=True)
    
    def identify_sinks_paper_method(self, attention_matrix: np.ndarray, 
                                    epsilon: float = 3.0,
                                    min_sinks: int = 1) -> List[int]:
        cumulative_attention = attention_matrix.mean(axis=0)
        sequence_length = len(cumulative_attention)
        
        sink_indices = []
        
        for j in range(sequence_length):
            other_scores = [cumulative_attention[k] for k in range(sequence_length) if k != j]
            other_mean = np.mean(other_scores)
            
            if cumulative_attention[j] > other_mean + epsilon:
                sink_indices.append(j)
        
        if len(sink_indices) < min_sinks:
            sorted_indices = np.argsort(cumulative_attention)[-min_sinks:].tolist()
            sink_indices = sorted(list(set(sink_indices + sorted_indices)))
        
        return sorted(sink_indices)
    
    def calculate_sink_attention(self, layer_idx: int = 0, 
                                epsilon: float = 3.0) -> Optional[Dict]:
        key = f'layer_{layer_idx}_full_attention'
        if key not in self.data:
            return None
        
        attention_matrix = self.data[key]
        sink_tokens = self.identify_sinks_paper_method(attention_matrix, epsilon=epsilon)
        
        sink_attention_total = 0
        for sink_idx in sink_tokens:
            sink_attn = attention_matrix[:, sink_idx].sum()
            sink_attention_total += sink_attn
        
        total_attention = attention_matrix.sum()
        sink_percentage = (sink_attention_total / total_attention) * 100
        
        return {
            'layer': layer_idx,
            'sink_attention_percentage': sink_percentage,
            'num_sinks': len(sink_tokens),
        }
    
    def get_all_layers(self) -> List[int]:
        layers = []
        for key in self.data.keys():
            if key.endswith('_full_attention'):
                layer_num = int(key.split('_')[1])
                layers.append(layer_num)
        return sorted(layers)


def batch_analyze_ar(npz_folder: str, pattern: str = "*.npz") -> Dict:
    npz_files = glob.glob(os.path.join(npz_folder, pattern))
    
    if not npz_files:
        return None
    
    all_data = {}
    
    for npz_path in npz_files:
        try:
            analyzer = AttentionSinkAnalyzer(npz_path)
            layers = analyzer.get_all_layers()
            
            for layer_idx in layers:
                result = analyzer.calculate_ar_bos_attention(layer_idx)
                if result:
                    if layer_idx not in all_data:
                        all_data[layer_idx] = []
                    all_data[layer_idx].append(result['bos_attention_percentage'])
        except Exception as e:
            continue
    
    layers = sorted(all_data.keys())
    mean_percentages = []
    std_percentages = []
    
    for layer_idx in layers:
        values = all_data[layer_idx]
        mean_percentages.append(np.mean(values))
        std_percentages.append(np.std(values))
    
    return {
        'layers': layers,
        'mean_percentages': mean_percentages,
        'std_percentages': std_percentages,
        'num_samples': len(npz_files)
    }


def batch_analyze_llada(npz_folder: str, pattern: str = "*.npz") -> Dict:
    npz_files = glob.glob(os.path.join(npz_folder, pattern))
    
    if not npz_files:
        return None
    
    all_percentages = {}
    all_num_sinks = {}
    
    for npz_path in npz_files:
        try:
            analyzer = LLaDAAttentionSinkAnalyzer(npz_path)
            layers = analyzer.get_all_layers()
            
            for layer_idx in layers:
                result = analyzer.calculate_sink_attention(layer_idx)
                if result:
                    if layer_idx not in all_percentages:
                        all_percentages[layer_idx] = []
                        all_num_sinks[layer_idx] = []
                    all_percentages[layer_idx].append(result['sink_attention_percentage'])
                    all_num_sinks[layer_idx].append(result['num_sinks'])
        except Exception as e:
            continue
    
    layers = sorted(all_percentages.keys())
    mean_percentages = []
    std_percentages = []
    mean_num_sinks = []
    std_num_sinks = []
    
    for layer_idx in layers:
        pct_values = all_percentages[layer_idx]
        sink_values = all_num_sinks[layer_idx]
        
        mean_percentages.append(np.mean(pct_values))
        std_percentages.append(np.std(pct_values))
        mean_num_sinks.append(np.mean(sink_values))
        std_num_sinks.append(np.std(sink_values))
    
    return {
        'layers': layers,
        'mean_percentages': mean_percentages,
        'std_percentages': std_percentages,
        'mean_num_sinks': mean_num_sinks,
        'std_num_sinks': std_num_sinks,
        'num_samples': len(npz_files)
    }


def batch_analyze_dream(npz_folder: str, pattern: str = "*.npz") -> Dict:
    npz_files = glob.glob(os.path.join(npz_folder, pattern))
    
    if not npz_files:
        return None
    
    all_percentages = {}
    all_num_sinks = {}
    
    for npz_path in npz_files:
        try:
            analyzer = DreamAttentionSinkAnalyzer(npz_path)
            layers = analyzer.get_all_layers()
            
            for layer_idx in layers:
                result = analyzer.calculate_sink_attention(layer_idx)
                if result:
                    if layer_idx not in all_percentages:
                        all_percentages[layer_idx] = []
                        all_num_sinks[layer_idx] = []
                    all_percentages[layer_idx].append(result['sink_attention_percentage'])
                    all_num_sinks[layer_idx].append(result['num_sinks'])
        except Exception as e:
            continue
    
    layers = sorted(all_percentages.keys())
    mean_percentages = []
    std_percentages = []
    mean_num_sinks = []
    std_num_sinks = []
    
    for layer_idx in layers:
        pct_values = all_percentages[layer_idx]
        sink_values = all_num_sinks[layer_idx]
        
        mean_percentages.append(np.mean(pct_values))
        std_percentages.append(np.std(pct_values))
        mean_num_sinks.append(np.mean(sink_values))
        std_num_sinks.append(np.std(sink_values))
    
    return {
        'layers': layers,
        'mean_percentages': mean_percentages,
        'std_percentages': std_percentages,
        'mean_num_sinks': mean_num_sinks,
        'std_num_sinks': std_num_sinks,
        'num_samples': len(npz_files)
    }


def batch_analyze_qwen(npz_folder: str, pattern: str = "*.npz") -> Dict:
    npz_files = glob.glob(os.path.join(npz_folder, pattern))
    
    if not npz_files:
        return None
    
    all_data = {}
    
    for npz_path in npz_files:
        try:
            analyzer = AttentionSinkAnalyzer(npz_path)
            layers = analyzer.get_all_layers()
            
            for layer_idx in layers:
                result = analyzer.calculate_ar_bos_attention(layer_idx)
                if result:
                    if layer_idx not in all_data:
                        all_data[layer_idx] = []
                    all_data[layer_idx].append(result['bos_attention_percentage'])
        except Exception as e:
            continue
    
    layers = sorted(all_data.keys())
    mean_percentages = []
    std_percentages = []
    
    for layer_idx in layers:
        values = all_data[layer_idx]
        mean_percentages.append(np.mean(values))
        std_percentages.append(np.std(values))
    
    return {
        'layers': layers,
        'mean_percentages': mean_percentages,
        'std_percentages': std_percentages,
        'num_samples': len(npz_files)
    }


def plot_four_model_comparison(
    llama_data: Dict,
    qwen_data: Dict,
    llada_data: Dict,
    dream_data: Dict,
    method_name: str = 'paper_method (ε=3.0)',
    save_path: str = None,
    show_std: bool = True,
):
    COLOR_BLUE   = (78/255, 101/255, 155/255)
    COLOR_PURPLE = (184/255, 168/255, 207/255)
    COLOR_YELLOW = (253/255, 207/255, 158/255)
    COLOR_BROWN  = (182/255, 118/255, 108/255)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    llama_layers = llama_data['layers']
    llama_mean = llama_data['mean_percentages']
    llama_std  = llama_data['std_percentages']
    
    ax.plot(
        llama_layers, llama_mean,
        'o-', linewidth=2, markersize=6,
        color=COLOR_BLUE, label='Llama (ARM)'
    )
    if show_std:
        llama_upper = np.array(llama_mean) + np.array(llama_std)
        llama_lower = np.array(llama_mean) - np.array(llama_std)
        ax.fill_between(
            llama_layers, llama_lower, llama_upper,
            color=COLOR_BLUE, alpha=0.2
        )

    qwen_layers = qwen_data['layers']
    qwen_mean = qwen_data['mean_percentages']
    qwen_std  = qwen_data['std_percentages']
    
    ax.plot(
        qwen_layers, qwen_mean,
        '^-', linewidth=2, markersize=6,
        color=COLOR_PURPLE, label='Qwen (ARM)'
    )
    if show_std:
        qwen_upper = np.array(qwen_mean) + np.array(qwen_std)
        qwen_lower = np.array(qwen_mean) - np.array(qwen_std)
        ax.fill_between(
            qwen_layers, qwen_lower, qwen_upper,
            color=COLOR_PURPLE, alpha=0.2
        )

    llada_layers = llada_data['layers']
    llada_mean   = llada_data['mean_percentages']
    llada_std    = llada_data['std_percentages']
    
    ax.plot(
        llada_layers, llada_mean,
        's-', linewidth=2, markersize=6,
        color=COLOR_YELLOW, label='Llada (MDM)'
    )
    if show_std:
        llada_upper = np.array(llada_mean) + np.array(llada_std)
        llada_lower = np.array(llada_mean) - np.array(llada_std)
        ax.fill_between(
            llada_layers, llada_lower, llada_upper,
            color=COLOR_YELLOW, alpha=0.2
        )

    dream_layers = dream_data['layers']
    dream_mean   = dream_data['mean_percentages']
    dream_std    = dream_data['std_percentages']
    
    ax.plot(
        dream_layers, dream_mean,
        'D-', linewidth=2, markersize=6,
        color=COLOR_BROWN, label='Dream (MDM)'
    )
    if show_std:
        dream_upper = np.array(dream_mean) + np.array(dream_std)
        dream_lower = np.array(dream_mean) - np.array(dream_std)
        ax.fill_between(
            dream_layers, dream_lower, dream_upper,
            color=COLOR_BROWN, alpha=0.2
        )

    ax.set_xlabel('Layer Index', fontsize=16)
    ax.set_ylabel('Absorption Rate (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.legend(fontsize=16, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    all_layers = llama_layers + qwen_layers + llada_layers + dream_layers
    ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main_four_model_analysis(llama_folder: str, qwen_folder: str, 
                             llada_folder: str, dream_folder: str,
                             pattern: str = "*.npz",
                             save_path: str = None):
    
    llama_data = batch_analyze_ar(llama_folder, pattern)
    qwen_data = batch_analyze_qwen(qwen_folder, pattern)
    llada_data = batch_analyze_llada(llada_folder, pattern)
    dream_data = batch_analyze_dream(dream_folder, pattern)
    
    if llama_data and qwen_data and llada_data and dream_data:
        plot_four_model_comparison(llama_data, qwen_data, llada_data, dream_data,
                                   method_name='paper_method (ε=3.0)',
                                   save_path=save_path,
                                   show_std=True)
    
    return llama_data, qwen_data, llada_data, dream_data


if __name__ == "__main__":
    
    llama_folder = "./llama_data"
    qwen_folder = "./qwen_data"
    llada_folder = "./llada_data"
    dream_folder = "./dream_data"
    
    main_four_model_analysis(
        llama_folder=llama_folder,
        qwen_folder=qwen_folder,
        llada_folder=llada_folder,
        dream_folder=dream_folder,
        pattern="*.npz",
        save_path="attention_sink_comparison.png"
    )


import numpy as np
import matplotlib.pyplot as plt
import json
import os
from transformers import AutoTokenizer
from glob import glob
from tqdm import tqdm


def attention_rollout_region(pooled_by_layer, add_residual=True):
    """Compute attention rollout across layers for region-level attention"""
    if len(pooled_by_layer) == 0:
        raise ValueError("Empty pooled_by_layer")

    n_regions = pooled_by_layer[0].shape[0]
    rollout = np.eye(n_regions, dtype=np.float64)

    for A in pooled_by_layer:
        A = np.asarray(A, dtype=np.float64)
        if add_residual:
            A = A + np.eye(n_regions, dtype=np.float64)
            row_sums = A.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            A = A / row_sums
        rollout = A @ rollout

    row_sums = rollout.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    rollout = rollout / row_sums
    return rollout


class RegionPooledSumVisualizer:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    def load_data(self, data_jsonl_path, attention_npz_path):
        """Load single sample data from JSONL and NPZ files"""
        attention_data = np.load(attention_npz_path, allow_pickle=True)
        
        saved_question = None
        if 'question' in attention_data:
            saved_question = str(attention_data['question'])
        
        data_list = []
        with open(data_jsonl_path, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if line:
                    data_list.append((line_idx, json.loads(line)))
        
        example_id = os.path.basename(attention_npz_path).replace('_attentions.npz', '')
        
        matched_data = None
        
        if saved_question:
            for line_idx, d in data_list:
                if d.get('question', '') == saved_question:
                    matched_data = d
                    break
        
        if matched_data is None:
            for line_idx, d in data_list:
                if str(d.get('id', '')) == example_id:
                    matched_data = d
                    break
        
        if matched_data is None and example_id.startswith('q_'):
            try:
                hash_part = example_id.replace('q_', '')
                for line_idx, d in data_list:
                    if hash(d['question']) % 100000 == int(hash_part):
                        matched_data = d
                        break
            except Exception:
                pass
        
        if matched_data is None and len(data_list) <= 10:
            npz_dir = os.path.dirname(attention_npz_path)
            all_npz = sorted([f for f in os.listdir(npz_dir) if f.endswith('_attentions.npz')])
            if os.path.basename(attention_npz_path) in all_npz:
                idx = all_npz.index(os.path.basename(attention_npz_path))
                if idx < len(data_list):
                    matched_data = data_list[idx][1]
        
        if matched_data is None:
            raise ValueError(f"Cannot find data for {example_id}")
        
        layers = []
        for key in attention_data.keys():
            if key.endswith('_full_attention'):
                layer_num = int(key.split('_')[1])
                layers.append(layer_num)
        
        return matched_data, attention_data, sorted(layers)
    
    def define_regions(self, data, attention_data):
        """Define regions: BOS, Query, Documents, Answer"""
        question = data['question']
        passages = data.get('ctxs', [])
        
        regions = []
        
        regions.append({'name': 'BOS', 'start': 0, 'end': 1})
        
        input_text = f"Query: {question}\n\nPassages:\n"
        query_tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
        query_end = len(query_tokens)
        
        regions.append({'name': 'Query', 'start': 1, 'end': query_end})
        
        current_pos = query_end
        for i, passage in enumerate(passages):
            if 'text' in passage:
                passage_text_content = passage['text']
            elif 'title' in passage:
                passage_text_content = passage['title']
            else:
                passage_text_content = str(passage)
            
            passage_text = f"[{i+1}] {passage_text_content}\n\n"
            passage_tokens = self.tokenizer.encode(passage_text, add_special_tokens=False)
            
            regions.append({
                'name': f'Doc{i+1}',
                'start': current_pos,
                'end': current_pos + len(passage_tokens)
            })
            current_pos += len(passage_tokens)
        
        input_length = int(attention_data['input_length'])
        answer_start = input_length
        num_generated = int(attention_data['num_generated_tokens'])
        answer_end = answer_start + num_generated
        
        regions.append({'name': 'Answer', 'start': answer_start, 'end': answer_end})
        
        return regions
    
    def pool_attention(self, attn_matrix, regions):
        """Sum pooling for region-level attention with row normalization"""
        n_regions = len(regions)
        pooled = np.zeros((n_regions, n_regions), dtype=np.float64)
        
        for i in range(n_regions):
            for j in range(n_regions):
                row_start, row_end = regions[i]['start'], regions[i]['end']
                col_start, col_end = regions[j]['start'], regions[j]['end']
                
                if row_end > row_start and col_end > col_start:
                    sub = attn_matrix[row_start:row_end, col_start:col_end]
                    pooled[i, j] = np.sum(sub)
        
        row_sums = pooled.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        pooled = pooled / row_sums
        
        return pooled, [r['name'] for r in regions]
    
    def load_all_pooled_matrices(self, data_jsonl_path, attention_dir, max_samples=None):
        """Load all samples and compute rollout-averaged attention"""
        npz_files = sorted(glob(os.path.join(attention_dir, '*_attentions.npz')))
        
        if max_samples is not None:
            npz_files = npz_files[:max_samples]
        
        all_pooled_by_layer = {}
        rollout_mats = []
        region_names = None
        
        for npz_path in tqdm(npz_files, desc="Loading attention files"):
            try:
                data, attention_data, layers = self.load_data(data_jsonl_path, npz_path)
                regions = self.define_regions(data, attention_data)
                
                if region_names is None:
                    region_names = [r['name'] for r in regions]
                
                sample_layer_mats = {}
                
                for layer_idx in layers:
                    key = f'layer_{layer_idx}_full_attention'
                    if key not in attention_data:
                        continue
                    
                    attn_matrix = attention_data[key]
                    pooled, _ = self.pool_attention(attn_matrix, regions)
                    
                    if layer_idx not in all_pooled_by_layer:
                        all_pooled_by_layer[layer_idx] = []
                    all_pooled_by_layer[layer_idx].append(pooled)
                    
                    sample_layer_mats[layer_idx] = pooled
                
                if len(sample_layer_mats) > 0:
                    sorted_layer_indices = sorted(sample_layer_mats.keys())
                    pooled_by_layer_list = [sample_layer_mats[i] for i in sorted_layer_indices]
                    rollout_matrix = attention_rollout_region(pooled_by_layer_list, add_residual=True)
                    rollout_mats.append(rollout_matrix)
            
            except Exception as e:
                continue
        
        rollout_avg = None
        if len(rollout_mats) > 0:
            stacked_rollout = np.stack(rollout_mats, axis=0)
            rollout_avg = np.mean(stacked_rollout, axis=0)
        
        return rollout_avg, region_names
    
    def visualize_rollout(self, data_jsonl_path, attention_dir, output_dir,
                         max_samples=None, gold_doc_number=None):
        """Visualize rollout-averaged attention"""
        rollout_avg, region_names = self.load_all_pooled_matrices(
            data_jsonl_path, attention_dir, max_samples=max_samples)
        
        if rollout_avg is None:
            raise ValueError("No valid attention data found")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        vmin, vmax = rollout_avg.min(), rollout_avg.max()
        
        im = ax.imshow(rollout_avg, cmap='Blues', vmin=vmin, vmax=vmax)
        n = len(region_names)

        display_names = list(region_names)
        if gold_doc_number is not None:
            gold_doc_name = f"Doc{gold_doc_number}"
            for idx, name in enumerate(display_names):
                if name == gold_doc_name:
                    display_names[idx] = "Gold Doc"

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=16)
        ax.set_yticklabels(display_names, fontsize=16)

        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'region_rollout_attention.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path


def visualize_rollout_attention(data_jsonl_path, attention_dir, model_path,
                                output_dir, max_samples=None, gold_doc_number=None):
    """Main function to visualize rollout attention"""
    visualizer = RegionPooledSumVisualizer(model_path)
    return visualizer.visualize_rollout(
        data_jsonl_path=data_jsonl_path,
        attention_dir=attention_dir,
        output_dir=output_dir,
        max_samples=max_samples,
        gold_doc_number=gold_doc_number,
    )


if __name__ == "__main__":
    result = visualize_rollout_attention(
        data_jsonl_path="/path/to/data.jsonl",
        attention_dir="/path/to/attention_outputs",
        model_path="/path/to/model",
        output_dir="/path/to/output",
        max_samples=1,
        gold_doc_number=1,  #Gold_Doc_Position
    )
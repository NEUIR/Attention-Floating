#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn.functional as F
import json
import re
import string
import os
from typing import List, Dict, Any
from collections import Counter

from transformers import AutoTokenizer, AutoModel


def normalize_answer(s):
    """Normalize answer string for evaluation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def accuracy_score(prediction, ground_truth):
    """Calculate accuracy score with lenient substring matching"""
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    
    if not norm_gt:
        return 0.0
    
    if norm_gt in norm_pred:
        return 1.0
    else:
        return 0.0


def f1_score(prediction, ground_truth):
    """Calculate F1 score"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics(prediction, ground_truths):
    """Compute accuracy and F1 scores"""
    if not ground_truths:
        return {'accuracy': 0.0, 'f1': 0.0}
    
    ground_truths = [gt for gt in ground_truths if isinstance(gt, str) and gt.strip()]
    if not ground_truths:
        return {'accuracy': 0.0, 'f1': 0.0}
    
    accuracy = max(accuracy_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    
    return {'accuracy': accuracy, 'f1': f1}


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise for sampling"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate number of tokens to transfer at each step"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    """Generate text using LLaDA model"""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def load_jsonl_data(jsonl_file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data from file"""
    data = []
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_passages(data: Dict[str, Any], max_passages: int = 5) -> List[str]:
    """Extract passage content from JSON data"""
    passages = []
    if 'passage' in data and isinstance(data['passage'], list):
        for passage in data['passage'][:max_passages]:
            if 'segment' in passage:
                passages.append(passage['segment'])
    return passages


def extract_ground_truth(data: Dict[str, Any]) -> List[str]:
    """Extract ground truth answers from JSON data"""
    ground_truths = []
    
    raw_answers = None
    for field in ['output', 'answer', 'answers']:
        if field in data:
            raw_answers = data[field]
            break
    
    if raw_answers is None:
        return ground_truths
    
    if not isinstance(raw_answers, list):
        raw_answers = [raw_answers]
    
    for item in raw_answers:
        if isinstance(item, str):
            if item.strip():
                ground_truths.append(item)
        elif isinstance(item, dict):
            if 'answer' in item and isinstance(item['answer'], str):
                if item['answer'].strip():
                    ground_truths.append(item['answer'])
            elif 'text' in item and isinstance(item['text'], str):
                if item['text'].strip():
                    ground_truths.append(item['text'])
    
    return ground_truths


def build_rag_prompt(query: str, passages: List[str]) -> str:
    """Build RAG prompt"""
    context = "\n\n".join([f"Passage {i+1}:\n{passage}" for i, passage in enumerate(passages)])

    prompt = f"""Use the following passages as context if they are helpful, 
but provide a SHORT and DIRECT answer.

RULES:
- First, check the context passages for the answer
- If the context contains the answer, use it
- If not, use your general knowledge
- Answer must be 1-10 words maximum
- Do NOT include any explanations or extra text
- Do NOT repeat or summarize the passages

Context:
{context}

Question: {query}

Short Answer:"""
    return prompt


def process_single_item(model, tokenizer, device, json_data: Dict[str, Any], 
                       steps=128, gen_length=128, block_length=32, 
                       temperature=0., cfg_scale=0., remasking='low_confidence',
                       max_passages=5):
    """Process single JSON data with RAG and generate response"""
    
    query = json_data.get('input', '') or json_data.get('query', '') or json_data.get('question', '')
    passages = extract_passages(json_data, max_passages=max_passages)
    ground_truths = extract_ground_truth(json_data)
    
    if not passages:
        passages = ["No relevant context available."]
    
    rag_prompt = build_rag_prompt(query, passages)
    
    m = [{"role": "user", "content": rag_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    input_ids = tokenizer(formatted_prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    out = generate(model, input_ids, steps=steps, gen_length=gen_length, 
                  block_length=block_length, temperature=temperature, 
                  cfg_scale=cfg_scale, remasking=remasking)
    
    response = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    metrics = {'accuracy': 0.0, 'f1': 0.0}
    if ground_truths:
        metrics = compute_metrics(response, ground_truths)
    
    return {
        'id': json_data.get('id', ''),
        'query': query,
        'response': response,
        'ground_truths': ground_truths,
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1']
    }


def process_dataset(model, tokenizer, device, json_file_path: str, 
                   output_dir: str, **generation_kwargs):
    """Process a single dataset file and save results"""
    
    dataset_name = os.path.splitext(os.path.basename(json_file_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    
    if json_file_path.endswith('.jsonl'):
        data = load_jsonl_data(json_file_path)
    else:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        data = [json_data] if isinstance(json_data, dict) else json_data
    
    results = []
    total_accuracy = 0.0
    total_f1 = 0.0
    
    for item in data:
        result = process_single_item(model, tokenizer, device, item, **generation_kwargs)
        results.append(result)
        total_accuracy += result['accuracy']
        total_f1 += result['f1']
    
    avg_accuracy = total_accuracy / len(results) if len(results) > 0 else 0
    avg_f1 = total_f1 / len(results) if len(results) > 0 else 0
    
    dataset_stats = {
        'dataset_name': dataset_name,
        'total_items': len(data),
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1,
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_stats, f, ensure_ascii=False, indent=2)
    
    return {
        'dataset_name': dataset_name,
        'total_items': len(data),
        'avg_accuracy': avg_accuracy,
        'avg_f1': avg_f1
    }


def process_multiple_datasets(model, tokenizer, device, json_files: List[str], 
                            output_dir: str, **generation_kwargs):
    """Process multiple datasets and create summary statistics"""
    
    all_stats = []
    
    for json_file in json_files:
        stats = process_dataset(model, tokenizer, device, json_file, output_dir, **generation_kwargs)
        all_stats.append(stats)
    
    total_items = sum(s['total_items'] for s in all_stats)
    weighted_accuracy = sum(s['avg_accuracy'] * s['total_items'] for s in all_stats) / total_items if total_items > 0 else 0
    weighted_f1 = sum(s['avg_f1'] * s['total_items'] for s in all_stats) / total_items if total_items > 0 else 0
    
    summary = {
        'overall_stats': {
            'total_datasets': len(all_stats),
            'total_items': total_items,
            'overall_avg_accuracy': weighted_accuracy,
            'overall_avg_f1': weighted_f1
        },
        'dataset_stats': all_stats
    }
    
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return summary


def main():
    dataset_dir = '/path/to/test_data'
    model_path = '/path/to/llada_8b_instruct'
    output_dir = '/path/to/results'
    max_passages = 5
    gen_length = 32
    steps = 32
    block_length = 32
    temperature = 0.0
    device = 'cuda'
    
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    generation_params = {
        'steps': steps,
        'gen_length': gen_length,
        'block_length': block_length,
        'temperature': temperature,
        'cfg_scale': 0.,
        'remasking': 'low_confidence',
        'max_passages': max_passages
    }
    
    expected_datasets = [
        "hotpotqa_dev_psg.jsonl",
        "marco_qa_psg.jsonl",
        "nq_dev_psg.jsonl",
        "tqa_dev_psg.jsonl",
        "trex_dev_psg.jsonl"
    ]
    
    dataset_files = []
    for filename in expected_datasets:
        filepath = os.path.join(dataset_dir, filename)
        if os.path.exists(filepath):
            dataset_files.append(filepath)
    
    summary = process_multiple_datasets(
        model, tokenizer, device, dataset_files,
        output_dir=output_dir,
        **generation_params
    )


if __name__ == '__main__':
    main()
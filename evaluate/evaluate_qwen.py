#!/usr/bin/env python3

import torch
import json
import re
import string
import os
from typing import List, Dict, Any
from collections import Counter

from transformers import AutoTokenizer, AutoModelForCausalLM


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
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    ground_truths = [gt for gt in ground_truths if isinstance(gt, str) and gt.strip()]
    if not ground_truths:
        return {'accuracy': 0.0, 'f1': 0.0}
    
    accuracy = max(accuracy_score(prediction, gt) for gt in ground_truths)
    f1 = max(f1_score(prediction, gt) for gt in ground_truths)
    
    return {'accuracy': accuracy, 'f1': f1}


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
    
    if 'output' in data:
        if isinstance(data['output'], list):
            for item in data['output']:
                if isinstance(item, dict) and 'answer' in item:
                    ground_truths.append(item['answer'])
                elif isinstance(item, dict) and 'text' in item:
                    ground_truths.append(item['text'])
                elif isinstance(item, str):
                    ground_truths.append(item)
        else:
            ground_truths = [data['output']]
    elif 'answer' in data:
        if isinstance(data['answer'], list):
            ground_truths = data['answer']
        else:
            ground_truths = [data['answer']]
    elif 'answers' in data:
        if isinstance(data['answers'], list):
            ground_truths = data['answers']
        else:
            ground_truths = [data['answers']]
    
    return ground_truths


def build_rag_prompt(query: str, passages: List[str]) -> str:
    """Build RAG prompt with knowledge fallback"""
    context = "\n\n".join([f"Passage {i+1}:\n{passage}" for i, passage in enumerate(passages)])
    
    examples = """
Here are examples showing the expected answer format:

Example 1:
Context:
Passage 1: The Eiffel Tower is located in Paris, France. It was completed in 1889.

Question: Where is the Eiffel Tower located?
Answer: Paris, France

Example 2:
Context:
Passage 1: Albert Einstein was born on March 14, 1879 in Ulm, Germany.

Question: When was Albert Einstein born?
Answer: March 14, 1879

Example 3:
Context:
Passage 1: The Great Wall of China stretches over 13,000 miles.

Question: What is the capital of Japan?
Answer: Tokyo
(Note: This answer uses general knowledge since the context doesn't contain it)

---

Now answer the following question in the same concise format:
"""
    
    prompt = f"""You are a helpful assistant that answers questions. 

INSTRUCTIONS:
1. First, check if the answer is in the provided context passages
2. If the answer is in the context, use it
3. If the context doesn't contain the answer, use your general knowledge
4. Always provide a direct, concise answer (typically 1-10 words)
5. Do NOT include explanations, reasoning, or phrases like "based on" or "according to"
6. Never say "no answer found" - always attempt to answer using available information

Context:
{context}
{examples}
Question: {query}
Answer:"""
    
    return prompt


@torch.no_grad()
def generate_qwen(model, tokenizer, input_ids, attention_mask=None, max_new_tokens=32, 
                   temperature=0.7, top_p=0.9, do_sample=True):
    """Generate response using qwen model"""
    if temperature == 0.0:
        do_sample = False
    
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return outputs


def process_single_item(model, tokenizer, device, json_data: Dict[str, Any], 
                       max_new_tokens=32, temperature=0.7, top_p=0.9, max_passages=5):
    """Process single JSON data with RAG and generate response"""
    
    query = json_data.get('input', '') or json_data.get('query', '') or json_data.get('question', '')
    passages = extract_passages(json_data, max_passages=max_passages)
    ground_truths = extract_ground_truth(json_data)
    
    if not passages:
        passages = ["No relevant context available."]
    
    rag_prompt = build_rag_prompt(query, passages)
    
    m = [{"role": "user", "content": rag_prompt}]
    formatted_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    output_ids = generate_qwen(
        model, tokenizer, input_ids, 
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=(temperature > 0.0)
    )
    
    response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    
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


def process_batch(model, tokenizer, device, batch_data: List[Dict[str, Any]], **generation_kwargs):
    """Process a batch of data items"""
    batch_results = []
    
    for item in batch_data:
        result = process_single_item(model, tokenizer, device, item, **generation_kwargs)
        batch_results.append(result)
    
    return batch_results


def process_dataset(model, tokenizer, device, json_file_path: str, 
                   output_dir: str, batch_size: int = 1, **generation_kwargs):
    """Process a single dataset file and save results"""
    
    dataset_name = os.path.splitext(os.path.basename(json_file_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    
    if json_file_path.endswith('.jsonl'):
        data = load_jsonl_data(json_file_path)
    else:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        if isinstance(json_data, dict):
            data = [json_data]
        else:
            data = json_data
    
    output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    
    results = []
    total_accuracy = 0.0
    total_f1 = 0.0
    
    for batch_idx in range(0, len(data), batch_size):
        batch_data = data[batch_idx:batch_idx + batch_size]
        
        batch_results = process_batch(model, tokenizer, device, batch_data, **generation_kwargs)
        
        for result in batch_results:
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
                            output_dir: str, batch_size: int = 1, **generation_kwargs):
    """Process multiple datasets and create summary statistics"""
    
    all_stats = []
    
    for json_file in json_files:
        stats = process_dataset(model, tokenizer, device, json_file, output_dir, 
                              batch_size=batch_size, **generation_kwargs)
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
    model_path = '/path/to/Qwen2.5-7B-Instruct'
    output_dir = '/path/to/results'
    batch_size = 4
    max_passages = 5
    max_new_tokens = 32
    temperature = 0.0
    top_p = 0.9
    device = 'cuda'
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    generation_params = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
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
        batch_size=batch_size,
        **generation_params
    )


if __name__ == '__main__':
    main()
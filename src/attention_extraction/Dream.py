import torch
import numpy as np
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
from tqdm import tqdm


class AutoregressiveAttentionExtractor:
    """
    Extract attention matrices and hidden states from autoregressive transformer models
    Compatible with: LLaMA, Qwen, Mistral, etc.
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            output_attentions=True,
            output_hidden_states=True,
            attn_implementation="eager"
        ).eval()

        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // self.num_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)

    def prepare_input(self, data: Dict) -> str:
        """
        Prepare input text from data dictionary
        Expected format: {"question": "...", "answer": "..."}
        """
        if 'question' not in data:
            raise ValueError("Data must contain 'question' field")
        
        question = data['question']
        input_text = f"{question}\nAnswer:"
        return input_text

    @torch.no_grad()
    def generate_with_attention(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> Tuple[torch.Tensor, List[int], Dict[int, Tuple[int, int, int]]]:
        """
        Generate tokens step by step and record generation metadata
        """
        generated_tokens: List[int] = []
        token_generation_step: Dict[int, Tuple[int, int, int]] = {}
        current_input = input_ids

        for step in range(max_new_tokens):
            outputs = self.model(
                current_input,
                output_attentions=False,
                use_cache=True
            )

            logits = outputs.logits
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_token_id = next_token.item()
            generated_tokens.append(next_token_id)

            pos = current_input.shape[1]
            token_generation_step[int(pos)] = (0, step, int(next_token_id))

            current_input = torch.cat([current_input, next_token], dim=1)

            if next_token_id == self.tokenizer.eos_token_id:
                break

        return current_input, generated_tokens, token_generation_step

    def extract_full_attention_matrix(
        self,
        full_sequence: torch.Tensor,
        layer_idx: int,
    ) -> np.ndarray:
        """
        Extract full attention matrix (averaged over heads) for a specific layer
        Returns: [T, T] numpy array
        """
        with torch.no_grad():
            outputs = self.model(
                full_sequence,
                output_attentions=True,
                use_cache=False
            )

            attentions = outputs.attentions
            layer_attn = attentions[layer_idx]
            avg_attn = layer_attn.mean(dim=1).squeeze(0)
            avg_attn = avg_attn.float().cpu().numpy()

            return avg_attn

    def extract_hidden_states(
        self,
        full_sequence: torch.Tensor,
        layer_idx: int
    ) -> np.ndarray:
        """
        Extract hidden states for a specific layer
        Returns: [T, D] numpy array
        """
        with torch.no_grad():
            outputs = self.model(
                full_sequence,
                output_hidden_states=True,
                use_cache=False
            )
            hidden_states = outputs.hidden_states[layer_idx + 1]
            hidden_states = hidden_states.squeeze(0).float().cpu().numpy()
            return hidden_states

    def extract_attentions(
        self,
        data: Dict,
        max_new_tokens: int = 64,
        layers_to_extract: List[int] = None,
        extract_full_matrix: bool = True,
        extract_hidden_states: bool = True,
    ) -> Dict:
        """
        Extract attention matrices and hidden states for a single example
        """
        input_text = self.prepare_input(data)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        if layers_to_extract is None:
            layers_to_extract = list(range(self.num_layers))

        full_sequence, generated_tokens, token_gen_step = self.generate_with_attention(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
        )

        result: Dict[str, object] = {
            "answer_tokens": np.array(generated_tokens),
            "input_length": input_length,
            "num_generated_tokens": len(generated_tokens),
            "question": data["question"],
            "gold_answer": data.get("answer", ""),
            "token_generation_step": token_gen_step,
            "full_sequence_ids": full_sequence.squeeze(0).cpu().numpy()
        }

        for layer_idx in layers_to_extract:
            if extract_full_matrix:
                attn_matrix = self.extract_full_attention_matrix(
                    full_sequence,
                    layer_idx,
                )
                result[f"layer_{layer_idx}_full_attention"] = attn_matrix

            if extract_hidden_states:
                h_states = self.extract_hidden_states(full_sequence, layer_idx)
                result[f"layer_{layer_idx}_hidden_states"] = h_states

        return result

    def save_attentions(self, result: Dict, output_path: str):
        """
        Save extracted data to .npz file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_dict = {}

        for key, value in result.items():
            if isinstance(value, np.ndarray):
                save_dict[key] = value
            else:
                save_dict[key] = np.array(value, dtype=object)

        np.savez(output_path, **save_dict)


def process_batch_examples(
    data_jsonl_path: str,
    model_path: str,
    output_dir: str,
    layers_to_extract: List[int] = None,
    max_new_tokens: int = 64,
    extract_full_matrix: bool = True,
    extract_hidden_states: bool = True,
):
    """
    Process batch of examples from JSONL file
    Expected format per line: {"question": "...", "answer": "..."}
    """
    data_list = []
    with open(data_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError:
                continue

    os.makedirs(output_dir, exist_ok=True)

    extractor = AutoregressiveAttentionExtractor(model_path)

    for idx, data in enumerate(tqdm(data_list, desc="Processing")):
        if "id" in data:
            example_id = data["id"]
        elif "question" in data:
            example_id = f"q_{hash(data['question']) % 100000}"
        else:
            example_id = f"example_{idx}"

        try:
            result = extractor.extract_attentions(
                data,
                max_new_tokens=max_new_tokens,
                layers_to_extract=layers_to_extract,
                extract_full_matrix=extract_full_matrix,
                extract_hidden_states=extract_hidden_states,
            )

            output_path = os.path.join(output_dir, f"{example_id}_attentions.npz")
            extractor.save_attentions(result, output_path)

        except Exception as e:
            print(f"Error processing {example_id}: {str(e)}")


if __name__ == "__main__":
    data_jsonl_path = "/path/to/data.jsonl"
    model_path = "/path/to/model"
    output_dir = "/path/to/output"

    layers_to_extract = None  # Extract all layers
    max_new_tokens = 64
    extract_full_matrix = True
    extract_hidden_states = True

    process_batch_examples(
        data_jsonl_path=data_jsonl_path,
        model_path=model_path,
        output_dir=output_dir,
        layers_to_extract=layers_to_extract,
        max_new_tokens=max_new_tokens,
        extract_full_matrix=extract_full_matrix,
        extract_hidden_states=extract_hidden_states,
    )
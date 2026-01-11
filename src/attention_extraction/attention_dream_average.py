import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import os
import math
from tqdm import tqdm


class DreamAttentionExtractor:
    """
    Dream Diffusion Model Attention Extractor
    Compatible with Dream-v0-Base-7B and Dream series models
    """
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        ).eval()

        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            self.base_model = self.model.model
            self.layers = self.base_model.layers
        elif hasattr(self.model, "layers"):
            self.base_model = self.model
            self.layers = self.base_model.layers
        else:
            raise AttributeError("Cannot find layers in Dream model")

        self.num_layers = len(self.layers)

        config = self.model.config
        self.num_heads = config.num_attention_heads
        self.d_model = config.hidden_size
        self.head_dim = self.d_model // self.num_heads

        if hasattr(config, "num_key_value_heads"):
            self.num_kv_heads = config.num_key_value_heads
        else:
            self.num_kv_heads = self.num_heads

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value tensors for Grouped Query Attention"""
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(
            batch, num_key_value_heads * n_rep, slen, head_dim
        )

    def extract_qkv_from_layer(self, layer, hidden_states, layer_idx):
        """Extract Q/K/V from Dream layer"""
        with torch.no_grad():
            dtype = hidden_states.dtype

            if hasattr(layer, "input_layernorm"):
                x_normed = layer.input_layernorm(hidden_states)
            elif hasattr(layer, "ln1"):
                x_normed = layer.ln1(hidden_states)
            else:
                x_normed = hidden_states

            if hasattr(layer, "self_attn"):
                attn = layer.self_attn
            elif hasattr(layer, "attn"):
                attn = layer.attn
            else:
                raise AttributeError(f"Cannot find attention module in layer {layer_idx}")

            q = attn.q_proj(x_normed)
            k = attn.k_proj(x_normed)
            v = attn.v_proj(x_normed)

            B, T, C = q.size()

            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if hasattr(attn, "rotary_emb"):
                position_ids = torch.arange(T, dtype=torch.long, device=q.device).unsqueeze(0)
                cos, sin = attn.rotary_emb(v, position_ids)
                q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
            
            return q, k, v

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply Rotary Position Embedding to Q and K"""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        
        return q_embed, k_embed

    def compute_attention_weights(self, q, k, head_dim):
        """Compute attention weights manually"""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = torch.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)
        return attn_weights

    def extract_full_attention_matrix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
    ) -> np.ndarray:
        """Extract full attention matrix (averaged over heads) for a layer"""
        with torch.no_grad():
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.bool)
            
            outputs = self.model.model(
                input_ids, 
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states[layer_idx]
            layer = self.layers[layer_idx]

            q, k, v = self.extract_qkv_from_layer(layer, hidden_states, layer_idx)

            if self.num_kv_heads != self.num_heads:
                assert self.num_heads % self.num_kv_heads == 0
                n_rep = self.num_heads // self.num_kv_heads
                k = self._repeat_kv(k, n_rep)

            attn_weights = self.compute_attention_weights(q, k, self.head_dim)
            attn_weights = attn_weights.mean(dim=1).squeeze(0)
            attn_weights = attn_weights.float().cpu().numpy()

            return attn_weights

    def prepare_input(self, data: Dict) -> Tuple[str, List[Tuple[int, int]]]:
        """Prepare input text from data dictionary"""
        if "question" not in data:
            raise ValueError("Expected 'question' field in data")

        question = data["question"]
        messages = [{"role": "user", "content": question}]
        passage_ranges: List[Tuple[int, int]] = []
        return messages, passage_ranges

    @torch.no_grad()
    def generate_with_attention(
        self,
        input_ids,
        attention_mask,
        passage_ranges,
        layers_to_extract,
        max_new_tokens=512,
        steps=512,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.0,
    ):
        """Generate with Dream diffusion and extract attention"""
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        
        output = self.model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=top_p,
            alg=alg,
            alg_temp=alg_temp,
        )

        generated_sequences = output.sequences
        generated_tokens = generated_sequences[0, input_ids.shape[1]:].cpu().numpy()

        layer_attentions_dict = {layer_idx: [] for layer_idx in layers_to_extract}
        token_generation_step: Dict[int, Tuple[int, int]] = {}

        full_attention_mask = torch.ones_like(generated_sequences, dtype=torch.bool)
        
        outputs = self.model.model(
            generated_sequences,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        for pos_idx, token_id in enumerate(generated_tokens):
            abs_pos = input_ids.shape[1] + pos_idx
            token_generation_step[abs_pos] = (pos_idx, int(token_id))

            for layer_idx in layers_to_extract:
                hidden_states = outputs.hidden_states[layer_idx]
                layer = self.layers[layer_idx]
                
                q, k, v = self.extract_qkv_from_layer(layer, hidden_states, layer_idx)
                
                if self.num_kv_heads != self.num_heads:
                    n_rep = self.num_heads // self.num_kv_heads
                    k = self._repeat_kv(k, n_rep)
                
                attn_weights = self.compute_attention_weights(q, k, self.head_dim)
                attn_weights = attn_weights.mean(dim=1).squeeze(0).float().cpu().numpy()
                
                attn_row = attn_weights[abs_pos, :]
                
                passage_attentions = []
                for start, end in passage_ranges:
                    passage_attn = attn_row[start:end]
                    passage_attentions.append(passage_attn)
                
                layer_attentions_dict[layer_idx].append({
                    "position": abs_pos,
                    "token_id": int(token_id),
                    "passage_attentions": passage_attentions,
                })

        return generated_sequences, generated_tokens, layer_attentions_dict, token_generation_step

    def extract_attentions(
        self,
        data: Dict,
        max_new_tokens: int = 512,
        layers_to_extract: Optional[List[int]] = None,
        steps: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        alg: str = "entropy",
        alg_temp: float = 0.0,
        extract_full_matrix: bool = True,
    ) -> Dict:
        """Extract attention matrices and generation metadata"""
        messages, passage_ranges = self.prepare_input(data)

        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            return_dict=True, 
            add_generation_prompt=True
        )
        
        input_ids = inputs.input_ids.to(device=self.device)
        attention_mask = inputs.attention_mask.to(device=self.device, dtype=torch.bool)
        input_length = input_ids.shape[1]

        if layers_to_extract is None:
            layers_to_extract = list(range(self.num_layers))

        output_seq, generated_tokens, layer_attentions_dict, token_gen_step = (
            self.generate_with_attention(
                input_ids=input_ids,
                attention_mask=attention_mask,
                passage_ranges=passage_ranges,
                layers_to_extract=layers_to_extract,
                max_new_tokens=max_new_tokens,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
            )
        )

        result: Dict[str, object] = {
            "answer_tokens": generated_tokens,
            "passage_ranges": np.array(passage_ranges),
            "input_length": input_length,
            "num_generated_tokens": len(generated_tokens),
            "question": data["question"],
            "gold_answer": data.get("answer", ""),
            "token_generation_step": token_gen_step,
        }

        for layer_idx in layers_to_extract:
            attention_data = layer_attentions_dict[layer_idx]
            attention_data.sort(key=lambda x: x["position"])
            layer_attentions = []
            for item in attention_data:
                layer_attentions.append(item["passage_attentions"])
            result[f"layer_{layer_idx}_attentions"] = layer_attentions

        if extract_full_matrix:
            full_sequence = output_seq
            full_attention_mask = torch.ones_like(full_sequence, dtype=torch.bool)
            
            for layer_idx in layers_to_extract:
                full_attn_matrix = self.extract_full_attention_matrix(
                    full_sequence, 
                    full_attention_mask,
                    layer_idx,
                )
                result[f"layer_{layer_idx}_full_attention"] = full_attn_matrix

        return result

    def save_attentions(self, result: Dict, output_path: str):
        """Save attention data to .npz file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        save_dict = {}
        for key, value in result.items():
            if key.endswith("_attentions"):
                save_dict[key] = np.array(value, dtype=object)
            else:
                save_dict[key] = value

        np.savez(output_path, **save_dict)


def process_batch_examples(
    data_json_path: str,
    model_path: str,
    output_dir: str,
    layers_to_extract: Optional[List[int]] = None,
    max_new_tokens: int = 512,
    steps: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    alg: str = "entropy",
    alg_temp: float = 0.0,
    extract_full_matrix: bool = True,
):
    """Process batch of examples from JSONL file"""
    data_list = []
    with open(data_json_path, "r", encoding="utf-8") as f:
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

    extractor = DreamAttentionExtractor(model_path)

    for idx, data in enumerate(tqdm(data_list, desc="Processing")):
        example_id = data.get("id", f"example_{idx}")

        try:
            result = extractor.extract_attentions(
                data,
                max_new_tokens=max_new_tokens,
                layers_to_extract=layers_to_extract,
                steps=steps,
                temperature=temperature,
                top_p=top_p,
                alg=alg,
                alg_temp=alg_temp,
                extract_full_matrix=extract_full_matrix,
            )

            output_path = os.path.join(
                output_dir, f"{example_id}_attentions.npz"
            )
            extractor.save_attentions(result, output_path)

        except Exception as e:
            print(f"Error processing {example_id}: {e}")


if __name__ == "__main__":
    data_json_path = "/path/to/data.jsonl"
    model_path = "/path/to/dream_model"
    output_dir = "/path/to/output"

    layers_to_extract = None
    max_new_tokens = 8
    steps = 8
    temperature = 0.2
    top_p = 0.95
    alg = "entropy"
    alg_temp = 0.0
    extract_full_matrix = True

    process_batch_examples(
        data_json_path=data_json_path,
        model_path=model_path,
        output_dir=output_dir,
        layers_to_extract=layers_to_extract,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        extract_full_matrix=extract_full_matrix,
    )
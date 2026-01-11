import torch
import numpy as np
import torch.nn.functional as F
import json
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import os
import math
from tqdm import tqdm


def add_gumbel_noise(logits, temperature):
    """Add Gumbel noise to logits"""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate number of tokens to transfer per step"""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


class LLaDAAttentionExtractor:
    """
    LLaDA Mask Diffusion Model Attention Extractor
    Compatible with LLaDA models using mask-based diffusion generation
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

        if hasattr(self.model, "transformer"):
            self.base_model = self.model
        elif hasattr(self.model, "model"):
            self.base_model = self.model.model
        else:
            raise AttributeError("Cannot find transformer in model")

        self.blocks = self.base_model.transformer.blocks
        self.num_layers = len(self.blocks)
        self.mask_id = 126336

        config = self.base_model.config
        self.num_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = self.d_model // self.num_heads

        if hasattr(config, "effective_n_kv_heads"):
            self.num_kv_heads = config.effective_n_kv_heads
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

    def extract_qkv_and_apply_rope(self, block, hidden_states, layer_idx):
        """Extract Q/K/V and apply Rotary Position Embedding"""
        with torch.no_grad():
            dtype = hidden_states.dtype

            x_normed = block.attn_norm(hidden_states)

            q = block.q_proj(x_normed)
            k = block.k_proj(x_normed)
            v = block.v_proj(x_normed)

            B, T, C = q.size()

            if getattr(block, "q_norm", None) is not None and getattr(
                block, "k_norm", None
            ) is not None:
                q = block.q_norm(q).to(dtype=dtype)
                k = block.k_norm(k).to(dtype=dtype)

            q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
            k = k.view(B, T, self.num_kv_heads, C // self.num_heads).transpose(1, 2)
            v = v.view(B, T, self.num_kv_heads, C // self.num_heads).transpose(1, 2)

            if hasattr(block, "rotary_emb"):
                q, k = block.rotary_emb(q, k)

            return q, k, v

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
        layer_idx: int,
    ) -> np.ndarray:
        """Extract full attention matrix (averaged over heads) for a layer"""
        with torch.no_grad():
            outputs = self.base_model(input_ids, output_hidden_states=True)

            hidden_states = outputs.hidden_states[layer_idx]
            block = self.blocks[layer_idx]

            q, k, v = self.extract_qkv_and_apply_rope(block, hidden_states, layer_idx)

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
        input_text = f"Question: {question}\nAnswer:"
        passage_ranges: List[Tuple[int, int]] = []
        return input_text, passage_ranges

    @torch.no_grad()
    def generate_with_attention(
        self,
        prompt,
        passage_ranges,
        layers_to_extract,
        steps=8,
        gen_length=8,
        block_length=8,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    ):
        """Generate with LLaDA diffusion process and extract attention"""
        x = torch.full(
            (1, prompt.shape[1] + gen_length),
            self.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()
        prompt_index = x != self.mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        layer_attentions_dict = {layer_idx: [] for layer_idx in layers_to_extract}
        token_generation_step: Dict[int, Tuple[int, int, int]] = {}

        for num_block in range(num_blocks):
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == self.mask_id
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = x == self.mask_id

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = self.mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    outputs = self.base_model(x_, output_hidden_states=True)
                    logits = outputs.logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                    all_hidden_states = tuple(h[:1] for h in outputs.hidden_states)
                else:
                    outputs = self.base_model(x, output_hidden_states=True)
                    logits = outputs.logits
                    all_hidden_states = outputs.hidden_states

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(
                            p, dim=-1, index=torch.unsqueeze(x0, -1)
                        ),
                        -1,
                    )
                elif remasking == "random":
                    x0_p = torch.rand(
                        (x0.shape[0], x0.shape[1]), device=x0.device
                    )
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(
                    x0, dtype=torch.bool, device=x0.device
                )
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                transferred_positions = torch.where(transfer_index[0])[0].cpu().numpy()
                transferred_tokens = x0[0, transfer_index[0]].cpu().numpy()

                for pos_idx, pos in enumerate(transferred_positions):
                    if pos >= prompt.shape[1]:
                        token_id = transferred_tokens[pos_idx]
                        token_generation_step[int(pos)] = (
                            num_block,
                            i,
                            int(token_id),
                        )

                        for layer_idx in layers_to_extract:
                            hidden = all_hidden_states[layer_idx]
                            block = self.blocks[layer_idx]

                            q, k, v = self.extract_qkv_and_apply_rope(
                                block, hidden, layer_idx
                            )

                            if self.num_kv_heads != self.num_heads:
                                n_rep = self.num_heads // self.num_kv_heads
                                k = self._repeat_kv(k, n_rep)

                            attn_weights = self.compute_attention_weights(
                                q, k, self.head_dim
                            )
                            attn_weights = (
                                attn_weights.mean(dim=1).squeeze(0).float().cpu().numpy()
                            )

                            attn_row = attn_weights[pos, :]

                            passage_attentions = []
                            for start, end in passage_ranges:
                                passage_attn = attn_row[start:end]
                                passage_attentions.append(passage_attn)

                            layer_attentions_dict[layer_idx].append(
                                {
                                    "position": int(pos),
                                    "token_id": int(token_id),
                                    "passage_attentions": passage_attentions,
                                }
                            )

                x[transfer_index] = x0[transfer_index]

        generated_tokens = x[0, prompt.shape[1]:].cpu().numpy()
        return x, generated_tokens, layer_attentions_dict, token_generation_step

    def extract_attentions(
        self,
        data: Dict,
        max_new_tokens: int = 8,
        layers_to_extract: List[int] = None,
        steps: int = 8,
        block_length: int = 8,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        extract_full_matrix: bool = True,
    ) -> Dict:
        """Extract attention matrices and generation metadata"""
        input_text, passage_ranges = self.prepare_input(data)

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        input_length = input_ids.shape[1]

        if layers_to_extract is None:
            layers_to_extract = list(range(self.num_layers))

        output_seq, generated_tokens, layer_attentions_dict, token_gen_step = (
            self.generate_with_attention(
                prompt=input_ids,
                passage_ranges=passage_ranges,
                layers_to_extract=layers_to_extract,
                steps=steps,
                gen_length=max_new_tokens,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
                remasking="low_confidence",
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
            for layer_idx in layers_to_extract:
                full_attn_matrix = self.extract_full_attention_matrix(
                    full_sequence, layer_idx
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
    layers_to_extract: List[int] = None,
    max_new_tokens: int = 8,
    steps: int = 8,
    block_length: int = 8,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
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

    extractor = LLaDAAttentionExtractor(model_path)

    for idx, data in enumerate(tqdm(data_list, desc="Processing")):
        example_id = data.get("id", f"example_{idx}")

        try:
            result = extractor.extract_attentions(
                data,
                max_new_tokens=max_new_tokens,
                layers_to_extract=layers_to_extract,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                cfg_scale=cfg_scale,
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
    model_path = "/path/to/llada_model"
    output_dir = "/path/to/output"

    layers_to_extract = None
    max_new_tokens = 8
    steps = 8
    block_length = 8
    temperature = 0.0
    cfg_scale = 0.0
    extract_full_matrix = True

    process_batch_examples(
        data_json_path=data_json_path,
        model_path=model_path,
        output_dir=output_dir,
        layers_to_extract=layers_to_extract,
        max_new_tokens=max_new_tokens,
        steps=steps,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        extract_full_matrix=extract_full_matrix,
    )
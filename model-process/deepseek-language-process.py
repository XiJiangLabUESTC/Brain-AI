import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from deepseek_vl2.models import DeepseekVLV2ForCausalLM
from utils import Norm,safe_minmax_scale,process_single_model_plus,get_gpt2_pos_embedding
import gc


def get_attention_graph(model_name, save_to_file=False, file_path=None, cache_dir=None, device_map="auto"):  
    
    print(f'downloading {model_name} with device map: {device_map}')
   
    model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            device_map=device_map,  
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    model.eval()
    
   
    config = model.config.language_config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads if hasattr(config, 'num_key_value_heads') else num_heads
    num_key_value_groups = num_heads // num_kv_heads
   

    head_dim1 = getattr(config, "head_dim", hidden_size // num_heads)
    print('hidden_size // num_heads=/n',head_dim1)
    head_dim = qk_head_dim = model.language.model.layers[0].self_attn.q_head_dim
    print('real q_head_dim=',head_dim)
    dim = hidden_size

    device = next(model.parameters()).device
    
    seq_len = 50
 
    base_pos_embedding = get_gpt2_pos_embedding(
        seq_len=seq_len,
        target_dim=hidden_size,  
        device=device
    )
    print(f"Loaded GPT-2 base position embedding: {base_pos_embedding.shape}")
   
    Aggregation_result = {}
    with torch.no_grad():
        # Outer loop: iterate through each layer
        for layer_idx, layer in enumerate(model.language.model.layers):
            print(f"\nProcessing layer {layer_idx}")
            start_time = time.time()
            
            # 1. Get current layer's attention module
            attn = layer.self_attn
            
            # 2. Extract Query weights (entire layer's Q weights, for subsequent head-wise slicing)
            W_q = attn.q_proj.weight.to(device).float()  # [num_heads × head_dim, hidden_size]
            
            # 3. Extract components of Key weights (parameters related to entire layer's K weights)
            # 3.1 RoPE part shared by all heads (shape: [qk_rope_head_dim, hidden_size])
            kv_a_rope_weight = attn.kv_a_proj_with_mqa.weight[-attn.qk_rope_head_dim:, :].to(device).float()
            # 3.2 LoRA related weights (for computing non-RoPE part of Key)
            kv_a_lora_weight = attn.kv_a_proj_with_mqa.weight[:attn.kv_lora_rank, :].to(device).float()
            # 3.3 Base weights for non-RoPE part of Key (shape: [num_kv_heads × qk_nope_head_dim, hidden_size])
            #    Note: using num_kv_heads instead of num_heads here, as KV heads are fewer in GQA
            kv_b_key_weight = attn.kv_b_proj.weight[:num_kv_heads * attn.qk_nope_head_dim, :].to(device).float()
            # 3.4 Combine with LoRA to get non-RoPE part for all KV heads (shape unchanged)
            kv_combined_key_weight = (kv_b_key_weight @ kv_a_lora_weight).to(device).float()
            
            # 4. Calculate GQA grouping information (fixed parameters for current layer, compute once outside head loop)
            num_key_value_groups = num_heads // num_kv_heads  # How many Q heads share each KV head
            
            # 5. Base position embedding (all heads in current layer share original encoding, each head will scale separately later)
            base_pos_embedding = get_gpt2_pos_embedding(seq_len=seq_len, target_dim=hidden_size, device=device)
            pos_std = torch.std(base_pos_embedding)  # Standard deviation of base position embedding
            
            # 6. Inner loop: iterate through each Query head in current layer
            layer_results = []
            for h in range(num_heads):  # h is Query head index (0 to num_heads-1)
                try:
                    # --------------------------
                    # 6.1 Process Q weights for current Query head
                    # --------------------------
                    start_q = h * head_dim
                    end_q = (h + 1) * head_dim
                    W_q_h = W_q[start_q:end_q, :]  # [head_dim, hidden_size]
                    
                    # --------------------------
                    # 6.2 Process K weights corresponding to current Query head (GQA compatible)
                    # --------------------------
                    # Calculate shared KV head index corresponding to current Q head (core GQA logic)
                    # Example: if num_key_value_groups=2, then h=0 and h=1 share kv_h=0
                    kv_h = h // num_key_value_groups  # kv_h is KV head index (0 to num_kv_heads-1)
                    
                    # Extract non-RoPE part of this KV head
                    start_k_nope = kv_h * attn.qk_nope_head_dim
                    end_k_nope = (kv_h + 1) * attn.qk_nope_head_dim
                    k_nope_h = kv_combined_key_weight[start_k_nope:end_k_nope, :]  # [qk_nope_head_dim, hidden_size]
                    
                    # Concatenate shared RoPE part to form complete K head weights (same dimension as Q head)
                    W_k_h = torch.cat([k_nope_h, kv_a_rope_weight], dim=0)  # [head_dim, hidden_size]
                    
                    # --------------------------
                    # 6.3 Calculate dynamic scaling factor for current head
                    # --------------------------
                    w_q_h_std = torch.std(W_q_h)
                    w_k_h_std = torch.std(W_k_h)
                    weight_h_std = (w_q_h_std + w_k_h_std) / 2
                    
                    # Calculate scaling factor
                    if pos_std == 0:
                        scale_factor_h = torch.sqrt(torch.tensor(dim, dtype=torch.float, device=device))
                    else:
                        scale_factor_h = weight_h_std / pos_std
                    pos_embedding_h = base_pos_embedding * scale_factor_h
                    
                    # --------------------------
                    # 6.4 Calculate attention scores
                    # --------------------------
                    q = torch.matmul(pos_embedding_h, W_q_h.T)  # [seq_len, head_dim]
                    k = torch.matmul(pos_embedding_h, W_k_h.T)  # [seq_len, head_dim]
                    attn_score = torch.matmul(q, k.T)  # [seq_len, seq_len]
                    
                    # Post-processing
                    attn_score = Norm(attn_score, head_dim)
                    attn_score = F.softmax(attn_score, dim=-1)
                    attn_score = safe_minmax_scale(attn_score, device=device)
                    
                    layer_results.append(attn_score.cpu().tolist())
                    
                except Exception as e:
                    print(f"Error processing layer {layer_idx}, head {h}: {e}")
                    layer_results.append(torch.zeros(seq_len, seq_len).tolist())
            
            # 7. Save current layer results and clean up memory
            Aggregation_result[str(layer_idx)] = layer_results
            print(f"Layer {layer_idx} processed, time: {time.time() - start_time:.2f} seconds")
            del W_q, kv_combined_key_weight, kv_a_rope_weight, base_pos_embedding
            gc.collect()
            torch.cuda.empty_cache()
    
    print('\nAggregation finished')
    if save_to_file:
        with open(file_path, 'w') as f:
            json.dump(Aggregation_result, f)
        print(f"Results saved to {file_path}")
    return Aggregation_result


if __name__ == '__main__':
    # Model configuration: (model name, dedicated GPU list), supports single or multiple GPUs
    models = [
        
        ("deepseek-ai/deepseek-vl2-small", [2,3]),
        # ("model_name", [x,x,x]),  
    ]
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Starting processing with manual GPU assignment...")
    
    # Verify specified GPUs exist
    all_gpus = set(range(num_gpus))
    for model_name, gpu_ids in models:
        for gpu_id in gpu_ids:
            if gpu_id not in all_gpus:
                raise ValueError(f"GPU {gpu_id} specified for {model_name} is invalid (available: {all_gpus})")
    
    # Change parallel processes count to number of models (one independent process per model)
    max_parallel = len(models)
    print(f"Using {max_parallel} parallel processes (one per model)")
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # When submitting tasks, pass model name, index and dedicated GPU list
        futures = [
            executor.submit(
                process_single_model_plus, 
                model_name, 
                idx, 
                gpu_ids,
                get_attention_graph_func=get_attention_graph,
                base_dir='your base_dir',
                cache_dir='your cache_dir'
                )
            for idx, (model_name, gpu_ids) in enumerate(models)
        ]
        
        for future in as_completed(futures):
            result = future.result()
            if result:
                print("Task completed successfully")
            else:
                print("Task failed")
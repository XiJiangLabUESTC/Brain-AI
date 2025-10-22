import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
from transformers import AutoModelForCausalLM  
import json
from utils import Norm,safe_minmax_scale,process_single_model_plus,get_gpt2_pos_embedding



def get_attention_graph(model_name, save_to_file=False, file_path=None,cache_dir=None, device_map="auto"):  
    print(f'downloading {model_name} with device map: {device_map}')
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map=device_map,  
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    model.eval()


    # Get model configuration
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    num_key_value_groups = num_heads // num_kv_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)

    dim = hidden_size
    device = next(model.parameters()).device

    seq_len = 50
    # Get GPT-2 base positional embedding (without immediate scaling)
    base_pos_embedding = get_gpt2_pos_embedding(
        seq_len=seq_len,
        target_dim=hidden_size,  # Match model hidden dimension
        device=device
    )
    print(f"Loaded GPT-2 base positional embedding: {base_pos_embedding.shape}")

    Aggregation_result = {}
    with torch.no_grad():
        for layer_idx, layer in enumerate(model.model.layers):
            print(f"\nProcessing layer {layer_idx}")
            start_time = time.time()
            
            attn = layer.self_attn
            W_q = attn.q_proj.weight.to(device).float()  # [hidden_size, hidden_size]
            W_k = attn.k_proj.weight.to(device).float()  # [hidden_size, hidden_size]
            
            # --------------------------
            # Statistical characteristics of current layer W_Q and W_K
            # --------------------------
            # Calculate weight standard deviation (reflects value dispersion, more robust than max/min)
            w_q_std = torch.std(W_q)
            w_k_std = torch.std(W_k)
            # Take average of both as reference baseline
            weight_std = (w_q_std + w_k_std) / 2
            
            # Calculate standard deviation of current base positional embedding
            pos_std = torch.std(base_pos_embedding)
            
            # Dynamically calculate scaling factor: make pos_embedding std close to weight std
            # Avoid division by zero (if pos_std is 0, no scaling)
            if pos_std == 0:
                scale_factor = 1
            else:
                scale_factor = weight_std / pos_std
            
            # Scale positional embedding (each layer separately)
            pos_embedding = base_pos_embedding * scale_factor
            #print(f"Layer {layer_idx}: W_Q std={w_q_std:.4f}, W_K std={w_k_std:.4f}, "
            #     f"Positional embedding scaling factor={scale_factor:.4f}")
            
            layer_results = []
            
            for h in range(num_heads):
                try:
                    # Calculate current head index range
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    
                    # Extract single attention head Q/K weights
                    W_q_h = W_q[start_idx:end_idx, :]  # [head_dim, hidden_size]
                    W_k_h = W_k[start_idx:end_idx, :]  # [head_dim, hidden_size]
                    
                    # Calculate Q and K using positional embedding and weights
                    # pos_embedding: [seq_len, hidden_size]
                    q = torch.matmul(pos_embedding, W_q_h.T)  # [seq_len, head_dim]
                    if W_k.shape[0] < num_heads * head_dim:  
                        # This is GQA (num_kv_heads < num_heads)
                        num_kv_heads = W_k.shape[0] // head_dim
                        num_key_value_groups = num_heads // num_kv_heads

                        kv_h = h // num_key_value_groups
                        start_k = kv_h * head_dim
                        end_k = (kv_h + 1) * head_dim
                        W_k_h = W_k[start_k:end_k, :]  # [head_dim, hidden_size]

                    # [seq_len, hidden_size] @ [hidden_size, head_dim] -> [seq_len, head_dim]
                    k = torch.matmul(pos_embedding, W_k_h.T)
                    # Calculate attention scores
                    attn_score = torch.matmul(q, k.T)  # [seq_len, seq_len]
                    attn_score = Norm(attn_score, head_dim)
                    attn_score = F.softmax(attn_score, dim=-1)
                    attn_score = safe_minmax_scale(attn_score, device=device)
                    
                    layer_results.append(attn_score.cpu().tolist())
                    
                except Exception as e:
                    print(f"Error processing layer {layer_idx}, head {h}: {e}")
                    layer_results.append(torch.zeros(seq_len, seq_len).tolist())
            
            Aggregation_result[str(layer_idx)] = layer_results
            print(f"Layer {layer_idx} processing completed, time taken: {time.time() - start_time:.2f} seconds")
            
            # Clean up memory
            del W_q, W_k
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
        ("Qwen/Qwen2.5-0.5B", [1,2]), 
        # ("model_name", [x,x,x]),  
    ]
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Starting processing with manual GPU assignment...")
    
    # Verify if specified GPUs exist
    all_gpus = set(range(num_gpus))
    for model_name, gpu_ids in models:
        for gpu_id in gpu_ids:
            if gpu_id not in all_gpus:
                raise ValueError(f"GPU {gpu_id} specified for {model_name} is invalid (available: {all_gpus})")
    
    # Change number of parallel processes to model count (each model independent process)
    max_parallel = len(models)
    print(f"Using {max_parallel} parallel processes (one per model)")
    
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit tasks with model name, index and dedicated GPU list
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
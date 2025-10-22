import time
import torch
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.nn.functional as F
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file  
from utils import Norm, safe_minmax_scale,process_single_model

def get_attention_graph(model_name, save_to_file=False, file_path=None, device=None, cache_dir=None):
    print(f'Loading model {model_name}')
    ckpt_path = hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors",
        token="your access token",#add your access token
        cache_dir=cache_dir 
    )
    model = timm.create_model("the corresponding ViT backbone", pretrained=False) 
    state_dict = load_file(ckpt_path, device="cpu")
    msg = model.load_state_dict(state_dict, strict=False)

    model.eval()
    model.to(device)

    
    dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads
    head_dim = dim // num_heads
    seq_len = model.patch_embed.num_patches + (1 if model.cls_token is not None else 0)
    print(f"Model info: dim={dim}, num_heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")

    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        pos_embed = model.pos_embed[0].to(device)  
        if pos_embed.shape[1] != dim:
            print(f"Warning: pos_embed shape {pos_embed.shape} does not match model dim {dim}")
            pos_embed = torch.zeros(1, seq_len, dim, device=device)
    else:
        #pos_embed = torch.eye(seq_len, dim, device=device)
        print(f"Warning: Model {model_name} does not have pos_embed. Using identity matrix instead.")
        sys.exit(1)  
    # Ensure that pos_imbedding is a two-dimensional tensor [seq_1en, dim]
    if pos_embedding.dim() == 3:
        pos_embedding = pos_embedding.squeeze(0)
    print(f"pos_embedding shape: {pos_embedding.shape}")


    Aggregation_result = {}
    with torch.no_grad():
        for layer_idx, block in enumerate(model.blocks):
            print(f"\nProcessing layer {layer_idx}")
            start_time = time.time()
            
            attn = block.attn
            qkv_weight = attn.qkv.weight.T.to(device)  
            
            W_q_all = qkv_weight[:, :dim]
            W_k_all = qkv_weight[:, dim:2*dim]
            
            # --------------------------
            # Statistical characteristics of current layer W_Q and W_K
            # --------------------------
            # Calculate weight standard deviation (reflects value dispersion, more robust than max/min)
            w_q_std = torch.std(W_q_all)
            w_k_std = torch.std(W_k_all)
            # Take average of both as reference baseline
            weight_std = (w_q_std + w_k_std) / 2
            
            # Calculate standard deviation of current base positional embedding
            pos_std = torch.std(pos_embedding)
            
            # Dynamically calculate scaling factor: make pos_embedding std close to weight std
            # Avoid division by zero (if pos_std is 0, no scaling)
            if pos_std == 0:
                scale_factor = 1
            else:
                scale_factor = weight_std / pos_std
            
            # Scale positional embedding (each layer separately)
            pos_embedding = pos_embedding * scale_factor
            #print(f"Layer {layer_idx}: W_Q std={w_q_std:.4f}, W_K std={w_k_std:.4f}, "
            #     f"Positional embedding scaling factor={scale_factor:.4f}")
            
            
            layer_results = []
            
            for h in range(num_heads):
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                
                W_q_h = W_q_all[:, start_idx:end_idx]
                W_k_h = W_k_all[:, start_idx:end_idx]
                
                # print(f"Layer {layer_idx}, Head {h}: W_q_h shape {W_q_h.shape}, W_k_h shape {W_k_h.shape}")
                
                try:
                    
                    attn_score = pos_embedding @ W_q_h @ W_k_h.T @ pos_embedding.T
                    attn_score = Norm(attn_score, head_dim)
                    attn_score = F.softmax(attn_score, dim=-1)
                    attn_score = safe_minmax_scale(attn_score, device=device)  
                    
                    layer_results.append(attn_score.cpu().tolist()) 
                    
                except Exception as e:
                    print(f"Error computing aggregation for layer {layer_idx}, head {h}: {e}")
                    layer_results.append(torch.zeros(seq_len, seq_len).tolist())
            
            Aggregation_result[str(layer_idx)] = layer_results
            print(f"Layer {layer_idx} processed in {time.time() - start_time:.2f}s")
    
    print('\nAggregation finished')
    if save_to_file:
        with open(file_path, 'w') as f:
            json.dump(Aggregation_result, f)
        print(f"Results saved to {file_path}")
    return Aggregation_result

# Main program: Process models in parallel
if __name__ == '__main__':  
    models = [
        'facebook/dinov3-vits16-pretrain-lvd1689m',
    # Modify to list of model names from timm library that need processing
    ]
    
    
    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Starting parallel processing...")
    
  
    # Determine maximum parallel processes based on available GPUs and number of models
    max_parallel = min(num_gpus, len(models)) if num_gpus > 0 else 1
    print(f"Using {max_parallel} parallel processes (matching number of GPUs)")
    

    # Initialize process pool for parallel execution
    with ProcessPoolExecutor(max_workers=max_parallel) as executor:

        # Submit processing tasks for each model
        futures = [
            executor.submit(
                process_single_model, 
                model, 
                idx, 
                num_gpus,
                get_attention_graph_func=get_attention_graph,  
                base_dir='your base_dir',
                cache_dir='your cache_dir',
            )
            # Modify to your desired base directory for storing results and cache directory for downloaded models
            for idx, model in enumerate(models)
        ]

        # Track completion of tasks
        for future in as_completed(futures):
            result = future.result()
            if result:
                print("Task completed successfully")
            else:
                print("Task failed")






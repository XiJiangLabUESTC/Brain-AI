import torch
from transformers import GPT2Model
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import Norm, safe_minmax_scale, process_single_model


def get_attention_graph(model_name, save_to_file=False, file_path=None, max_position=50, device=None, cache_dir=None):
    """Extract attention graphs from GPT-2 model using positional embeddings"""
    print(f'Downloading {model_name}')
    model = GPT2Model.from_pretrained(model_name, cache_dir)
    model.eval()
    model.to(device)

    # Extract positional embeddings and model configuration
    pos_embed = model.wpe.weight[:max_position].to(device)  # [pos, dim]
    dim = model.config.hidden_size
    num_heads = model.config.n_head
    head_dim = dim // num_heads

    Aggregation_result = {}

    with torch.no_grad():
        # Process each transformer layer
        for layer_idx, block in enumerate(model.h):
            print(f'Layer {layer_idx}')
            # Extract QKV weights from attention module
            qkv_weight = block.attn.c_attn.weight.to(device)  # [dim, 3*dim]
            
            # Split into query and key weight matrices
            W_q_full = qkv_weight[:, :dim]  # [dim, dim]
            W_k_full = qkv_weight[:, dim:dim*2]  # [dim, dim]

            layer_results = []

            # Process each attention head
            for h in range(num_heads):
                print(f'  Head {h}')
                # Extract head-specific query and key weights
                W_q = W_q_full[:, h * head_dim:(h + 1) * head_dim]  # [dim, head_dim]
                W_k = W_k_full[:, h * head_dim:(h + 1) * head_dim]  # [dim, head_dim]
                
                # Compute attention scores using positional embeddings
                agg1 = (pos_embed @ W_q)  # [pos, head_dim]
                agg2 = (pos_embed @ W_k)  # [pos, head_dim]
                agg = agg1 @ agg2.T  # [pos, pos] - attention matrix
                
                # Apply normalization, softmax and scaling
                agg = Norm(agg, head_dim)  
                agg = torch.nn.functional.softmax(agg, dim=-1)
                agg = safe_minmax_scale(agg)

                layer_results.append(agg.cpu().tolist())

            Aggregation_result[str(layer_idx)] = layer_results

    print('Aggregation finished')
    # Save results if requested
    if save_to_file:
        import json
        with open(file_path, 'w') as f:
            json.dump(Aggregation_result, f, indent=2)
        print(f"Saved to {file_path}")

    return Aggregation_result


# Main program: Process models in parallel
if __name__ == '__main__':  
    models = [
        'gpt2'
    
    ]
    # Modify to list of model names from timm library that need processing
    
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
                max_position=50
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
    

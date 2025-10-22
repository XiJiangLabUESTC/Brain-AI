import torch
import timm 
import json
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from utils import Norm, safe_minmax_scale,process_single_model


def get_attention_graph(model_name, save_to_file=False, file_path=None,
                         check_relative_position=True, device=None, cache_dir=None): 
    """
    Extracts positional embeddings from a model and computes attention aggregation graphs,
    with support for incorporating relative positional relationships.
    
    Args:
        model_name (str): Name of the model to process
        save_to_file (bool): Whether to save results to a file
        file_path (str): Path to save the results if save_to_file is True
        check_relative_position (bool): Whether to check and use relative position bias
        device (torch.device): Device to run computations on
        cache_dir (str): Directory to cache the pretrained model
        
    Returns:
        dict: Aggregated attention results organized by layer and attention head
    """
    print(f'downloading {model_name} to device {device}')
    model = timm.create_model(model_name, pretrained=True, cache_dir=cache_dir)
    
    model.eval()
    model.to(device)  

    # Get model dimension information
    dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads
    head_dim = dim // num_heads
    seq_len = model.patch_embed.num_patches + (1 if model.cls_token is not None else 0)
    print(f"Model info: dim={dim}, num_heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")

    # Process absolute position embeddings
    if hasattr(model, 'pos_embed') and model.pos_embed is not None:
        pos_embed = model.pos_embed[0].to(device)  
        if pos_embed.shape[1] != dim:
            # Throw error when dimension mismatch occurs
            raise ValueError(
                f"pos_embed shape {pos_embed.shape} does not match model dim {dim}. "
                "Please check the positional embedding configuration."
            )
    else:
        # Throw error when model has no position embeddings
        raise ValueError(
            f"Model {model_name} does not have a 'pos_embed' attribute. "
            "A valid positional embedding is required for this operation."
        )
   
    pos_embedding = pos_embed
    
    
    # Ensure pos_embedding is a 2D tensor [seq_len, dim]
    if pos_embedding.dim() == 3:
        pos_embedding = pos_embedding.squeeze(0)
    
    print(f"pos_embedding shape: {pos_embedding.shape}")
    
    # Extract relative position bias information
    has_relative_position = False
    relative_position_bias = None
    if check_relative_position:
        if hasattr(model.blocks[0].attn, 'relative_position_bias_table') and \
           hasattr(model.blocks[0].attn, 'relative_position_index'):
            has_relative_position = True
            relative_position_bias = []
            
            for layer_idx, block in enumerate(model.blocks):
                attn = block.attn
                rpb_table = attn.relative_position_bias_table.to(device)  
                rpb_index = attn.relative_position_index.to(device)  
                bias = rpb_table[rpb_index.view(-1)].view(seq_len, seq_len, -1)
                bias = bias.permute(2, 0, 1).contiguous()
                relative_position_bias.append(bias)
            
            print(f"Extracted relative position bias for {len(model.blocks)} layers")
        else:
            print("Warning: Model does not have relative position bias mechanism. Using absolute position only.")

    Aggregation_result = {}

    with torch.no_grad():
        for layer_idx, block in enumerate(model.blocks):
            print(f"\nProcessing layer {layer_idx}")
            start_time = time.time()
            
            attn = block.attn
            qkv_weight = attn.qkv.weight.T.to(device)  
            
            W_q_all = qkv_weight[:, :dim]
            W_k_all = qkv_weight[:, dim:2*dim]
            
            # print(f"Layer {layer_idx}: W_q_all shape {W_q_all.shape}, W_k_all shape {W_k_all.shape}")
            
            layer_results = []
            
            for h in range(num_heads):
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                
                W_q_h = W_q_all[:, start_idx:end_idx]
                W_k_h = W_k_all[:, start_idx:end_idx]
                
                # print(f"Layer {layer_idx}, Head {h}: W_q_h shape {W_q_h.shape}, W_k_h shape {W_k_h.shape}")
                
                try:
                    # Calculate attention scores
                    abs_attn_score = pos_embedding @ W_q_h @ W_k_h.T @ pos_embedding.T
                    
                    if has_relative_position:
                        rel_attn_bias = relative_position_bias[layer_idx][h]
                        attn_score = abs_attn_score + rel_attn_bias
                    else:
                        attn_score = abs_attn_score
                    
                    attn_score = Norm(attn_score, head_dim)
                    attn_score = F.softmax(attn_score, dim=-1) 
                    attn_score = safe_minmax_scale(attn_score)
                    
                    
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
        'vit_base_patch16_224.orig_in21k'
    
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
                check_relative_position=True  
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
    


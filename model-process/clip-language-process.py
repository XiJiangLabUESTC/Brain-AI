# from transformers import BlipForConditionalGeneration
# model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')
# print(dir(model))
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import CLIPModel
from utils import Norm, safe_minmax_scale,process_single_model

def get_attention_graph(model_name, save_to_file=False, file_path=None, device=None, cache_dir=None, max_position=50):
    print(f'Loading model {model_name}')
    model = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # Text-decoder part
    text_encoder = model.text_model

    pos_embed = text_encoder.embeddings.position_embedding.weight[:max_position].to(device)  # [pos, dim]
    dim = text_encoder.config.hidden_size
    num_heads = text_encoder.config.num_attention_heads
    head_dim = dim // num_heads

    print('Extracting text decoder aggregation...')  
    text_results = {}  

    with torch.no_grad():
        # Attention layers are inside the BERT encoder
        for layer_idx, block in enumerate(text_encoder.encoder.layers):
            print(f'Layer {layer_idx}')
            qkv_weight = block.self_attn.q_proj.weight.to(device), \
                         block.self_attn.k_proj.weight.to(device)

            W_q_full, W_k_full = qkv_weight  # [out-dim, in—dim]

            layer_heads = []
            for h in range(num_heads):
                W_q = W_q_full[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, indim]
                W_k = W_k_full[h * head_dim:(h + 1) * head_dim, :]  # [head_dim, dim]

                agg1 = pos_embed @ W_q.T
                agg2 = pos_embed @ W_k.T

                agg = agg1 @ agg2.T
                agg = Norm(agg, head_dim)
                agg = torch.nn.functional.softmax(agg, dim=-1)
                agg = safe_minmax_scale(agg)

                layer_heads.append(agg.cpu().tolist())
            text_results[layer_idx] = layer_heads  
    print('Extraction done.')
    
    # Save results if requested
    if save_to_file:
        import json
        with open(file_path, 'w') as f:
            json.dump(text_results, f, indent=2)
        print(f"Saved to {file_path}")
    
    return text_results


# Main program: Process models in parallel
if __name__ == '__main__':  
    models = [
       'openai/clip-vit-base-patch16'
    
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
                max_position=77
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



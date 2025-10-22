import torch
from transformers import BertModel
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import Norm, safe_minmax_scale, process_single_model

def get_attention_graph(model_name='bert-base-uncased',save_to_file=False, file_path=None,max_position=50,device=None,cache_dir=None):
    """Extract attention graphs from BERT model using positional embeddings"""
    print(f'Downloading {model_name}')
    model = BertModel.from_pretrained(model_name,cache_dir)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pos_embed = model.embeddings.position_embeddings.weight[:max_position].to(device)
    dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = dim // num_heads

    Aggregation_result = {}

    with torch.no_grad():
        for layer_idx, layer in enumerate(model.encoder.layer):
            print(f'Layer {layer_idx}')
            query_weight = layer.attention.self.query.weight.T.to(device)  # [dim, dim]
            key_weight = layer.attention.self.key.weight.T.to(device)      # [dim, dim]

            layer_results = []
            for h in range(num_heads):
                print(f'  Head {h}')
                W_q = query_weight[:, h * head_dim: (h + 1) * head_dim]  # [dim, head_dim]
                W_k = key_weight[:, h * head_dim: (h + 1) * head_dim]    # [dim, head_dim]

                agg1 = pos_embed @ W_q  # [pos, head_dim]
                agg2 = pos_embed @ W_k  # [pos, head_dim]
                agg = agg1 @ agg2.T     # [pos, pos]
                agg = Norm(agg, head_dim)
                agg = torch.nn.functional.softmax(agg, dim=-1)
                agg = safe_minmax_scale(agg)

                layer_results.append(agg.cpu().tolist())

            Aggregation_result[str(layer_idx)] = layer_results

    print('Aggregation finished')
    if save_to_file:
        import json
        with open(file_path, 'w') as f:
            import json
            json.dump(Aggregation_result, f, indent=2)
        print(f"Saved to {file_path}")

    return Aggregation_result


# Main program: Process models in parallel
if __name__ == '__main__':  
    models = [
        'bert-base-uncased'
    
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
    

import os
import torch
import numpy as np
import networkx as nx 
import json
import community  
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import gc
from transformers import GPT2LMHeadModel
from transformers import ViTModel


def Norm(A, dim=None):

    if dim is None:
        return A.shape[1]**-0.5 * A 
    else:
        return (1/dim)**0.5 * A 

def safe_minmax_scale(matrix, epsilon=1e-5, delta=1e-5, device=None):
    """Safely scales matrix values to [delta, 1-epsilon] range while preserving zeros
    
    Performs min-max scaling on non-zero elements only, ensuring the scaled values
    stay within a safe range that avoids numerical issues in subsequent operations
    like softmax.
    """
    matrix = torch.as_tensor(matrix, device=device)
    nonzero_mask = matrix != 0
    nonzero_values = matrix[nonzero_mask]
    min_val = torch.min(nonzero_values)
    max_val = torch.max(nonzero_values)
    if max_val == min_val:
        return matrix.clone()
    scaled_values = delta + ((nonzero_values - min_val) / (max_val - min_val)) * (1 - epsilon - delta)
    result = matrix.clone()
    result[nonzero_mask] = scaled_values
    return result

def compute_metrics_for_head(matrix):
    """Calculate graph metrics for a single attention head's adjacency matrix
    
    Computes various network metrics including clustering coefficient, modularity,
    degree statistics, shortest path characteristics, and efficiency measures
    from a given adjacency matrix representing an attention graph.
    
    Args:
        matrix (list): Adjacency matrix representing the attention graph
        
    Returns:
        dict: Dictionary containing computed metrics with None values for failed calculations
    """
    try:
        # Get the size of the matrix (number of nodes)
        n = len(matrix)
        metrics = {}
        
        # Convert input matrix to numpy array
        matrix_arr = np.array(matrix)
        # Create symmetric connectivity matrix by averaging with transpose
        conn_matrix = (matrix_arr + matrix_arr.T) / 2
        # Create networkx graph from connectivity matrix
        G_conn = nx.from_numpy_array(conn_matrix)
        
        # Calculate average clustering coefficient with weighted edges
        try:
            metrics['clustering'] = nx.average_clustering(G_conn, weight='weight')
        except Exception:
            metrics['clustering'] = None
        
        # Calculate community modularity using Louvain method
        try:
            partition = community.best_partition(G_conn, weight='weight', random_state=0)
            metrics['modularity'] = community.modularity(partition, G_conn, weight='weight')
        except Exception:
            metrics['modularity'] = None
        
        # Calculate degree statistics (average and standard deviation)
        try:
            degrees = [d for _, d in G_conn.degree(weight='weight')]
            metrics['average_degree'] = np.mean(degrees)
            metrics['degree_std'] = np.std(degrees)
        except Exception:
            metrics['average_degree'] = None
            metrics['degree_std'] = None
        

        # Create distance matrix where distance = 1 - absolute weight (for non-zero connections)
        distance_matrix = np.where(conn_matrix != 0, 1 - np.abs(conn_matrix), np.inf)
        # Ensure distance matrix is symmetric
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        # Create networkx graph from distance matrix
        G_dist = nx.from_numpy_array(distance_matrix)
        
        # Calculate average shortest path length for connected graphs
        try:
            if nx.is_connected(G_dist):
                metrics['average_shortest_path'] = nx.average_shortest_path_length(G_dist, weight='weight')
            else:
                metrics['average_shortest_path'] = float('inf')
                
        except Exception:
            metrics['average_shortest_path'] = None
        
        # Calculate global efficiency (average of inverse shortest paths)
        try:
            total_eff = 0
            for node in G_dist.nodes():
                # Get shortest paths from current node to all others
                lengths = nx.single_source_dijkstra_path_length(G_dist, node, weight='weight')
                for target, d in lengths.items():
                    if node != target:
                        total_eff += 1 / d
            
            # Normalize by total number of node pairs
            metrics['global_efficiency'] = total_eff / (n * (n - 1))
        except Exception:
            metrics['global_efficiency'] = None
        
        # Clean up to free memory
        del G_conn, G_dist, matrix_arr
        gc.collect()
        
        return metrics
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        # Return dictionary with all None values if main calculation fails
        return {
            'clustering': None, 'modularity': None, 'average_degree': None,
            'degree_std': None, 'average_shortest_path': None, 'global_efficiency': None
        }
    
def compute_perhead_graph_measures(aggregation_results, file_path, 
                                  save_to_file=False, num_workers=None, max_heads_per_batch=4):
    """Calculate graph measures for each attention head across all layers
    
    Processes attention aggregation results to compute graph metrics for each head
    using parallel processing for efficiency. Results can be saved to a JSON file.
    
    Args:
        aggregation_results (dict): Dictionary containing attention matrices organized by layer
        file_path (str): Path to save results if save_to_file is True
        save_to_file (bool): Whether to save results to a file
        num_workers (int): Number of worker processes for parallel computing (defaults to min(cpu_count, 8))
        max_heads_per_batch (int): Maximum number of heads to process in each batch
        
    Returns:
        dict: Dictionary containing graph measures for each head, organized by layer
    """
    graph_measure = {}
    
    # Set default number of workers if not specified
    if num_workers is None:
        num_workers = min(os.cpu_count(), 8)
    
    # Process each layer's attention heads
    for layer, aggregation_result in aggregation_results.items():
        print(f"Processing layer {layer} with {num_workers} workers...")
        start_time = time.time()
        
        total_heads = len(aggregation_result)
        layer_graph_measures = []
        
        # Process heads in batches to manage memory and parallel processing
        for batch_start in range(0, total_heads, max_heads_per_batch):
            batch_end = min(batch_start + max_heads_per_batch, total_heads)
            batch_matrices = aggregation_result[batch_start:batch_end]
            
            print(f"Processing heads {batch_start} to {batch_end-1} of layer {layer}")
            batch_start_time = time.time()
            
            # Use process pool executor for parallel computation
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks for the current batch
                tasks = [executor.submit(compute_metrics_for_head, matrix)
                         for matrix in batch_matrices]
                
                # Collect results as they complete
                for future in as_completed(tasks):
                    try:
                        result = future.result()
                        layer_graph_measures.append(result)
                    except Exception as e:
                        print(f"Error in parallel task: {str(e)}")
                        # Append default None values for failed calculations
                        layer_graph_measures.append({
                            'clustering': None, 'modularity': None, 'average_degree': None,
                            'degree_std': None, 'average_shortest_path': None, 'global_efficiency': None
                        })
            
            print(f"Batch processed in {time.time() - batch_start_time:.2f}s")
            # Clean up memory after each batch
            gc.collect()
        
        # Store results for the current layer
        graph_measure[layer] = layer_graph_measures
        print(f"Layer {layer} completed in {time.time() - start_time:.2f} seconds")
    
    # Save results to file if requested
    if save_to_file:
        with open(file_path, 'w') as f:
            json.dump(graph_measure, f, indent=4)
        print(f"Graph measures saved to {file_path}")
    
    return graph_measure

def process_single_model(model, model_idx, num_gpus, get_attention_graph_func, 
                        base_dir=None, cache_dir=None, **kwargs):
    """Function to independently process a single model, intended for multiprocessing
    
    Args:
        model (str): Name of the model to process
        model_idx (int): Index identifier for the model in the processing queue
        num_gpus (int): Total number of available GPUs for processing
        get_attention_graph_func (callable): Function to compute attention graphs
        base_dir (str): Base directory for saving results
        cache_dir (str): Directory for caching pretrained models
        **kwargs: Additional arguments to pass to get_attention_graph_func
        
    Returns:
        bool: True if processing completes successfully, False otherwise
    """
    try:
        print(f"\n{'='*50}")
        print(f"Process for model {model} (index {model_idx}) started")
        print(f"{'='*50}")
        
        # Generate safe filename by replacing problematic characters
        model_safe_name = model.replace('.', '_').replace('/', '_')
        agg_path = f'{base_dir}/{model_safe_name}_agg.json'
        metrics_path = f'{base_dir}/{model_safe_name}.json'
        
        # Create base directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        # Check if aggregation results already exist
        if os.path.exists(agg_path):
            print(f"Process {model_idx}: Loading existing aggregation for {model}")
            with open(agg_path, 'r') as f:
                Aggregation_result = json.load(f)
        else:
            print(f"Process {model_idx}: Computing aggregation for {model}")
            # Allocate independent GPU for current process's model
            if num_gpus > 0:
                gpu_id = model_idx % num_gpus  # Independent GPU allocation per process
                device = torch.device(f'cuda:{gpu_id}')
                print(f"Process {model_idx}: Using GPU {gpu_id} (device: {device}) for model {model}")
            else:
                device = torch.device('cpu')
                print(f"Process {model_idx}: No GPUs detected, using CPU")
            
            # Calculate aggregation results using the provided function
            Aggregation_result = get_attention_graph_func(
                model, 
                save_to_file=False,
                file_path=agg_path,
                cache_dir=cache_dir,
                device=device,
                **kwargs  # Pass any additional arguments
            )
        
        # Calculate graph metrics
        print(f"Process {model_idx}: Computing graph measures for {model}")
        graph_measure = compute_perhead_graph_measures(
            Aggregation_result,
            file_path=metrics_path,
            save_to_file=True,
            num_workers=min(4, os.cpu_count()),  
            max_heads_per_batch=4
        )
        
        # Clean up resources for current process
        del Aggregation_result, graph_measure
        gc.collect()
        if num_gpus > 0:
            torch.cuda.empty_cache()  # Clear GPU cache used by current process
        
        print(f"Process {model_idx}: Completed processing {model}")
        return True
    
    except Exception as e:
        print(f"Process {model_idx}: Error processing {model}: {str(e)}")
        return False
        
def get_gpt2_pos_embedding(seq_len=50, target_dim=None, device=None):
    """
    Extract position embeddings from GPT-2
    seq_len: Required sequence length
    target_dim: Target dimension (will project if different from GPT-2's hidden dimension)
    """
    # Load GPT-2 model (using smallest version, can replace with "gpt2-medium" etc. for other versions)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_pos_emb = gpt2.transformer.wpe.weight  # GPT-2 position embedding weights [max_seq_len, hidden_size]
    gpt2_hidden_size = gpt2_pos_emb.shape[1]
    
    # Take the first seq_len position embeddings
    if seq_len > gpt2_pos_emb.shape[0]:
        raise ValueError(f"GPT-2 maximum sequence length is {gpt2_pos_emb.shape[0]}, cannot satisfy {seq_len} requirement")
    pos_emb = gpt2_pos_emb[:seq_len, :].detach()  # [50, gpt2_hidden_size]
    
    # Interpolate along the feature dimension (keep sequence length unchanged, adjust feature dimension)
    if target_dim is not None and target_dim != gpt2_hidden_size:
        # Original shape: [seq_len, gpt2_hidden_size]
        # Step 1: Merge seq_len and gpt2_hidden_size into one dimension -> [seq_len * gpt2_hidden_size]
        flat_emb = pos_emb.view(-1)  # Shape: [seq_len * gpt2_hidden_size]
        
        # Step 2: Add batch and channel dimensions -> 3D format [1, 1, seq_len * gpt2_hidden_size]
        flat_emb_3d = flat_emb.unsqueeze(0).unsqueeze(0)
        
        # Step 3: Use linear interpolation to target total dimension (seq_len * target_dim)
        target_total_dim = seq_len * target_dim
        interpolated = torch.nn.functional.interpolate(
            flat_emb_3d,
            size=(target_total_dim,),  # Target total length: seq_len * target_dim
            mode='linear',
            align_corners=False
        )
        
        # Step 4: Remove temporary dimensions and reshape to [seq_len, target_dim]
        pos_emb = interpolated.squeeze(0).squeeze(0).view(seq_len, target_dim)
    return pos_emb.to(device)

def get_vit_base_pos_embedding(seq_len=197, target_dim=None, device=None):
    """
    Get ViT-Base position embeddings adapted for length 197 (224x224 resolution, 16x16 patches)
    
    Summary:
    - Loads pretrained ViT-Base model and extracts position embeddings
    - Handles dimension adjustment if target_dim differs from original 768
    - Validates expected sequence length of 197 (includes CLS token)
    - Returns position embeddings on specified device
    """
    # Load pretrained ViT-Base model to get position embeddings
    vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
    vit_pos_emb = vit.embeddings.position_embeddings  # (ViT-Base position embeddings)
    if vit_pos_emb.dim() == 3 and vit_pos_emb.shape[0] == 1:
        vit_pos_emb = vit_pos_emb.squeeze(0)  # Transform from [1, 197, 768] to [197, 768]

    vit_hidden_size = vit_pos_emb.shape[1]  # 768
    
    # Verify ViT-Base position embedding length is 197 (including CLS token)
    if vit_pos_emb.shape[0] != 197:
        raise ValueError(f"ViT-Base position embedding length is {vit_pos_emb.shape[0]}, does not match expected 197")
    
    pos_emb = vit_pos_emb.detach()  # [197, 768]
    
    # If needed, adjust position embedding dimension to target dimension (image encoder's hidden_size)
    if target_dim is not None and target_dim != vit_hidden_size:
        flat_emb = pos_emb.view(-1)  # [197*768]
        flat_emb_3d = flat_emb.unsqueeze(0).unsqueeze(0)  # Adapt to interpolation function input shape
        target_total_dim = seq_len * target_dim
        interpolated = torch.nn.functional.interpolate(
            flat_emb_3d,
            size=(target_total_dim,),
            mode='linear',
            align_corners=False
        )
        pos_emb = interpolated.squeeze(0).squeeze(0).view(seq_len, target_dim)
    
    return pos_emb.to(device)

def process_single_model_plus(model_name, model_idx, gpu_ids, get_attention_graph_func, base_dir=None, cache_dir=None, **kwargs):
    """
    Assign dedicated multiple GPUs for each model
    model_name: Name of the model
    model_idx: Index of the model
    gpu_ids: Dedicated GPU list for this model (e.g., [0,1])
    """
    try:
        print(f"\n{'='*50}")
        print(f"Process for model {model_name} (index {model_idx}) started")
        print(f"Assigned GPUs: {gpu_ids}")
        print(f"{'='*50}")
        
        # Generate safe filename
        model_safe_name = model_name.replace('/', '_').replace('.', '_')
        agg_path = f'{base_dir}/{model_safe_name}_agg.json'
        metrics_path = f'{base_dir}/{model_safe_name}.json'
        
        os.makedirs(base_dir, exist_ok=True)
        
        # Check if aggregation results already exist
        if os.path.exists(agg_path):
            print(f"Process {model_idx}: Loading existing aggregation for {model_name}")
            with open(agg_path, 'r') as f:
                Aggregation_result = json.load(f)
        else:
            print(f"Process {model_idx}: Computing aggregation for {model_name}")
            
            # Configure visible GPUs for current process (only expose specified GPUs)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
            # # Verify visibility
            # visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            # print(f"Process {model_idx}: CUDA_VISIBLE_DEVICES set to {visible_gpus}")
            
            # Device mapping strategy: Evenly distribute model layers across specified GPUs
            device_map = "balanced"  # Automatically balance across visible GPUs
            
            # Compute aggregation results
            Aggregation_result = get_attention_graph_func(
                model_name, 
                save_to_file=False,
                file_path=agg_path,
                cache_dir=cache_dir,
                device_map=device_map,
                **kwargs
            )
        
        # Compute graph metrics
        print(f"Process {model_idx}: Computing graph measures for {model_name}")
        graph_measure = compute_perhead_graph_measures(
            Aggregation_result,
            file_path=metrics_path,
            save_to_file=True,
            num_workers=min(4, os.cpu_count()),
            max_heads_per_batch=4
        )
        
        # Clean up resources
        del Aggregation_result, graph_measure
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"Process {model_idx}: Completed processing {model_name}")
        return True
    
    except Exception as e:
        print(f"Process {model_idx}: Error processing {model_name}: {str(e)}")
        return False
        
"""File to run GPT4Point inference programmatically for batch evaluation.

Adapted from PointLLM inference code to work with GPT4Point's LAVIS-based architecture.
Uses GPT4Point's proper model loading, text processing, and generation parameters.
"""


# GPT4Point imports
from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.models import load_model
from lavis.datasets.transforms.transforms_point import pc_norm_with_color

# Suppress transformers logging errors
import logging
import argparse
import torch
import os
import numpy as np
import json
import open3d as o3d
logging.getLogger('transformers').setLevel(logging.ERROR)



class FakeUpload:
    """
    A simple container class to mimic file upload objects for point cloud inference.
    The original PointLLM gradio code this file is based on only ever used file
    upload objects, so this class substitutes for that to allow programmatic evaluation that does not require those file upload objects that come from the gradio UI.
    """
    def __init__(self, path, hex, scene_name):
        self.name = path
        self.hex = hex
        self.scene_name = scene_name

def make_named_upd_txt_files(identifier_at_scene_list, updtext_versionfolder_subfolder_path):
    """
    Returns a list of file paths for upd text samples based on the provided subfolder and scenes.
    """
    return [os.path.join(updtext_versionfolder_subfolder_path, name + ".txt") for name in identifier_at_scene_list]

def make_named_ply_files(identifier_at_scene_list, unzipped_point_cloud_path):
    """
    Returns a list of FakeUpload objects, each representing a point cloud file.
    Tries new path format first (without extra folder), then falls back to old format.
    """
    results = []
    new_format_count = 0
    old_format_count = 0
    missing_count = 0
    
    for identifier_at_scene in identifier_at_scene_list:
        identifier, scene = identifier_at_scene.split('@')
        
        # Try new path format first (without extra folder): /path/identifier/scene.ply
        new_path = os.path.join(unzipped_point_cloud_path, identifier, scene + ".ply")
        
        # Fall back to old path format (with extra folder): /path/identifier/scene/scene.ply
        old_path = os.path.join(unzipped_point_cloud_path, identifier, scene, scene + ".ply")
        
        # Check which path exists and use that one
        if os.path.exists(new_path):
            results.append(FakeUpload(new_path, identifier, scene))
            new_format_count += 1
        elif os.path.exists(old_path):
            results.append(FakeUpload(old_path, identifier, scene))
            old_format_count += 1
        else:
            # If neither exists, use new path format as default (will fail later with clear error)
            results.append(FakeUpload(new_path, identifier, scene))
            missing_count += 1
    
    print(f"[INFO] Path format summary: {new_format_count} new format, {old_format_count} old format, {missing_count} missing files", flush=True)
    return results

def load_point_cloud(file_path, debug_mode=False):
    """Load a point cloud from a PLY file and process it for GPT4Point input."""
    if debug_mode:
        print(f"[DEBUG] Loading point cloud from: {file_path}", flush=True)
    
    if not os.path.exists(file_path):
        print(f"[ERROR] Point cloud file does not exist: {file_path}", flush=True)
        return None
    
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)  # xyz
        colors = np.asarray(pcd.colors)  # rgb, if available
        
        if debug_mode:
            print(f"[DEBUG] Raw point cloud - points: {points.shape}, colors: {colors.shape}", flush=True)
        
        if points.size == 0:
            print(f"[ERROR] Point cloud has no points: {file_path}", flush=True)
            return None
        
        # If no colors, create default white colors (matching GPT4Point dataset behavior)
        if colors.size == 0:
            white_colors = np.ones_like(points)  # White color [1.0, 1.0, 1.0]
            colors = white_colors
        
        # Ensure colors are in range [0, 1]
        if np.max(colors) > 1:
            colors = colors.astype(np.float32) / 255
        
        # Concatenate points and colors
        point_cloud = np.concatenate((points, colors), axis=1)  # (N, 6)
        
        # Downsample to 8192 points if needed (GPT4Point expects exactly 8192)
        if point_cloud.shape[0] > 8192:
            # Simple random sampling - GPT4Point training data uses random sampling
            indices = np.random.choice(point_cloud.shape[0], 8192, replace=False)
            point_cloud = point_cloud[indices]
        elif point_cloud.shape[0] < 8192:
            # Upsample by repeating points if we have fewer than 8192
            repeat_factor = 8192 // point_cloud.shape[0] + 1
            point_cloud = np.tile(point_cloud, (repeat_factor, 1))[:8192]
        
        # Use GPT4Point's normalization (matches the evaluation processor)
        point_cloud = pc_norm_with_color(point_cloud)
        
        # Debug: Check point cloud before final conversion
        if debug_mode:
            print(f"[DEBUG] Point cloud after normalization - shape: {point_cloud.shape}, range: [{point_cloud.min():.4f}, {point_cloud.max():.4f}]", flush=True)
        
        point_cloud = torch.from_numpy(point_cloud).unsqueeze_(0).to(torch.float32).cuda()
        
        return point_cloud
    except Exception as e:
        print(f"[ERROR] Failed to load point cloud {file_path}: {e}", flush=True)
        return None

def inference(
        pcl_list_txt_file_path,
        updtext_versionfolder_subfolder_path,
        unzipped_point_cloud_path,
        upd_subset_name,
        model,
        json_tag=None
    ):
    """Perform batch inference on point clouds and prompts using GPT4Point."""
    with open(pcl_list_txt_file_path, 'r') as f:
        identifier_at_scene_list = f.read().splitlines()
    pcl_list_txt_filename_noext = os.path.basename(os.path.normpath(pcl_list_txt_file_path)).replace('.txt', '')

    print(f"[INFO] Found {len(identifier_at_scene_list)} samples to process", flush=True)

    pc_ply_list = make_named_ply_files(identifier_at_scene_list, unzipped_point_cloud_path)
    upd_txt_file_list = make_named_upd_txt_files(identifier_at_scene_list, updtext_versionfolder_subfolder_path)

    if len(pc_ply_list) == 0:
        print("[ERROR] No point cloud files found!", flush=True)
        return

    results = {}  # Dictionary to store all results
    
    total_samples = len(pc_ply_list)
    for idx, (ply_file, txt_file) in enumerate(zip(pc_ply_list, upd_txt_file_list), 1):
        try:
            print(f"[PROGRESS] Processing sample {idx}/{total_samples}: {ply_file.hex}@{ply_file.scene_name}", flush=True)
            
            # Read prompt (UPD-3D text files contain complete prompts, no preprocessing needed)
            with open(txt_file, 'r') as f:
                prompt = f.read().strip()

            # Load and process point cloud (enable debug for first few samples)
            debug_mode = idx <= 3  # Only debug first 3 samples
            point_clouds = load_point_cloud(ply_file.name, debug_mode)
            if point_clouds is None:
                print(f"[ERROR] Failed to load point cloud: {ply_file.name}", flush=True)
                continue
            
            # Debug: Verify point cloud data (only for first few samples)
            if debug_mode:
                print(f"[DEBUG] Point cloud shape: {point_clouds.shape}, dtype: {point_clouds.dtype}, device: {point_clouds.device}", flush=True)
                print(f"[DEBUG] Point cloud min: {point_clouds.min():.4f}, max: {point_clouds.max():.4f}, mean: {point_clouds.mean():.4f}", flush=True)
                print(f"[DEBUG] First few points: {point_clouds[0, :3, :3]}", flush=True)
            
            # Create sample format expected by GPT4Point
            samples = {
                "point": point_clouds,
                "text_input": [prompt]  # Use raw prompt text directly
            }
            
            # Debug: Verify model input format (only for first few samples)
            if debug_mode:
                print(f"[DEBUG] Model input - point shape: {samples['point'].shape}, text: '{samples['text_input'][0][:100]}...'", flush=True)

            # Generate response using GPT4Point's generate method with evaluation parameters
            with torch.inference_mode():
                output_texts = model.generate(
                    samples,
                    use_nucleus_sampling=False,
                    num_beams=5,  # Matches GPT4Point eval config
                    max_length=30,  # Matches GPT4Point eval config
                    min_length=8,   # Matches GPT4Point eval config
                    temperature=1.0,
                    num_captions=1
                )
            
            # Extract the output text
            if isinstance(output_texts, list) and len(output_texts) > 0:
                outputs = output_texts[0].strip()
            else:
                outputs = str(output_texts).strip()

            # Store result
            results[ply_file.hex + '@' + ply_file.scene_name.split(".")[0]] = {
                "prompt": prompt, 
                "response": outputs
            }

            print(f"[INFO] Processed {ply_file.hex}@{ply_file.scene_name}: {outputs}", flush=True)

        except Exception as e:
            print(f"[ERROR] Failed to process pair ({txt_file}, {ply_file.name}): {e}", flush=True)

    # Write all results to a JSON file
    try:
        tag_part = f"{json_tag}_" if json_tag else ""
        json_filename = f'inf_rslts_gpt4point_{tag_part}{pcl_list_txt_filename_noext}_{upd_subset_name}.json'
        with open(json_filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results saved to {json_filename}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to write results to JSON file: {e}", flush=True)

def init_model(args):
    """Initialize the GPT4Point model for batch inference using proper LAVIS framework."""
    model_path = os.path.expanduser(args.model_name)
    
    print(f'[INFO] Loading GPT4Point model from: {model_path}', flush=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEBUG] Using device: {device}", flush=True)
    
    if torch.cuda.is_available():
        print(f"[DEBUG] GPU memory before model loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated", flush=True)
        print(f"[DEBUG] GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total", flush=True)
    
    try:
        # Load model using LAVIS registry system - matches GPT4Point evaluation approach
        model = load_model(
            name="gpt4point_opt", 
            model_type="gpt4point_opt2.7b",
            is_eval=True,
            device=device
        )
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            if os.path.isfile(model_path):
                print(f'[INFO] Loading checkpoint from: {model_path}', flush=True)
                try:
                    # Try loading with strict=True first
                    model.load_checkpoint(model_path)
                    print(f'[INFO] Checkpoint loaded successfully (strict mode)', flush=True)
                except Exception as e:
                    print(f'[WARNING] Strict checkpoint loading failed: {e}', flush=True)
                    print('[INFO] Attempting to load checkpoint with strict=False (allowing missing keys)...', flush=True)
                    try:
                        # Load checkpoint manually with strict=False
                        checkpoint = torch.load(model_path, map_location='cpu')
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                        
                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        
                        if missing_keys:
                            print(f'[WARNING] Missing keys in checkpoint: {len(missing_keys)} keys', flush=True)
                            print(f'[DEBUG] First 10 missing keys: {missing_keys[:10]}', flush=True)
                        
                        if unexpected_keys:
                            print(f'[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} keys', flush=True)
                            print(f'[DEBUG] First 10 unexpected keys: {unexpected_keys[:10]}', flush=True)
                        
                        print(f'[INFO] Checkpoint loaded successfully (non-strict mode)', flush=True)
                        
                    except Exception as e2:
                        print(f'[ERROR] Failed to load checkpoint even with strict=False: {e2}', flush=True)
                        import traceback
                        print(f'[ERROR] Non-strict loading traceback: {traceback.format_exc()}', flush=True)
                        raise
            elif os.path.isdir(model_path):
                # Try common checkpoint names
                checkpoint_files = [
                    "pytorch_model.bin", 
                    "model.pth", 
                    "checkpoint.pth",
                    "checkpoint_best.pth"
                ]
                checkpoint_loaded = False
                for ckpt_file in checkpoint_files:
                    ckpt_path = os.path.join(model_path, ckpt_file)
                    if os.path.exists(ckpt_path):
                        print(f'[INFO] Loading checkpoint from: {ckpt_path}', flush=True)
                        model.load_checkpoint(ckpt_path)
                        checkpoint_loaded = True
                        break
                if not checkpoint_loaded:
                    print(f'[WARNING] No checkpoint found in directory: {model_path}', flush=True)
        else:
            print(f'[WARNING] Model path does not exist: {model_path}', flush=True)
            print('[INFO] Using default pretrained model', flush=True)
        
        print("[DEBUG] Setting model to eval mode...", flush=True)
        model.eval()
        print("[DEBUG] Model eval mode set", flush=True)
        
        print("[INFO] Model loaded successfully", flush=True)
        print(f"[DEBUG] Model device: {next(model.parameters()).device}", flush=True)
        print(f"[DEBUG] Model parameters count: {sum(p.numel() for p in model.parameters()):,}", flush=True)
        
        # Check GPU memory usage after model loading
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB", flush=True)
            print(f"[DEBUG] GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB", flush=True)
            print(f"[DEBUG] GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", flush=True)
        
        return model
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", flush=True)
        import traceback
        print(f"[ERROR] Full traceback: {traceback.format_exc()}", flush=True)
        raise

def existing_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"readable_dir: '{path}' is not a valid directory")
    return path

def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"readable_file: '{path}' is not a valid file")
    return path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Programmatic evaluation code for GPT4Point")
    parser.add_argument("--model_name", type=str, default="./lavis/output/GPT4Point/", help="Path to the GPT4Point model checkpoint")
    parser.add_argument("--upd_text_folder_path", type=existing_dir, required=True, 
                        help="Path to the upd_text/ folder.")
    parser.add_argument("--upd_version_name", type=str, required=False, 
                        help="Name of the upd version (e.g., 'v1').", default="3D-FRONT")
    parser.add_argument("--upd_version_name_subfolder", type=str, required=True, 
                        help="Subfolder name for the upd version (e.g., 'standard').")
    parser.add_argument("--unzipped_point_cloud_path", type=existing_dir, required=True, 
                        help="Path to the unzipped point cloud folder containing dirs identifier/scene/scene.ply")
    parser.add_argument("--pcl_list_txt_file_path", type=existing_file, required=True, 
                        help="Path to the text file containing point cloud identifiers and scene names.")
    parser.add_argument("--json_tag", type=str, required=False, 
                        help="Optional tag to include in the output JSON filename, eg 'ft-comb' for finetune-combined", 
                        default=None)
    args = parser.parse_args()

    # Check that the passed paths are valid
    updtext_versionfolder_subfolder_path = os.path.join(
        args.upd_text_folder_path,
        args.upd_version_name,
        args.upd_version_name_subfolder
    )
    if not os.path.isdir(updtext_versionfolder_subfolder_path):
        raise ValueError(f"Error: '{updtext_versionfolder_subfolder_path}' is not a valid folder path.")

    # Initialize model
    print("[DEBUG] Starting model initialization...", flush=True)
    model = init_model(args)
    print("[DEBUG] Model initialization completed", flush=True)
    
    # Quick model test
    print("[DEBUG] Testing model with dummy input...", flush=True)
    try:
        dummy_points = torch.randn(1, 8192, 6).cuda()
        dummy_samples = {"point": dummy_points, "text_input": ["test"]}
        with torch.no_grad():
            _ = model.generate(dummy_samples, max_length=10, num_beams=1)
        print("[DEBUG] Model test successful", flush=True)
    except Exception as e:
        print(f"[ERROR] Model test failed: {e}", flush=True)
        raise

    # Run batch inference
    print("[DEBUG] Starting inference...", flush=True)
    inference(
        pcl_list_txt_file_path=args.pcl_list_txt_file_path,
        updtext_versionfolder_subfolder_path=updtext_versionfolder_subfolder_path,
        unzipped_point_cloud_path=args.unzipped_point_cloud_path,
        upd_subset_name=args.upd_version_name_subfolder,
        model=model,
        json_tag=args.json_tag
    )
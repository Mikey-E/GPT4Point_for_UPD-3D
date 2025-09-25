"""File to run GPT4Point inference programmatically for batch evaluation.

Adapted from PointLLM inference code to work with GPT4Point's LAVIS-based architecture.
Uses GPT4Point's proper model loading, text processing, and generation parameters.
"""


import argparse\nimport torch\nimport os\nimport numpy as np\nimport json\nimport open3d as o3d

# GPT4Point imports
from lavis.common.config import Config
from lavis.common.registry import registry
from lavis.models import load_model
from lavis.processors import load_processor
from lavis.datasets.transforms.transforms_point import pc_norm_with_color

# Suppress transformers logging errors
import logging
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

def load_point_cloud(file_path):
    """Load a point cloud from a PLY file and process it for GPT4Point input."""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)  # xyz
        colors = np.asarray(pcd.colors)  # rgb, if available
        
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
        text_processor,
        json_tag=None
    ):
    """Perform batch inference on point clouds and prompts using GPT4Point."""
    with open(pcl_list_txt_file_path, 'r') as f:
        identifier_at_scene_list = f.read().splitlines()
    pcl_list_txt_filename_noext = os.path.basename(os.path.normpath(pcl_list_txt_file_path)).replace('.txt', '')

    pc_ply_list = make_named_ply_files(identifier_at_scene_list, unzipped_point_cloud_path)
    upd_txt_file_list = make_named_upd_txt_files(identifier_at_scene_list, updtext_versionfolder_subfolder_path)

    results = {}  # Dictionary to store all results
    
    total_samples = len(pc_ply_list)
    for idx, (ply_file, txt_file) in enumerate(zip(pc_ply_list, upd_txt_file_list), 1):
        try:
            print(f"[PROGRESS] Processing sample {idx}/{total_samples}: {ply_file.hex}@{ply_file.scene_name}", flush=True)
            
            # Read prompt
            with open(txt_file, 'r') as f:
                prompt = f.read().strip()

            # Load and process point cloud
            point_clouds = load_point_cloud(ply_file.name)
            if point_clouds is None:
                print(f"[ERROR] Failed to load point cloud: {ply_file.name}", flush=True)
                continue

            # Process text input using GPT4Point's text processor (matches evaluation setup)
            processed_text = text_processor(prompt)
            
            # Create sample format expected by GPT4Point
            samples = {
                "point": point_clouds,
                "text_input": [processed_text]  # GPT4Point expects a list
            }

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
                model.load_checkpoint(model_path)
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
        
        model.eval()
        
        # Load text processor to match GPT4Point's evaluation setup
        text_processor = load_processor("blip_caption", 
                                      cfg={"prompt": "a 3D point cloud of "})
        
        return model, text_processor
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", flush=True)
        raise

def existing_dir(path):
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"readable_dir: '{path}' is not a valid directory")
    return path

def existing_file(path):
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"readable_file: '{path}' is not a valid file")
    return path

if __name__ == "__main__":\n    parser = argparse.ArgumentParser(description="Programmatic evaluation code for GPT4Point")\n    parser.add_argument("--model_name", type=str, default="./lavis/output/GPT4Point/",\n                        help="Path to the GPT4Point model checkpoint")
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
    model, text_processor = init_model(args)

    # Run batch inference
    inference(
        pcl_list_txt_file_path=args.pcl_list_txt_file_path,
        updtext_versionfolder_subfolder_path=updtext_versionfolder_subfolder_path,
        unzipped_point_cloud_path=args.unzipped_point_cloud_path,
        upd_subset_name=args.upd_version_name_subfolder,
        model=model,
        text_processor=text_processor,
        json_tag=args.json_tag
    )
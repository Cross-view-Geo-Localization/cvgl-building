import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.registry import build_model
from utils.predict import ForwardMode
from data.transforms import get_transforms_val
from models.supersalad import SuperSALADNetwork

class ImageRetriever:
    def __init__(self, config_path):
        """
        Initializes the retriever using the exact configuration used during training.
        """
        print(f"Loading configuration from {config_path}...")
        self.config = OmegaConf.load(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Build Model
        self.model = build_model(self.config)
        
        # 2. Load Checkpoint (if specified in config)
        if self.config.training.checkpoint_start is not None:
            print(f"Loading checkpoint: {self.config.training.checkpoint_start}")
            ckpt = torch.load(self.config.training.checkpoint_start, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            
        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. Setup Transforms based on Model Config
        data_cfg = self.model.get_config()
        query_mean, query_std = data_cfg["query"]["mean"], data_cfg["query"]["std"]
        ref_mean, ref_std = data_cfg["reference"]["mean"], data_cfg["reference"]["std"]
        
        drone_size = (self.config.data.drone_img_size, self.config.data.drone_img_size) \
                     if isinstance(self.config.data.drone_img_size, int) \
                     else tuple(self.config.data.drone_img_size)
                     
        sate_size = (self.config.data.sat_img_size, self.config.data.sat_img_size) \
                    if isinstance(self.config.data.sat_img_size, int) \
                    else tuple(self.config.data.sat_img_size)

        # Generate validation transforms using your custom function
        self.val_drone_tf, self.val_sate_tf = get_transforms_val(
            image_size_sat=sate_size,
            image_size_drone=drone_size,
            query_mean=query_mean, query_std=query_std,
            ref_mean=ref_mean, ref_std=ref_std,
        )
        
        self.gallery_features = None
        self.gallery_paths = []

    def extract_feature(self, img_path, is_query=True):
        """
        Extracts features. Uses drone transforms for queries and satellite transforms for references.
        """
        img = Image.open(img_path).convert('RGB')
        
        # 1. Convert PIL Image to NumPy array for Albumentations
        img_np = np.array(img)
        
        # Apply the correct transform based on the image domain
        transform = self.val_drone_tf if is_query else self.val_sate_tf
        
        # 2. Pass as keyword argument 'image' and extract the 'image' key from the result dict
        transformed_data = transform(image=img_np)
        img_tensor = transformed_data["image"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(img_tensor, mode=ForwardMode.QUERY if is_query else ForwardMode.REFERENCE)
            
            # Flatten and L2 normalize for cosine similarity matching
            feature = feature.view(feature.size(0), -1)
            feature = F.normalize(feature, p=2, dim=1)
            
        return feature

    def build_gallery(self, reference_folder, cache_path):
        """Processes all satellite images to build the offline gallery."""
        if os.path.exists(cache_path):
            print(f"Loading gallery from cache: {cache_path}")
            cache = torch.load(cache_path, map_location=self.device)
            
            self.gallery_features = cache['features'].to(self.device)
            self.gallery_paths = cache['paths']
            
            print(f"Successfully loaded {len(self.gallery_paths)} images from cache.")
            return
        
        print(f"Building gallery from {reference_folder}...")
        features_list = []
        
        for fname in sorted(os.listdir(reference_folder)):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                img_path = os.path.join(reference_folder, fname)
                self.gallery_paths.append(img_path)
                
                # is_query=False ensures we use val_sate_tf
                feat = self.extract_feature(img_path, is_query=False) 
                features_list.append(feat)
                
        self.gallery_features = torch.cat(features_list, dim=0)
        print(f"Gallery built with {len(self.gallery_paths)} images.")

        print(f"Saving gallery to cache: {cache_path}...")
        torch.save({
            'features': self.gallery_features.cpu(), 
            'paths': self.gallery_paths
        }, cache_path)
        print("Save complete.")

    def retrieve(self, query_path, top_k=5):
        """Finds the top-K matches in the gallery for a given drone query image."""
        if self.gallery_features is None:
            raise ValueError("Gallery not built. Call build_gallery() first.")
            
        # is_query=True ensures we use val_drone_tf
        query_feat = self.extract_feature(query_path, is_query=True) 
        
        # Matrix multiplication yields Cosine Similarity (since features are L2 normalized)
        similarities = torch.mm(query_feat, self.gallery_features.t()).squeeze()
        
        topk_scores, topk_indices = torch.topk(similarities, k=top_k)
        
        results = []
        for score, idx in zip(topk_scores, topk_indices):
            results.append({
                'path': self.gallery_paths[idx.item()],
                'score': score.item()
            })
            
        return results

    def visualize_results(self, query_path, results, output_path="retrieval_result.png"):
        """Plots the query image on the left and the top-K retrieved satellite images on the right."""
        top_k = len(results)
        fig, axes = plt.subplots(1, top_k + 1, figsize=(15, 4))
        
        query_img = Image.open(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title("Query (Drone)", fontweight="bold")
        axes[0].axis('off')
        
        for i, res in enumerate(results):
            img = Image.open(res['path'])
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Rank {i+1}\nSim: {res['score']:.3f}")
            axes[i + 1].axis('off')
            
        plt.tight_layout()
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved to: {output_path}")

        plt.close(fig)


if __name__ == "__main__":
    CONFIG_PATH = "/home/tts26/sonh/DroneCVGL/config/visloc_sinkhorn.yaml"
    
    retriever = ImageRetriever(CONFIG_PATH)

    REFERENCE_DIR = "/home/tts26/sonh/data/UAV1/tile/images"
    DATABASE_CACHE_PATH = "/home/tts26/sonh/DroneCVGL/retrieval_result/database_cache.pt"
    QUERY_IMAGE_PATH = "/home/tts26/sonh/data/UAV1/drone/DJI_20260128125205_0462_D_0981818161/DJI_20260128125205_0462_D_0981818161_120.jpg"

    # 1. Build gallery (using satellite transforms)
    retriever.build_gallery(REFERENCE_DIR, DATABASE_CACHE_PATH)

    # 2. Retrieve & Visualize (using drone transforms for the query)
    top_results = retriever.retrieve(QUERY_IMAGE_PATH, top_k=5)
    SAVE_PATH = "/home/tts26/sonh/DroneCVGL/retrieval_result/images"
    retriever.visualize_results(QUERY_IMAGE_PATH, top_results, SAVE_PATH)
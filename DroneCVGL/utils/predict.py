import time
import torch
from tqdm import tqdm
from torch.amp import autocast
import torch.nn.functional as F
from enum import Enum


class ForwardMode(Enum):
    TRAIN = 'train'
    QUERY = 'query'
    REFERENCE = 'reference'


def predict(config, model, dataloader, mode=ForwardMode.QUERY):
    train_config = config.training
    eval_config = config.eval
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with autocast(device_type="cuda", enabled=train_config.mixed_precision):
         
                img = img.to(train_config.device)
                img_feature = model(img, mode=mode)
            
                # normalize is calculated in fp32
                if eval_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
    
    return img_features, ids_list


def predict_query_rotations(config, model, dataloader, aggregation='keep_all', mode=ForwardMode.QUERY):
    train_config = config.training
    eval_config = config.eval
    model.eval()
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader), desc=f"Extracting Query Rotations ({aggregation})")
    else:
        bar = dataloader
        
    img_features_list = []
    ids_list = []
    
    with torch.no_grad():
        for img, ids in bar:
            ids_list.append(ids)
            B = img.size(0) # Batch size
            
            # 1. Generate 4 rotations for the entire batch
            img_0   = img
            img_90  = torch.rot90(img, k=1, dims=[2, 3])
            img_180 = torch.rot90(img, k=2, dims=[2, 3])
            img_270 = torch.rot90(img, k=3, dims=[2, 3])
            
            # 2. Combine into a single large batch: Shape [4 * B, C, H, W]
            query_batch = torch.cat([img_0, img_90, img_180, img_270], dim=0)
            
            with autocast(device_type="cuda", enabled=train_config.mixed_precision):
                query_batch = query_batch.to(train_config.device)
                features = model(query_batch, mode=mode)
                
                if eval_config.normalize_features:
                    features = F.normalize(features, dim=-1)
            
            # 3. Reshape back to group the 4 views per original image
            # features is [4*B, D] -> view as [4, B, D] -> permute to [B, 4, D]
            features = features.view(4, B, -1).permute(1, 0, 2)
            
            # 4. Apply Aggregation
            if aggregation == 'avg':
                features = features.mean(dim=1) # Average across the 4 rotations -> Shape: [B, D]
            elif aggregation == 'max':
                features, _ = features.max(dim=1) # Element-wise max across rotations -> Shape: [B, D]
            elif aggregation != 'keep_all':
                raise ValueError(f"Unknown aggregation method: {aggregation}. Use 'keep_all', 'avg', or 'max'.")
            
            # 5. Re-normalize after aggregation (Optional but highly recommended)
            # Taking the mean or max changes the vector's L2 norm. Re-normalizing ensures
            # proper cosine similarity calculation later.
            if aggregation in ['avg', 'max'] and eval_config.normalize_features:
                features = F.normalize(features, dim=-1)
            
            img_features_list.append(features.to(torch.float32))
  
    # Concatenate all batches 
    img_features = torch.cat(img_features_list, dim=0) 
    ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
    
    if train_config.verbose:
        bar.close()
    
    return img_features, ids_list
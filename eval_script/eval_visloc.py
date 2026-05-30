import os
import sys
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

# Thêm đường dẫn gốc của project vào sys.path nếu cần thiết
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.transforms import get_transforms_val
from data.visloc import VisLocDatasetEval
from core.metrics.visloc import evaluate as evaluate_visloc
from utils.registry import build_model

# Import các model để đăng ký (registry)
from models.siamese_network import SiameseNetwork                           
from models.siamese_network_max_avg import SiameseNetworkMaxAvg               
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork         

if __name__ == '__main__':
    #-----------------------------------------------------------------------------#
    # Config                                                                      #
    #-----------------------------------------------------------------------------#
    # Thay đổi config tại đây nếu cần
    config_path = "/home/tts26/sonh/DroneCVGL/config/visloc_sinkhorn.yaml"
    config = OmegaConf.load(config_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print(f"Evaluation Config: {config_path}")
    print("="*60)

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
    print("\nModel: {}".format(config.model.model_name))
    model = build_model(config)
                          
    data_config = model.get_config()
    print("Model Data Config:", data_config)
    
    query_mean, query_std = data_config['query']['mean'], data_config['query']['std']
    ref_mean, ref_std = data_config['reference']['mean'], data_config['reference']['std']
    
    drone_size = (config.data.drone_img_size, config.data.drone_img_size) \
                 if isinstance(config.data.drone_img_size, int) \
                 else tuple(config.data.drone_img_size)
    sate_size = (config.data.sat_img_size, config.data.sat_img_size) \
                if isinstance(config.data.sat_img_size, int) \
                else tuple(config.data.sat_img_size)

    # Load pretrained Checkpoint    
    if config.training.checkpoint_start is not None:
        print("Start from checkpoint:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: No checkpoint provided in config.training.checkpoint_start. Evaluating with random weights.")

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.training.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.training.gpu_ids)
            
    # Model to device   
    model = model.to(device)
    model.eval()

    print("\nImage Size Satellite (Ref):", sate_size)
    print("Image Size Drone (Query)  :", drone_size)

    #-----------------------------------------------------------------------------#
    # Transforms                                                                  #
    #-----------------------------------------------------------------------------#
    val_drone_tf, val_sate_tf = get_transforms_val(
        image_size_sat=sate_size,
        image_size_drone=drone_size,
        query_mean=query_mean, query_std=query_std,
        ref_mean=ref_mean,     ref_std=ref_std,
    )

    #-----------------------------------------------------------------------------#
    # Datasets & DataLoaders                                                      #
    #-----------------------------------------------------------------------------#
    
    # 1. Eval: query (drone)
    query_dataset_test = VisLocDatasetEval(
        pairs_meta_file=config.data.test_pairs_meta_file,
        data_root=config.data.data_folder,
        view="drone",
        mode="pos",
        transforms=val_drone_tf,
    )

    query_img_list = query_dataset_test.images_name
    pairs_drone2sate_dict = query_dataset_test.pairs_drone2sate_dict
    query_center_loc_xy_list = query_dataset_test.images_center_loc_xy

    query_loader = DataLoader(
        query_dataset_test,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # 2. Eval: gallery (satellite)
    gallery_dataset_test = VisLocDatasetEval(
        pairs_meta_file=config.data.test_pairs_meta_file,
        data_root=config.data.data_folder,
        view="sate",
        sate_img_dir=config.data.sate_img_dir,
        transforms=val_sate_tf,
    )

    gallery_img_list = gallery_dataset_test.images_name
    gallery_center_loc_xy_list = gallery_dataset_test.images_center_loc_xy
    gallery_topleft_loc_xy_list = gallery_dataset_test.images_topleft_loc_xy

    gallery_loader = DataLoader(
        gallery_dataset_test,
        batch_size=config.eval.batch_size_eval,
        num_workers=config.training.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    print("\nQuery Images (Drone)     :", len(query_dataset_test))
    print("Gallery Tiles (Satellite):", len(gallery_dataset_test))
   
    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#
    print(f"\n{30*'-'}[ Evaluate VisLoc ]{30*'-'}")

    results = evaluate_visloc(
        config=config,
        model=model,
        query_loader=query_loader,
        gallery_loader=gallery_loader, 
        query_list=query_img_list,
        gallery_list=gallery_img_list,
        query_center_loc_xy_list=query_center_loc_xy_list,
        gallery_center_loc_xy_list=gallery_center_loc_xy_list,
        gallery_topleft_loc_xy_list=gallery_topleft_loc_xy_list,
        pairs_dict=pairs_drone2sate_dict,
        ranks_list=[1, 5, 10],
        step_size=1000,
        cleanup=True
    )
    
    print("\nEvaluation Results:")
    print(results)
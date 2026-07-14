import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset # Import thêm ConcatDataset
from omegaconf import OmegaConf

sys.path.insert(0, "/home/tts26/sonh/DroneCVGL")

from data.transforms import get_transforms_val
from data.sues200 import SUES200DatasetEval, get_transforms
from core.metrics.sues200 import evaluate
from utils.registry import build_model
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork, AttentionSinkhornSiameseNetwork 
from models.siamese_network_max_avg import SiameseNetworkMaxAvg
from models.siamese_network import SiameseNetwork
from models.siamese_network_GeM import SiameseNetworkGeM
from models.dac import DAC
from models.supersalad import SuperSALADNetwork
from models.siamese_network_with_atttention import SiameseNetworkWithAttention

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = OmegaConf.load("./DroneCVGL/config/sinkhorn_siamese.yaml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Cập nhật đường dẫn tách biệt train và test
query_folder_train = './data/SUES-200-512x512/train/drone_view_512' 
query_folder_test  = './data/SUES-200-512x512/test/drone_view_512' 

ref_folder_train = './data/SUES-200-512x512/train/satellite-view'
ref_folder_test  = './data/SUES-200-512x512/test/satellite-view'
 
if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model.model_name))

    model = build_model(config)
                          
    data_config = model.get_config()
    print(data_config)
    query_mean, query_std = data_config['query']['mean'], data_config['query']['std']
    ref_mean, ref_std = data_config['reference']['mean'], data_config['reference']['std']

    query_img_size  = (config.data.drone_img_size, config.data.drone_img_size) \
                   if isinstance(config.data.drone_img_size, int) \
                   else tuple(config.data.drone_img_size)
    ref_img_size   = (config.data.sat_img_size,   config.data.sat_img_size) \
                   if isinstance(config.data.sat_img_size, int) \
                   else tuple(config.data.sat_img_size)
    

    # load pretrained Checkpoint    
    if config.training.checkpoint_start is not None:
        print("Start from:", config.training.checkpoint_start)
        ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)   

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.training.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.training.gpu_ids)
            
    # Model to device   
    model = model.to(device)

    print("\nImage Size Query:", ref_img_size)
    print("Image Size Ground:", query_img_size)
    print("Mean: {}".format(query_mean))
    print("Std:  {}\n".format(query_std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Transforms
    drone_transform_val, sat_transform_val = get_transforms_val(image_size_sat=ref_img_size, image_size_drone=query_img_size)
                                                                                 
    # 1. Khởi tạo Query Datasets (Drone)
    query_dataset_train_split = SUES200DatasetEval(data_folder=query_folder_train,
                                                   mode="drone",
                                                   transforms=drone_transform_val)
    
    query_dataset_test_split = SUES200DatasetEval(data_folder=query_folder_test,
                                                  mode="drone",
                                                  transforms=drone_transform_val)
    
    # Gộp 2 phần train và test của Query lại
    query_dataset_full = ConcatDataset([query_dataset_train_split, query_dataset_test_split])
    
    query_dataloader_full = DataLoader(query_dataset_full,
                                       batch_size=config.eval.batch_size_eval,
                                       num_workers=config.training.num_workers,
                                       shuffle=False,
                                       pin_memory=True)
    
    # 2. Khởi tạo Reference Datasets (Satellite)
    # Lưu ý: Cần lấy sample_ids tương ứng từ mỗi split riêng biệt để mapping chính xác
    ref_dataset_train_split = SUES200DatasetEval(data_folder=ref_folder_train,
                                                 mode="satellite",
                                                 transforms=sat_transform_val,
                                                 sample_ids=query_dataset_train_split.get_sample_ids(),
                                                 ref_n=config.eval.eval_reference_n)
    
    ref_dataset_test_split = SUES200DatasetEval(data_folder=ref_folder_test,
                                                mode="satellite",
                                                transforms=sat_transform_val,
                                                sample_ids=query_dataset_test_split.get_sample_ids(),
                                                ref_n=config.eval.eval_reference_n)

    # Gộp 2 phần train và test của Reference lại
    ref_dataset_full = ConcatDataset([ref_dataset_train_split, ref_dataset_test_split])
    
    ref_dataloader_full = DataLoader(ref_dataset_full,
                                     batch_size=config.eval.batch_size_eval,
                                     num_workers=config.training.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    
    print("Query Images Eval (Train + Test):", len(query_dataset_full))
    print("Ref Images Eval (Train + Test):", len(ref_dataset_full))
   
    print("\n{}[{}]{}".format(30*"-", "SUES200 FULL EVAL", 30*"-"))

    # Đưa dataloader đã gộp vào hàm evaluate
    r1_test = evaluate(config=config,
                       model=model,
                       query_loader=query_dataloader_full,
                       ref_loader=ref_dataloader_full, 
                       ranks=[1, 5, 10],
                       cleanup=True)

    # Nếu dùng evaluate_with_rotations, bạn cũng làm tương tự
    # r1_test = evaluate_with_rotations(config=config,
    #                    model=model,
    #                    query_loader=query_dataloader_full,
    #                    ref_loader=ref_dataloader_full, 
    #                    ranks=[1, 5, 10],
    #                    aggregation='max',
    #                    cleanup=True)



# import sys
# import torch
# from torch.utils.data import DataLoader
# from omegaconf import OmegaConf

# sys.path.insert(0, "/home/tts26/sonh/DroneCVGL")

# from data.sues200 import SUES200DatasetEval, get_transforms
# from core.metrics.sues200 import evaluate
# from utils.registry import build_model
# from models.sinkhorn_siamese_network import SinkhornSiameseNetwork, AttentionSinkhornSiameseNetwork 
# from models.siamese_network_max_avg import SiameseNetworkMaxAvg
# from models.siamese_network import SiameseNetwork
# from models.siamese_network_GeM import SiameseNetworkGeM
# from models.dac import DAC

# #-----------------------------------------------------------------------------#
# # Config                                                                      #
# #-----------------------------------------------------------------------------#

# config = OmegaConf.load("./DroneCVGL/config/base.yaml")
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
# query_folder = './data/SUES-200-512x512/drone_view_512' 
# ref_folder = './data/SUES-200-512x512/satellite-view'
 
# if __name__ == '__main__':

#     #-----------------------------------------------------------------------------#
#     # Model                                                                       #
#     #-----------------------------------------------------------------------------#
        
#     print("\nModel: {}".format(config.model.model_name))

#     model = build_model(config)
                          
#     data_config = model.get_config()
#     print(data_config)
#     query_mean, query_std = data_config['query']['mean'], data_config['query']['std']
#     query_img_size = (config.data.drone_img_size, config.data.drone_img_size)
#     ref_mean, ref_std = data_config['reference']['mean'], data_config['reference']['std']
#     ref_img_size = (config.data.sat_img_size, config.data.sat_img_size)
    

#     # load pretrained Checkpoint    
#     if config.training.checkpoint_start is not None:
#         print("Start from:", config.training.checkpoint_start)
#         ckpt = torch.load(config.training.checkpoint_start, map_location="cpu")
#         state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
#         model.load_state_dict(state_dict, strict=False)   

#     # Data parallel
#     print("GPUs available:", torch.cuda.device_count())  
#     if torch.cuda.device_count() > 1 and len(config.training.gpu_ids) > 1:
#         model = torch.nn.DataParallel(model, device_ids=config.training.gpu_ids)
            
#     # Model to device   
#     model = model.to(device)

#     print("\nImage Size Query:", ref_img_size)
#     print("Image Size Ground:", query_img_size)
#     print("Mean: {}".format(query_mean))
#     print("Std:  {}\n".format(query_std)) 


#     #-----------------------------------------------------------------------------#
#     # DataLoader                                                                  #
#     #-----------------------------------------------------------------------------#

#     # Transforms
#     val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(query_img_size[0], mean=query_mean, std=query_std)
                                                                                 
#     query_dataset_test = SUES200DatasetEval(data_folder=query_folder,
#                                                mode="drone",
#                                                transforms=val_transforms,
#                                                )
    
#     query_dataloader_test = DataLoader(query_dataset_test,
#                                        batch_size=config.eval.batch_size_eval,
#                                        num_workers=config.training.num_workers,
#                                        shuffle=False,
#                                        pin_memory=True)
    
#     ref_dataset_test = SUES200DatasetEval(data_folder=ref_folder,
#                                                mode="satellite",
#                                                transforms=val_transforms,
#                                                sample_ids=query_dataset_test.get_sample_ids(),
#                                                ref_n=config.eval.eval_reference_n,
#                                                )
    
#     ref_dataloader_test = DataLoader(ref_dataset_test,
#                                        batch_size=config.eval.batch_size_eval,
#                                        num_workers=config.training.num_workers,
#                                        shuffle=False,
#                                        pin_memory=True)
    
    
#     print("Query Images Test:", len(query_dataset_test))
#     print("Ref Images Test:", len(ref_dataset_test))
   
#     print("\n{}[{}]{}".format(30*"-", "SUES200", 30*"-"))

#     r1_test = evaluate(config=config,
#                        model=model,
#                        query_loader=query_dataloader_test,
#                        ref_loader=ref_dataloader_test, 
#                        ranks=[1, 5, 10],
#                        cleanup=True)

#     # r1_test = evaluate_with_rotations(config=config,
#     #                    model=model,
#     #                    query_loader=query_dataloader_test,
#     #                    ref_loader=ref_dataloader_test, 
#     #                    ranks=[1, 5, 10],
#     #                    aggregation='max',
#     #                    cleanup=True)

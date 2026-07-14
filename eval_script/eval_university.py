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

if __name__ == "__main__":
    config = OmegaConf.load("./DroneCVGL/config/visloc_sinkhorn.yaml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if config.data.dataset == 'U1652-D2S':
        config.data.query_folder_train = './data/University-Release/train/drone'
        config.data.reference_folder_train = './data/University-Release/train/satellite'
        config.data.query_folder_test = './data/University-Release/test/query_drone'
        config.data.reference_folder_test = './data/University-Release/test/gallery_satellite'
    elif config.data.dataset == 'U1652-S2D':
        config.data.query_folder_train = './data/University-Release/train/drone'
        config.data.reference_folder_train = './data/University-Release/train/satellite'
        config.data.query_folder_test = './data/University-Release/test/query_satellite'
        config.data.reference_folder_test = './data/University-Release/test/gallery_drone'
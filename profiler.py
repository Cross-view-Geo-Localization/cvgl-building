import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import torch.utils.benchmark as benchmark
from torch.profiler import profile, record_function, ProfilerActivity
from omegaconf import OmegaConf

from utils.registry import build_model
from utils.logger import Logger
from utils.predict import ForwardMode
# MODELS
from models.sinkhorn_siamese_network import SinkhornSiameseNetwork
from models.siamese_network import SiameseNetwork
from models.aspp import ASPPSinkhornSiameseNetwork
from models.siamese_network_max_avg import SiameseNetworkMaxAvg
from models.siamese_network_GeM import SiameseNetworkGeM
from models.siamese_network_with_atttention import SiameseNetworkWithAttention
from models.mobilegeo import MobileGeo

# Create a fpga wrapper to profile in CPU and CUDA
class FpgaProfilerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        
        self.backbone = base_model.model
        self.dim_reduce = base_model.dim_reduce
        self.sinkhorn = base_model.sinkhorn
        
        self.gpu_device = torch.device("cuda")
        self.cpu_device = torch.device("cpu")
        
        self.backbone.to(self.gpu_device)
        self.dim_reduce.to(self.gpu_device)
        self.sinkhorn.to(self.cpu_device)

    def forward(self, image_gpu):
        # --- STAGE 1: FPGA Programmable Logic (Simulated by GPU) ---
        feat = self.backbone(image_gpu)[-1]
        feat = self.dim_reduce(feat)
        
        # --- STAGE 2: Memory Transfer ---
        feat_cpu = feat.to(self.cpu_device)
        
        # --- STAGE 3: Host CPU Execution ---
        out = self.sinkhorn(feat_cpu)
        
        return out

# Create a dummy wrapper to lock in the QUERY mode
class QueryProfilerWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def forward(self, image1):
        # We handle the 'mode' argument here, keeping it hidden from torchinfo
        return self.base_model(image1, mode=ForwardMode.QUERY)
    

config = OmegaConf.load("./DroneCVGL/config/base.yaml")
script_dir = os.path.dirname(os.path.abspath(__file__))
summary_path = os.path.join(os.path.join(script_dir, "summary_results", config.model.model_name))
os.makedirs(summary_path, exist_ok=True)
sys.stdout = Logger(os.path.join(summary_path, config.model.model_args.model_name + ".txt"))

device = "cpu" if not torch.cuda.is_available() else "cuda"
model = build_model(config)
model = QueryProfilerWrapper(model).to(device)
model.eval()
dummy_input = torch.randn(1, 3, config.data.img_size, config.data.img_size).to(device)


#-----------------------------------------------------------------------------#
# WARM-UP LOOP                                                                     
#-----------------------------------------------------------------------------#
print("Warming up GPU...")
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)
        
# Synchronize to ensure all warm-up tasks are physically finished
torch.cuda.synchronize()


#-----------------------------------------------------------------------------#
# Summarize model's architecture and Params count, MACs                                                                       
#-----------------------------------------------------------------------------#
print(f"\n{30*'-'}\n Profiling Model Complexity \n{30*'-'}")

model_stats = summary(
    model, 
    input_data=dummy_input,
    col_names=("input_size", "output_size", "num_params", "mult_adds"),
    depth=5,
    verbose=1 # Giữ nguyên verbose=1 để in bảng chi tiết
)

# Trích xuất dữ liệu tổng từ đối tượng model_stats
total_macs = model_stats.total_mult_adds
total_params = model_stats.total_params

# Quy đổi: 1 MAC = 2 FLOPs, chuyển sang đơn vị Giga (G)
gflops = (total_macs * 2) / 1e9

print(f"\n[Summary Metrics]")
print(f"Total Parameters:  {total_params / 1e6:.2f} M")
print(f"Total Computation: {gflops:.4f} GFLOPs")

#-----------------------------------------------------------------------------#
# Profile Inference Time                                                                   
#-----------------------------------------------------------------------------#
print(f"\n{30*'-'}\n Profiling Inference Time \n{30*'-'}")

timer = benchmark.Timer(
    stmt='model(x)',
    setup='import torch',
    globals={'model': model, 'x': dummy_input},
    num_threads=1,
    label='Latency Measurement',
    sub_label=config.model.model_name
)

# Đo lường
profile_result = timer.timeit(200)
print(profile_result) # Vẫn in ra format chuẩn của PyTorch benchmark

# Bóc tách ra số ms cụ thể
mean_latency_ms = profile_result.mean * 1000
median_latency_ms = profile_result.median * 1000

print(f"\n[Summary Metrics]")
print(f"Mean Inference Time:   {mean_latency_ms:.2f} ms")
print(f"Median Inference Time: {median_latency_ms:.2f} ms")

#-----------------------------------------------------------------------------#
# Profile VRAM usage                                                                
#-----------------------------------------------------------------------------#

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    _ = model(dummy_input)

peak_memory_bytes = torch.cuda.max_memory_allocated()
peak_memory_mb = peak_memory_bytes / (1024 ** 2)

print(f"Peak VRAM Usage: {peak_memory_mb:.2f} MB")

#-----------------------------------------------------------------------------#
# Ultimate Profile                                                                
#-----------------------------------------------------------------------------#

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True
) as prof:
    with record_function("model_inference"):
        model(dummy_input)
# Print a table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export to view in chrome://tracing
prof.export_chrome_trace("trace.json")

#-----------------------------------------------------------------------------#
# DATABASE SEARCH PROFILING                                            
#-----------------------------------------------------------------------------#
print(f"\n{30*'-'}\n Profiling Database Search \n{30*'-'}")

# 1. Dynamically get the descriptor dimension from your model
with torch.no_grad():
    dummy_output = model(dummy_input)
    descriptor_dim = dummy_output.shape[1]

# 2. Setup the dummy database and query
num_database_images = 10000
print(f"Simulating search across {num_database_images} satellite images (Dim: {descriptor_dim})")

# L2 Normalize them to simulate real-world cosine similarity
dummy_database = F.normalize(torch.randn(num_database_images, descriptor_dim, device=device), p=2, dim=-1)
# Batch size of 1 for the drone's active query
dummy_query = F.normalize(torch.randn(1, descriptor_dim, device=device), p=2, dim=-1) 

def database_search(query, database, top_k=5):
    # Cosine similarity via Matrix Multiplication
    similarities = query @ database.T
    # Find the top K matches
    scores, indices = torch.topk(similarities, k=top_k, dim=-1)
    return scores, indices

# --- Profile Search Latency ---
search_timer = benchmark.Timer(
    stmt='search(q, db)',
    setup='import torch',
    globals={'search': database_search, 'q': dummy_query, 'db': dummy_database},
    num_threads=1,
    label='Search Latency',
    sub_label=f'1 vs {num_database_images} (Dim: {descriptor_dim})'
)
print(search_timer.timeit(100)) # Running 1000 times because search is very fast!

# --- Profile Search Memory ---
if device == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = database_search(dummy_query, dummy_database)

    search_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # Total memory = the size of the database tensor itself + operation overhead
    db_size_mb = (dummy_database.element_size() * dummy_database.nelement()) / (1024 ** 2)
    
    print(f"Database Tensor Size: {db_size_mb:.2f} MB")
    print(f"Peak VRAM Usage during Search: {search_vram:.2f} MB")
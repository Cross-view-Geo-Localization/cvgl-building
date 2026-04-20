# CUDA_VISIBLE_DEVICES=2 python3 SparK/pretrain/main.py \
#   --data_path=./data/University-Release \
#   --exp_name=hgnetv2_university_pretrain \
#   --model=hgnetv2_b1.ssld_stage1_in22k_in1k \
#   --bs=128

# CUDA_VISIBLE_DEVICES=2 python3 SparK/pretrain/main.py --exp_name=debug --data_path=../../data/University-Release --model=hgnetv2_b1.ssld_stage1_in22k_in1k --bs=32
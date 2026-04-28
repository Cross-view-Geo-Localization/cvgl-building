import torch
import numpy as np
from tqdm import tqdm
import gc
from DroneCVGL.utils.predict import predict, ForwardMode, predict_query_rotations
import torch.nn.functional as F


def apply_superglobal(qf, gf, initial_scores, M=100, K=5):
    """
    SuperGlobal reranking.
    """
    # 1. Get Top-M candidate index. 
    # FIX: Add .copy() to remove the negative stride caused by [::-1]
    initial_index = np.argsort(initial_scores)[::-1].copy()
    top_m_index = initial_index[:M]
    
    # 2. Construct mini-database
    top_m_gf = gf[top_m_index] # [M, D] - This will now work perfectly
    mini_db = torch.cat([qf.unsqueeze(0), top_m_gf], dim=0) # [M+1, D]
    
    # L2 Normalization
    mini_db = F.normalize(mini_db, p=2, dim=1)
    
    # 3. Internal Similarity Matrix
    sim_mini = mini_db @ mini_db.T # [M+1, M+1]
    
    # 4. Find K-Nearest Neighbors
    _, topk_idx = torch.topk(sim_mini, k=K+1, dim=1)
    
    # 5. Aggregation
    neighbors = mini_db[topk_idx] # Shape: [M+1, K+1, D]
    new_mini_db = neighbors.mean(dim=1) # Shape: [M+1, D]
    
    # L2 Normalization again after aggregation
    new_mini_db = F.normalize(new_mini_db, p=2, dim=1)
    
    # 6. Re-calculate scores between NEW query and NEW candidates
    new_qf = new_mini_db[0]
    new_gf = new_mini_db[1:]
    new_scores = (new_gf @ new_qf).cpu().numpy()
    
    # 7. Re-order the Top-M list based on new scores
    rerank_order = np.argsort(new_scores)[::-1].copy() # Added .copy() here too just in case
    reranked_top_m_index = top_m_index[rerank_order]
    
    # 8. Concatenate reranked Top-M with the rest of the list
    final_index = np.concatenate([reranked_top_m_index, initial_index[M:]])
    
    return final_index

def evaluate(config,
            model,
            query_loader,
            ref_loader,
            ranks=[1, 5, 10],
            cleanup=True,
            use_superglobal=True, # THÊM tham số kích hoạt SuperGlobal
            sg_M=100,             # THÊM tham số M
            sg_K=5):              # THÊM tham số K
    
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader, mode=ForwardMode.QUERY)
    img_features_ref, ids_ref = predict(config, model, ref_loader, mode=ForwardMode.REFERENCE)
    
    gl = ids_ref.cpu().numpy()
    ql = ids_query.cpu().numpy()
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_ref)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        # CẬP NHẬT: Truyền thêm các tham số SuperGlobal vào eval_query
        ap_tmp, CMC_tmp = eval_query(
            img_features_query[i], ql[i], img_features_ref, gl, 
            use_superglobal=use_superglobal, M=sg_M, K=sg_K
        )
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_ref)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    if cleanup:
        del img_features_query, ids_query, img_features_ref, ids_ref
        gc.collect()
    
    return CMC[0]


# CẬP NHẬT: Nhận thêm tham số SuperGlobal
def eval_query(qf, ql, gf, gl, use_superglobal=False, M=100, K=5):
    
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
 
    if use_superglobal:
        # Nếu bật SuperGlobal, bỏ qua sắp xếp thô và dùng hàm rerank
        index = apply_superglobal(qf, gf, score, M=M, K=K)
    else:
        # Logic dự phòng (mặc định cũ)
        index = np.argsort(score)
        index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    # ... (Giữ nguyên code cũ) ...
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.isin(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.isin(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


def calc_sim(
    config,
    model,
    reference_dataloader,
    step_size=1000,
    cleanup=True
):
    print("Extract Reference Images Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, mode=ForwardMode.REFERENCE)
    near_dict = calculate_nearest(
        reference_features,
        reference_labels,
        neighbour_range=config.neighbour_range,
        step_size=step_size
    )

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels
        gc.collect()
        
    return near_dict

def calculate_nearest(reference_features, reference_labels, neighbour_range=64, step_size=1000):
    R = len(reference_features)
    steps = R // step_size + 1
    similarity = []

    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = reference_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    
    # Matrix R x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range + 1, dim=1)
    topk_references = []

    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i, :]])

    topk_references = torch.stack(topk_references, dim=0)

    # mask for ids without gt hits
    mask = topk_references != reference_labels.unsqueeze(1)
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()

    nearest_dict = dict()

    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]

        # Convert the integer key back to a 4-digit zero-padded string for University-1652 string index label
        key = str(reference_labels[i].item()).zfill(4)

        # convert the nearest neighbor IDs to strings
        nearest_dict[key] = [str(n).zfill(4) for n in nearest]
    
    return nearest_dict
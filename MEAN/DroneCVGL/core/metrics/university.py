import torch
import numpy as np
from tqdm import tqdm
import gc
from utils.predict import predict, ForwardMode

def evaluate(config,
            model,
            query_loader,
            reference_loader,
            ranks=[1, 5, 10],
            cleanup=True):
    
    
    print("Extract Features:")
    img_features_query, ids_query = predict(config, model, query_loader, mode=ForwardMode.QUERY)
    img_features_reference, ids_reference = predict(config, model, reference_loader, mode=ForwardMode.REFERENCE)
    
    gl = ids_reference.cpu().numpy()
    ql = ids_query.cpu().numpy()
    
    print("Compute Scores:")
    CMC = torch.IntTensor(len(ids_reference)).zero_()
    ap = 0.0
    for i in tqdm(range(len(ids_query))):
        ap_tmp, CMC_tmp = eval_query(img_features_query[i], ql[i], img_features_reference, gl)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    
    AP = ap/len(ids_query)*100
    
    CMC = CMC.float()
    CMC = CMC/len(ids_query) #average CMC
    
    # top 1%
    top1 = round(len(ids_reference)*0.01)
    
    string = []
             
    for i in ranks:
        string.append('Recall@{}: {:.4f}'.format(i, CMC[i-1]*100))
        
    string.append('Recall@top1: {:.4f}'.format(CMC[top1]*100))
    string.append('AP: {:.4f}'.format(AP))             
        
    print(' - '.join(string)) 
    
    results = {}
    for r in ranks:
        results[f"r{r}"] = float(CMC[r-1] * 100)
    results["r_top1"] = float(CMC[top1] * 100)
    results["AP"] = float(AP)

    # cleanup and free memory on GPU
    if cleanup:
        del img_features_query, ids_query, img_features_reference, ids_reference
        gc.collect()
        #torch.cuda.empty_cache()
    
    return results


def eval_query(qf,ql,gf,gl):

    score = gf @ qf.unsqueeze(-1)
    
    score = score.squeeze().cpu().numpy()
 
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]    

    # good index
    query_index = np.argwhere(gl==ql)
    good_index = query_index

    # junk index
    junk_index = np.argwhere(gl==-1)
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


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
        neighbour_range=config.training.neighbour_range,
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
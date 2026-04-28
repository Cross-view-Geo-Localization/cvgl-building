import os
import cv2
import numpy as n
from torch.utils.data import Dataset
import copy
from tqdm import tqdm
import numpy as np
import time
import random

def get_data(path):
    data = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            data[name] = {"path": os.path.join(root, name)}
            for _, _, files in os.walk(data[name]["path"], topdown=False):
                data[name]["files"] = files
                
    return data


class U1652DatasetTrain(Dataset):
    
    def __init__(self,
                 query_folder,
                 reference_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 prob_flip=0.5,
                 shuffle_batch_size=128):
        super().__init__()

        self.query_dict = get_data(query_folder)
        self.reference_dict = get_data(reference_folder)
        
        # use only folders that exists for both reference and query
        self.ids = list(set(self.query_dict.keys()).intersection(self.reference_dict.keys()))
        self.ids.sort()
        
        self.pairs = []
        
        for idx in self.ids:
            
            reference_img = "{}/{}".format(self.reference_dict[idx]["path"],
                                       self.reference_dict[idx]["files"][0])
            
            query_path = self.query_dict[idx]["path"]
            query_imgs = self.query_dict[idx]["files"]
            
            for q in query_imgs:
                self.pairs.append((idx, "{}/{}".format(query_path, q), reference_img))
        
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.prob_flip = prob_flip
        self.shuffle_batch_size = shuffle_batch_size
        
        self.samples = copy.deepcopy(self.pairs)

        # pairs_by_idx for the hard negative sampling
        self.pairs_by_idx = {}
        for pair in self.pairs:
            idx = pair[0]
            if idx not in self.pairs_by_idx:
                self.pairs_by_idx[idx] = []
            self.pairs_by_idx[idx].append(pair)
            
        self.unseen_pools = copy.deepcopy(self.pairs_by_idx)
        for idx in self.unseen_pools:
            random.shuffle(self.unseen_pools[idx])

    def __getitem__(self, index):
        
        idx, query_img_path, reference_img_path = self.samples[index]
        
        # for query there is only one file in folder
        query_img = cv2.imread(query_img_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        
        reference_img = cv2.imread(reference_img_path)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
        
        return query_img, reference_img, idx
    
    def __len__(self):
        return len(self.samples)
    
    
    def shuffle(self):

            '''
            custom shuffle function for unique class_id sampling in batch
            '''
            
            print("\nShuffle Dataset:")
            
            pair_pool = copy.deepcopy(self.pairs)
              
            # Shuffle pairs order
            random.shuffle(pair_pool)
            
            # Lookup if already used in epoch
            pairs_epoch = set()   
            idx_batch = set()
     
            # buckets
            batches = []
            current_batch = []
             
            # counter
            break_counter = 0
            
            # progressbar
            pbar = tqdm()
    
            while True:
                
                pbar.update()
                
                if len(pair_pool) > 0:
                    pair = pair_pool.pop(0)
                    
                    idx, _, _ = pair
                    
                    if idx not in idx_batch and pair not in pairs_epoch:
                        
                        idx_batch.add(idx)
                        current_batch.append(pair)
                        pairs_epoch.add(pair)
            
                        break_counter = 0
                        
                    else:
                        # if pair fits not in batch and is not already used in epoch -> back to pool
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                            
                        break_counter += 1
                        
                    if break_counter >= 512:
                        break
                   
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
       
            pbar.close()
            
            # wait before closing progress bar
            time.sleep(0.3)
            
            self.samples = batches
            
            print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples))) 
            print("Break Counter:", break_counter)
            print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
            print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][0], self.samples[-1][0]))  
    
    def hard_negative_sampling_shuffle(self, sim_dict=None, neighbour_select=64, neighbour_range=128):
        """
        Custom shuffle function for University-1652 (1-to-N sampling).
        Groups similar classes together, but samples 1 random drone per class per epoch.
        """
        print("\nShuffle Dataset:")
        
        idx_pool = list(self.pairs_by_idx.keys())
        random.shuffle(idx_pool)
        
        neighbour_split = neighbour_select // 2
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)
           
        # Lookup if already used in epoch
        pairs_epoch = set()   
        idx_batch = set()
 
        # buckets
        batches = []
        current_batch_pairs = [] # store the actual (idx, query, ref) tuples here
        
        break_counter = 0
        
        pbar = tqdm(total=len(idx_pool))

        while True:
            
            if len(idx_pool) > 0:
                idx = idx_pool.pop(0)
                pair = self._get_pair_without_replacement(idx)

                # Check if class is valid and batch has space
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch_pairs) < self.shuffle_batch_size:
                
                    idx_batch.add(idx)
                    # RANDOM SAMPLING: Pick exactly ONE drone-satellite pair for this class
                    current_batch_pairs.append(pair)
                    pairs_epoch.add(pair)
                    break_counter = 0
                  
                    # HARD NEGATIVE MINING: Find similar classes
                    if sim_dict is not None and len(current_batch_pairs) < self.shuffle_batch_size:
                        
                        near_similarity = similarity_pool[idx][:neighbour_range]
                        near_neighbours = copy.deepcopy(near_similarity[:neighbour_split])
                        
                        far_neighbours = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(far_neighbours)
                        far_neighbours = far_neighbours[:neighbour_split]
                        
                        near_similarity_select = near_neighbours + far_neighbours
                        
                        for idx_near in near_similarity_select:

                            near_pair = self._get_pair_without_replacement(idx_near)
                       
                            if len(current_batch_pairs) >= self.shuffle_batch_size:
                                break
                            
                            # check if similar class is not already in batch or epoch
                            if idx_near not in idx_batch and near_pair not in pairs_epoch:
                        
                                idx_batch.add(idx_near)
                                # RANDOM SAMPLING: Pick ONE drone pair for the hard negative class
                                current_batch_pairs.append(near_pair)
                                pairs_epoch.add(near_pair)
                                
                                if idx_near in similarity_pool[idx]:
                                    similarity_pool[idx].remove(idx_near)
                                break_counter = 0
                                
                else:
                    # if idx fits not in batch and is not already used in epoch -> back to pool
                    if idx not in idx_batch and pair not in pairs_epoch:
                        idx_pool.append(idx)
                        
                    break_counter += 1
                    
                if break_counter >= 1024:
                    break
               
            else:
                break

            # When batch is full, dump into main batch list
            if len(current_batch_pairs) >= self.shuffle_batch_size:
                batches.extend(current_batch_pairs)
                pbar.update(len(idx_batch)) # update progress bar by classes processed
                idx_batch = set()
                current_batch_pairs = []

        pbar.close()
        time.sleep(0.3)
        
        # self.samples now contains a perfectly flat list of 1-to-1 pairs, 
        # but intelligently grouped into batches of 128 similar classes!
        self.samples = batches
        
        print("Classes left in pool (should be 0):", len(idx_pool))
        print("Total pairs scheduled for this epoch:", len(self.samples))
        print("Break Counter:", break_counter)

    def _get_pair_without_replacement(self, idx):
        '''
        Pops one drone image from the unseen pool. 
        If the pool is empty, it refills and reshuffles.
        '''
        if len(self.unseen_pools[idx]) == 0:
            # Refill from master and reshuffle
            self.unseen_pools[idx] = copy.deepcopy(self.pairs_by_idx[idx])
            random.shuffle(self.unseen_pools[idx])
            
        return self.unseen_pools[idx].pop()
    

class U1652DatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 mode,
                 transforms=None,
                 sample_ids=None,
                 reference_n=-1):
        super().__init__()
 

        self.data_dict = get_data(data_folder)

        # use only folders that exists for both reference and query
        self.ids = list(self.data_dict.keys())
                
        self.transforms = transforms
        
        self.given_sample_ids = sample_ids
        
        self.images = []
        self.sample_ids = []
        
        self.mode = mode
        
        
        self.reference_n = reference_n
        

        for i, sample_id in enumerate(self.ids):
                
            for j, file in enumerate(self.data_dict[sample_id]["files"]):
                    
                self.images.append("{}/{}".format(self.data_dict[sample_id]["path"],
                                                      file))
                
                self.sample_ids.append(sample_id) 
        
        
    def __getitem__(self, index):
        
        img_path = self.images[index]
        sample_id = self.sample_ids[index]
        
        img = cv2.imread(img_path)
        if img is None:
            import os
            file_exists = os.path.exists(img_path)
            raise FileNotFoundError(f"\nCRASH ALERT! OpenCV returned None.\n"
                                    f"Attempted Path: {img_path}\n"
                                    f"Does this file actually exist on disk? {file_exists}\n")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        #if self.mode == "sat":
        
        #    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        #    img180 = cv2.rotate(img90, cv2.ROTATE_90_CLOCKWISE)
        #    img270 = cv2.rotate(img180, cv2.ROTATE_90_CLOCKWISE)
            
        #    img_0_90 = np.concatenate([img, img90], axis=1)
        #    img_180_270 = np.concatenate([img180, img270], axis=1)
            
        #    img = np.concatenate([img_0_90, img_180_270], axis=0)
            
        
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = int(sample_id)
        if self.given_sample_ids is not None:
            if sample_id not in self.given_sample_ids:
                label = -1
        
        return img, label

    def __len__(self):
        return len(self.images)
    
    def get_sample_ids(self):
        return set(self.sample_ids)
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

class Cut(ImageOnlyTransform):
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    
    def apply(self, image, **params):
        
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        return ("size", "cutting")     
    

def get_transforms_train(image_size_sat,
                         image_size_drone,
                         query_mean=[0.485, 0.456, 0.406],
                         query_std=[0.229, 0.224, 0.225],
                         ref_mean=[0.485, 0.456, 0.406],
                         ref_std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    
    
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_range=(90, 100), p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.HorizontalFlip(p=0.5),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.25, p=1.0),
                                               A.CoarseDropout(
                                                    num_holes_range=(6, 15),
                                                    hole_height_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                                                    hole_width_range=(int(0.1 * image_size_sat[0]), int(0.2 * image_size_sat[0])),
                                                    p=1.0),
                                              ], p=0.3),
                                      A.Normalize(ref_mean, ref_std),
                                      A.ShiftScaleRotate(
                                            shift_limit=0.05,  # Shift by max 5%. Forces model to handle off-center targets.
                                            scale_limit=0.10,  # Zoom in/out by 10% to simulate different satellite altitudes.
                                            rotate_limit=180,  # Full 360-degree rotation. Crucial for top-down invariance.
                                            interpolation=cv2.INTER_LINEAR,
                                            border_mode=cv2.BORDER_CONSTANT, # Use black borders. 
                                            value=0, # Black pixel value
                                            p=0.5
                                        ),
                                      ToTensorV2(),
                                     ])
            
    

    drone_transforms = A.Compose([# Cut(cutting=ground_cutting, p=1.0),
                                    A.ImageCompression(quality_range=(90, 100), p=0.5),
                                    A.Resize(image_size_drone[0], image_size_drone[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                    A.HorizontalFlip(p=0.5),
                                    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1, p=0.5),
                                    A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                            ], p=0.3),
                                    A.OneOf([
                                            A.GridDropout(ratio=0.25, p=1.0),
                                            A.CoarseDropout(
                                                num_holes_range=(6, 15),
                                                hole_height_range=(int(0.1 * image_size_drone[0]), int(0.2 * image_size_drone[0])),
                                                hole_width_range=(int(0.1 * image_size_drone[0]), int(0.2 * image_size_drone[0])),
                                                p=1.0),
                                            ], p=0.3),
                                    A.Normalize(query_mean, query_std),
                                    A.ShiftScaleRotate(
                                            shift_limit=0.05,  # Shift by max 5%. Forces model to handle off-center targets.
                                            scale_limit=0.10,  # Zoom in/out by 10% to simulate different satellite altitudes.
                                            rotate_limit=15,  # Full 360-degree rotation. Crucial for top-down invariance.
                                            interpolation=cv2.INTER_LINEAR,
                                            border_mode=cv2.BORDER_CONSTANT, # Use black borders. 
                                            value=0, # Black pixel value
                                            p=0.5
                                        ),
                                    ToTensorV2(),
                                   ])
                
            
               
    return drone_transforms, satellite_transforms


def get_transforms_val(image_size_sat,
                       image_size_drone,
                       query_mean=[0.485, 0.456, 0.406],
                       query_std=[0.229, 0.224, 0.225],
                       ref_mean=[0.485, 0.456, 0.406],
                       ref_std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    
    
    
    satellite_transforms = A.Compose([A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(ref_mean, ref_std),
                                      ToTensorV2(),
                                     ])
            
    
 

    drone_transforms = A.Compose([#Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(image_size_drone[0], image_size_drone[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(query_mean, query_std),
                                   ToTensorV2(),
                                  ])
            
    return drone_transforms, satellite_transforms
    

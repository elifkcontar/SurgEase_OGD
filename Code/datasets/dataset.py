import torch
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2

#from utils.utils_bb import get_train_transform, collate_fn
# the dataset class
class BB_Dataset(Dataset):
    def __init__(self, csv_filename, width, height, classes, transforms=None):
        self.transforms = transforms
        self.height = height
        self.width = width
        self.classes = classes
        
        # get all the image paths in sorted order
        self.all_images = []
        self.df = pd.read_csv(csv_filename)
        for item in range(len(self.df['frame_path'])):
            #Get unique frame names, for frame with multiple objects
            if self.df.loc[item, 'frame_path'] not in self.all_images: 
                self.all_images.append(self.df.loc[item, 'frame_path'])
    
    def __getitem__(self, idx):
        # capture the image name and the full image path
        image = Image.open(self.all_images[idx])
        image.load()
        
        # capture the corresponding XML file for getting the annotations
        boxes = []
        labels = []
        
        # get the height and width of the image
        image_width = image.size[0]
        image_height = image.size[1]
        
        df = self.df.loc[self.df['frame_path'] == self.all_images[idx]]
        df = df.reset_index()
        # box coordinates for xml files are extracted and corrected for image size given
        for item in range(len(df['frame_path'])):
            # map the current object name to `classes` list to get...
            # ... the label index and append to `labels` list
            label_value = df.loc[item, 'label_value']
            h = df.loc[item, 'h']
            w = df.loc[item, 'w']
            x = df.loc[item, 'x']
            y = df.loc[item, 'y']

            labels.append(self.classes.index(label_value))
            
            # xmin = left corner x-coordinates
            xmin = int(x*image_width)
            # xmax = right corner x-coordinates
            xmax = int((x+w)*image_width)
            # ymin = left corner y-coordinates
            ymin = int(y*image_height)
            # ymax = right corner y-coordinates
            ymax = int((y+h)*image_height)
            
            # resize the bounding boxes according to the...
            # ... desired `width`, `height`
            #xmin_final = (xmin/image_width)*self.width
            #xmax_final = (xmax/image_width)*self.width
            #ymin_final = (ymin/image_height)*self.height
            #yamx_final = (ymax/image_height)*self.height
            
            boxes.append([xmin, ymin, xmax, ymax])
        
        # bounding box to tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id
        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = image,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image, target
    
    def __len__(self):
        return len(self.all_images)


import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

#define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# prepare the final datasets and data loaders
#csv_filename = r'C:\Users\Elif\OneDrive - Surgease Innovations Ltd\Desktop\OGD\label_all.csv'
#train_dataset = BB_Dataset(csv_filename, 640,480, ['osephagus_', 'laryx_', 'stomach_'], get_train_transform())
#valid_dataset = BB_Dataset(csv_filename, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
#train_loader = DataLoader(
#    train_dataset,
#    batch_size=4,
#    shuffle=True,
#    num_workers=0,
#    collate_fn=collate_fn
#)
#valid_loader = DataLoader(
#    valid_dataset,
#    batch_size=BATCH_SIZE,
#    shuffle=False,
#    num_workers=0,
#    collate_fn=collate_fn
#)
#print(f"Number of training samples: {len(train_dataset)}")
#print(f"Number of validation samples: {len(valid_dataset)}\n")


# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    csv_filename = r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\OGD\label_all.csv'
    dataset = BB_Dataset(csv_filename, 640,480, ['oesophagus_', 'laryx_', 'stomach_'])
    print(f"Number of training images: {len(dataset)}")
    CLASSES=['oesophagus_', 'laryx_', 'stomach_']
    # function to visualize a single sample
    def visualize_sample(image, target):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        box = target['boxes'][0]
        label = CLASSES[target['labels'][0]]
        cv2.rectangle(
            image, 
            (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
            (0, 255, 0), 1
        )
        cv2.putText(
            image, label, (int(box[0]), int(box[1]-5)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
    
    import time
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        start=time.time()
        image, target = dataset[i]
        visualize_sample(np.asarray(image), target)
        print(time.time()-start)
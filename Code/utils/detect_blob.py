from functools import partial
from charset_normalizer import detect
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps

import pandas as pd
import os

class DetectBlob(object):
    """
    returned image size is (320,352) by default
    """
    def __init__(self, min_threshold=10, min_area=60000, radius_scale=1.00, final_size=(540, 960)):
        params = cv2.SimpleBlobDetector_Params()
        params.minArea = 246000#pi*280*280
        params.maxArea = 635000#pi*450*450
        params.minThreshold = min_threshold
        params.maxThreshold = 170#60
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        params.filterByColor = False
        params.filterByArea = True

        self.detector = cv2.SimpleBlobDetector_create(params)
        self.radius_scale = radius_scale
        self.final_size = final_size

    def __call__(self, sample):
        image_pil = sample
        im_gray = np.array(ImageOps.grayscale(image_pil))
        im_original = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        keypoints = self.detector.detect(im_gray)
        if len(keypoints) > 0:
            keypoint_top = [keypoints[0]]
        else:
            print("Cannot detect blob, Resizing to final dimension ")
            image_final = transforms.Resize(self.final_size)(image_pil)
            image_final = image_pil
            return image_final

        keypoint_center_x = int(keypoint_top[0].pt[0])
        keypoint_radius = int((keypoint_top[0].size / 2) * self.radius_scale)
        keypoint_leftmost = (keypoint_center_x - keypoint_radius) if (keypoint_center_x - keypoint_radius) > 0 else 0
        keypoint_rightmost = (keypoint_center_x + keypoint_radius) if (keypoint_center_x + keypoint_radius) < \
                                                                      im_original.shape[
                                                                          1] else im_original.shape[1]

        image_cropped = im_original[:, keypoint_leftmost:keypoint_rightmost, :]

        image_cropped_pil = Image.fromarray(np.uint8(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)))
        image_cropped_pil = transforms.Resize(self.final_size)(image_cropped_pil)
        return image_cropped_pil


def crop_from_blob_autamatic():
#Crop frames with labelled
#    import pandas as pd
#    import os

    #path = "C:/Users/Elif/Desktop/Datasets/IBD_cropped/"
    #df = pd.read_csv(r'C:\Users\Elif\Desktop\gi\label_all.csv')
    #Create folder structure
#    videos = df['video_name'].unique()
#    for video in videos:
#        os.mkdir(path+video)
#        os.mkdir(path+video+"/frames/")
#        a=0
#    detect = DetectBlob()
    #Save images
#    for i in range(len(df.index)):
#        img = Image.open(df.loc[i,'frame_path'])
#        ret = detect.__call__(img)
#        part_path=df.loc[i,'frame_path'].split("\\")[-3:]
#        ret.save(path+part_path[0]+'/'+part_path[1]+'/'+part_path[2])
    return

def select_crop_region_from_videos():
    import pandas as pd
    import os
    import matplotlib.pyplot as plt

    df = pd.read_csv(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\SurgEase_IBD\label_all.csv')
    videos = df['video_name'].unique()    #Save images
    current_video_name = ''
    for i in range(len(df.index)):
        img = Image.open(df.loc[i,'frame_path'])
        if(current_video_name!=df.loc[i,'video_name']):
            plt.imshow(img)
            plt.show()
            print(df.loc[i,'video_name'])
            a=0
        current_video_name = df.loc[i,'video_name']

def crop_video(video, x=[0,0], y=[0,0], resize_dim=(0,0)):
    ret = video.crop((x[0], y[0], x[1], y[1]))
    if (resize_dim!=(0,0)):
        ret = ret.resize(resize_dim)
    return ret

def add_circular_blob(video, center=[0,0], radius=0):
    mask = create_circular_mask(video.height, video.width, center=center, radius=radius)
    im = np.asarray(video)
    masked_img = im.copy()
    masked_img[~mask] = 0
    ret = Image.fromarray(np.uint8(masked_img))
    return ret 

def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def crop_cloudfactory_dataset():
    type_0=[98,68,158,159,141,91,85,64,69,139,86,140,66]
    type_1=[39, 135,12, 95,26,57,10,9,13,88,132,67,18,17,62,33,47,19,60,155,92,21,138,58,38,137,35,73,90,59,75,79,120,84,23,78,89,82,30,93,133,37,134,81,24,77,96,51,31,83,80,74,76,136,63,46,34,163,22,129,65,72,11,27,156,128,32,56,61]
    type_2=[126, 154,124,122,121,180,125,127,171,172]
    type_3=[146,170,144,41,143,169,142,145,178,44]
    type_4=[48,49,52,45,40,43]
    type_5=[7,53,54,119,117,149,1,14,123,2,162,15,152,8,176,150,6,148,118,4,55,174,153,161,151,147]
    type_6=[101,108,113,102,112,106,100,103,107,111,110,115,104,109,105,114]
    
    path = "C:/Users/ElifKübraÇontar/OneDrive - Surgease Innovations Ltd/Desktop/Datasets/IBD_cloudfactory_cropped/"
    df = pd.read_csv(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\gi\label_all.csv')
    #Create folder structure
    csv_videos = df['video_name'].unique()
    for video in csv_videos:
        os.mkdir(path+video)
        os.mkdir(path+video+"/frames/")    
    #Save images
    for i in range(len(df.index)):
        img = Image.open(df.loc[i,'frame_path'])
        if('.mp4' in df.loc[i,'video_name']):
            continue
        video_number = int((df.loc[i,'video_name']).replace("video", "")) 

        if video_number  in type_0:
            ret = crop_video(img, x=[176,783], y=[0,539], resize_dim=(540,480))
        elif video_number  in type_1:
            #TODO:calculate numbers in circualr blob functions
            single_video = add_circular_blob(img, x=171, radius=629)
            ret = crop_video(single_video, x=[171,800], y=[9,489])
        elif video_number  in type_2:
            ret = crop_video(img, x=[190,735], y=[0,483])
        elif video_number  in type_3:
            ret = crop_video(img, x=[287,834], y=[0,483])
        elif video_number  in type_4:
            ret = crop_video(img, x=[253,798], y=[0,480])
        elif video_number  in type_5:
            ret = crop_video(img, x=[348,929], y=[13,522])
        elif video_number  in type_6:
            #TODO:calculate numbers in circualr blob functions
            single_video = add_circular_blob(img, x=30, radius=626)
            ret = crop_video(single_video, x=[30,656], y=[26,509])
        part_path=df.loc[i,'frame_path'].split("\\")[-3:]
        ret.save(path+part_path[0]+'/'+part_path[1]+'/'+part_path[2])

    return

def crop_reference_dataset():
    type_1=[
        '1a2a0e9a-db19-46c6-a6e6-47aa0635af0f',
        '1eaf7f25-5cf0-4ceb-aed2-413138a2a0a1',
        '2af49402-2e1a-4ec7-b61a-354928ec8c1c',
        '3b603af5-1bfe-4370-a912-ac66676ed283',
        '5d2efd1e-f258-4467-bc5b-9619646dd3c6',
        '7a4b28fd-057a-497b-9ef8-9323342e46f9',
        '38b0291d-f958-4ada-bae1-9900cc7e3a61',
        '75d7a2d5-0cad-46f4-9f4c-cfa0102a7a49',	
        '116efd4f-1d8c-4a7f-88b2-4e182d4ea349',
        '4395d931-9fe3-44dc-9cff-4909c708ee9c',
        '812214e4-9363-415f-bf93-60552a7c4514',
        '3820833a-f144-4174-ac48-bbfee7bb729d',
        '97791175-18f6-4cf6-a2fa-ab4677ca6bf3',
        'afd5d475-f97c-4004-b84d-f353bea55c32',
        'c1af2385-e998-4d92-a4d6-198dc732286b',
        'ca8c4cae-7911-46cc-ab2b-356fe7a36585',
        'dea3d095-8ee4-4451-8417-9749c3c43d71',
        'dfef2edd-4467-4705-a38b-7c2e977e7ba8',
        'f8df8cf1-65eb-4c3e-bb52-e449dfc8e2bb'
]
    type_6=[
        'faa6b9a0-c4e5-4dc1-8621-7f18c6d21cdd',	
        'f5064d1d-6b7d-4cb2-89d6-4dce1d04b41a',
        'f88cb164-9701-4345-8e6d-20656a1e0bc1',	
        'f29acc69-1aea-4223-8f09-c10cecf4eef7',
        'e8d1d3cc-12e0-48a1-8b11-67da60972ce4',
        'e21d53b3-e89a-4ff7-a398-74b19a1d941d',			
        'e77cf29b-5543-44e0-83b1-afe59f994eb8',
        'e956a7b3-79b8-4419-8804-43b5d93d502f',		
        'ea4bec48-7bc0-4b13-ad53-9dd26882aeaa',
        'eba11b0e-6057-4645-be92-0d358c37f6ff',
        'efe6ab38-76da-4e99-a114-15f097ba61ad',	
        'f5d7304a-8fa2-4fbd-9216-b3d48dd97d29',
        'dfe832e8-6af9-4f7c-a75a-9ee1e0c20cd6',
        'd01ba63f-edd4-4cd0-8ee2-301d65cfcd0f',
        'd89efea1-2984-4c31-ba61-9f2f9a72e2ba',
        'dcfddc53-0adb-4eb1-9692-f5c8b90104f8',
        'dd6b649f-213a-4e2a-b74a-3a5963191539',
        'dd8a6b08-e015-4445-b7a5-162cc580c7a0',
        'dda43f0f-a858-448f-bba3-676a9a6885f3',
        'c3aca1c5-c2fc-4722-bd57-1ccde7925769',
        'c3bafc75-7218-41e7-a6a4-2a4f3d239860',
        'b0d67547-d84b-4f4a-b1b1-618efe98e84e',
        'b667a52f-a362-4f73-83cd-09dbdfdc6439',	
        '22613826-90ba-4afb-98d0-ac260e76570d',	
        'abfd6f75-f1ce-4fe5-a6ab-b83aaea9b6f6',
        'a7d650e1-c61c-4fb4-926e-70ed34ccf461',
        'a2ffe937-26fe-4ff0-9d88-8d366a15fa5d',
        '75958069-887a-4ca3-b763-78e4d9b60e64',
        '262e2896-759e-4702-bc18-559ee21c6c71',
        '4811e5c1-e81b-41c7-b3a4-944b4d130fa0',
        '07018b5b-105d-4fe0-a921-134200ca5e04',
        '60652e8c-58ce-44e5-953d-ba2cab94a47b',
        '442297c7-338e-49be-9e23-67dc8fe12d27',
        '493714e2-8aee-4857-af69-8ca4bbdc935c',
        '695ab999-01af-4f8a-8bc4-c1755919222e',
        '878fd8c7-f2ee-4756-a5b9-60fa33407b63',
        '993d4eda-afe8-4464-9e40-a55ed4bc443d',
        '2147e6dd-1d45-4a67-a077-dff8e5421021',
        '40e1d646-e95e-4be2-b47b-0dc58385534d',
        '42c8384b-30ec-4c8e-aeb4-8513de6e7204',
        '081f8eb1-f3cc-485a-b96e-b30f922d71f6',
        '81cfbd06-8f5c-4152-acf1-5fa86c2c7b19',
        '85fd736d-d2bf-4df1-b3f6-e1b436bc72f9',
        '099b28ab-37f9-4617-83fe-0b762019459c',
        '101cecc9-4a4d-4063-add2-bfe44b78f795',
        '44bc1cde-535c-45c5-9a6f-540edc8aa985',
        '9d651522-de7b-4629-93e1-31360a07a7f1',
        '13dbf381-3f92-42a6-bae6-d7daf1c8fff5',
        '22cebf41-cf83-43de-a9b3-3a80a4536aa3',
        '26b3ce1b-bfa2-429b-9c0e-6270f7829bfe',	
        '9d05e322-7e0d-4dad-a6a4-a525ba48e2de',
        '7e73964e-9a63-4024-8f40-19485ca304f9',
        '7b10b1e0-984f-4de6-b555-7dc208797556',
        '6fd3ce73-271f-4cda-9bac-4a013271d11d',
        '6f01fac5-fd10-4101-be97-3e66e5e62c42',
        '0b5c08ad-4e7a-4f4d-b030-a7ffd2bfa829',
        '1bbbc50d-3a1a-46b4-887a-257c09bb6f1c',	
        '2e22e0ec-bfb3-42ab-a0be-987a96d59a2b',
        '3c12d264-f046-4d21-b6c7-c2b00d2971bd',
        '3eb74806-d0c0-4877-94ec-04c4f4345bd6', 
        '3f100e46-39e1-41c0-9032-e0579445f7a3',
        '4fce9ab8-d0cb-4ec7-a093-37f28f785a35',
        '5b8d1ad5-4b12-49e2-80fd-6ab83e907b9d',
        '5b28f14b-5f15-4968-af8c-206080b508d5',
        '5dc67e26-9e4e-4df2-9ac9-04f84cbff848',
        '5e8cd614-2ef3-43bd-a287-6fcfe71ecdb3',
        '6aefb7c6-50a9-4526-ab75-f8f80c8bb727',
        '6c9bc334-4d3b-4afe-b7c5-db4a43165695',
        '6dd986f7-cc7b-4672-b426-07418b284f19',
        '6e8ffc78-f041-4e56-acc3-a0f923e98498'

    ]
    
    src_path = r'C:\Dataset\reference'
    dst_path = r'C:\Dataset\reference-cropped'
    #Create folder structure
    src_videos = os.listdir(src_path)
    for video in src_videos:
        if(os.path.isdir(dst_path+'\\'+video)):
            continue
        if(video=='data_row.csv'):
            continue
        os.mkdir(dst_path+'\\'+video)
        os.mkdir(dst_path+'\\'+video+"/images/")    
        #copy label_row json file
        open(dst_path+'\\'+video+"/label_row.json", "w").write(open(src_path+'\\'+video+"/label_row.json").read())
        #Save images
        images = os.listdir(src_path+'\\'+video+"/images/")
        for image in images:
            img = Image.open(src_path+'\\'+video+"/images/"+image)
            img=img.resize((960,540))
            if video  in type_1:
                single_video = add_circular_blob(img, center=[img.width//2,img.height//2], radius=315)
                ret = crop_video(single_video, x=[171,800], y=[10,489])
            elif video  in type_6:
                single_video = add_circular_blob(img, center=[343,267], radius=313)
                ret = crop_video(single_video, x=[30,656], y=[26,509])
            else:
                continue
            ret=ret.resize((625,480))
            ret.save(dst_path+'\\'+video+"/images/"+image)
    return

#crop_cloudfactory_dataset()
crop_reference_dataset()
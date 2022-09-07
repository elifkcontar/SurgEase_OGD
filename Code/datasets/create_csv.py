from ctypes.wintypes import HWINSTA
import os
from random import sample
from encord.client import EncordClient
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

def get_fixed_video_names(label_rows_data_titles):
        label_rows_data_titles_fixed = {}
        for data_title in label_rows_data_titles:            
            fixed_data_title = data_title
            label_rows_data_titles_fixed[fixed_data_title] = data_title
        return label_rows_data_titles_fixed

def create_csv(folder_names, frames_root_path, cord_project_ID, cord_API_key, name=''):

    df = pd.DataFrame(columns =['frame_path', 'video_name', 'vascular_pattern_scores', 'bleeding_score', 'erosions_score', 'UCEIS_score'])   

    client = EncordClient.initialise(cord_project_ID, cord_API_key) 
    project = client.get_project()
    
    label_rows = project["label_rows"]
    label_rows_data_titles = [item["data_title"] for item in label_rows]
    label_rows_data_titles_fixed = get_fixed_video_names(label_rows_data_titles)         
    
    folder_names = folder_names

    for folder in folder_names: 
        print(folder)    
        folder_labelled = False
        folder_name_on_cord_project = label_rows_data_titles_fixed.get(folder, -1)
        if folder_name_on_cord_project is not -1:                            
            folder_label_hash = [item["label_hash"] for item in label_rows if item["data_title"] == folder_name_on_cord_project][0]
            folder_data_hash = [item["data_hash"] for item in label_rows if item["data_title"] == folder_name_on_cord_project][0]
            
            annotations = client.get_label_row(folder_label_hash)
            #print(annotations)
            folder_path = os.path.join(frames_root_path, folder, "frames")
            
            
            #for item in os.scandir(folder_path):
            for item in sorted(os.scandir(folder_path), key=os.path.getmtime):
                if item.is_file() and item.name.endswith(".png"):                    

                    file_name = item.name[:-4]
                    frame_annotations_ = annotations["data_units"][folder_data_hash]["labels"].get(file_name, -1)
                    if frame_annotations_ is not -1:
                        frame_object_annotations = frame_annotations_["objects"]
                        for annotation in frame_object_annotations:
                            name=''
                            value=''
                            h=-1
                            w=-1
                            x=-1
                            y=-1

                            if(annotation['shape']=='bounding_box'):
                                try:
                                    name = annotation["name"]
                                    value = annotation["value"]
                                    h = annotation["boundingBox"]['h']
                                    w = annotation["boundingBox"]['w']
                                    x = annotation["boundingBox"]['x']
                                    y = annotation["boundingBox"]['y']

                                except:
                                    name = ''
                            else:
                                continue

                            if name=='':
                                continue
                            else:
                                new_row = {'frame_path':item.path, 
                                'video_name':folder,
                                'label_name': name,                        
                                'label_value': value,
                                'h': h,
                                'w': w, 
                                'x': x,
                                'y': y}
                                df=df.append(new_row, ignore_index=True)
        else:
            print("Folder "+folder+" is not found in the Cord project, check folder/video names!")

        df.to_csv('label_all.csv')

def sample_csv(src_csv_path, dst_csv_path, sampling=50):
    df_final = pd.DataFrame(columns =['frame_path', 'video_name', 'vascular_pattern_scores', 'bleeding_score', 'erosions_score', 'UCEIS_score'])   
    df = pd.read_csv(src_csv_path)
    for i in range (df.shape[0]):
        if((i%sampling)==0):
            df_final=df_final.append(df.iloc[i], ignore_index=True)
    df_final.to_csv(dst_csv_path)

        
frames_root_path =  "C:\\Users\\ElifKübraÇontar\\OneDrive - Surgease Innovations Ltd\\Desktop\\OGD\\Dataset\\data_all_frames_540x960"
cord_project_ID = 'ff31ec11-fd68-4db9-a8c2-a4e97cbb4b7e'
cord_API_key = 'cR2DFqKUVsnZCzJ_xvG7ZkxniWv7MeINnAYI8dNMnHQ'

file = open(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\OGD\Code\splitted_folder_names\train_folders.txt')
train_folders = file.read().splitlines()
file.close()

file = open(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\OGD\Code\splitted_folder_names\val_folders.txt')
val_folders = file.read().splitlines()
file.close()

file = open(r'C:\Users\ElifKübraÇontar\OneDrive - Surgease Innovations Ltd\Desktop\OGD\Code\splitted_folder_names\test_folders.txt')
test_folders = file.read().splitlines()
file.close()


all_folders=[]
all_folders.extend(train_folders)
all_folders.extend(val_folders)
all_folders.extend(test_folders)
print(len(all_folders))
folder_names = all_folders
create_csv(folder_names, frames_root_path, cord_project_ID, cord_API_key)     
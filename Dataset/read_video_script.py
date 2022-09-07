import os
#import cv2
import shutil

from encord.client import EncordClient
client = EncordClient.initialise(
  '5c349378-3e58-4028-af97-96ad9d137dd3',  # Project ID
  'fL_ktLRxfD90iQayw2dLVg-GVkPxQZVq9p6C6cvpvlg'  # API key
)
# Get and print project info (labels, datasets)
project = client.get_project()


video_url_list = []
video_name_list= []
target_folder = "/IBD/data_all_frames_540x960"
all_video_name_list = [item["data_title"] for item in project["label_rows"]]
labeled_video_name_list = sorted([video_info["data_title"] for video_info in project["label_rows"] if video_info["label_status"] == "LABELLED"])
reviewed_video_name_list = sorted([video_info["data_title"] for video_info in project["label_rows"] if video_info["label_status"] == "REVIEWED"])
reviewed2_video_name_list = sorted([video_info["data_title"] for video_info in project["label_rows"] if video_info["label_status"] == "REVIEWEDTWICE"])

print('Number of labeled video is: ',len(labeled_video_name_list))
print('Number of reviewed video is: ',len(reviewed_video_name_list))
print('Number of reviewed twice video is: ',len(reviewed2_video_name_list))
print(labeled_video_name_list)
print(reviewed_video_name_list)
print(reviewed2_video_name_list)



#Load video names and signed urls && Change environment cv2
with open("labeled_video_name_list", "rb") as fp:   # Unpickling
    labeled_video_name_list = pickle.load(fp)
with open("video_url_list", "rb") as fp:   # Unpickling
    video_url_list = pickle.load(fp)
print(labeled_video_name_list)

#Save video names and signed urls
import pickle
with open("labeled_video_name_list", "wb") as fp:   #Pickling
    pickle.dump(labeled_video_name_list, fp)
with open("video_url_list", "wb") as fp:   #Pickling
    pickle.dump(video_url_list, fp)


video = labeled_video_name_list[0]
print(video)
for no in range(len(project['label_rows'])):
        if project['label_rows'][no]["data_title"] == video:
            item = no

item_label_hash = project['label_rows'][item]['label_hash']
item_data_hash = project['label_rows'][item]['data_hash']

#sample_label = client.get_label_row(item_label_hash)
data_row = client.get_data(item_label_hash, get_signed_url=True)
print(data_row)
a=0

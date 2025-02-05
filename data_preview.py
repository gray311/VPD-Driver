import os
import json
import copy
import random
from nuscenes_dataset import NuscenesLoader
from tqdm import tqdm
from PIL import Image
# '''
# [
#     {
#         "id": ,
#         "video": , 
#         "image": [],
#         "source": ,
#         "conversations": [{"from": "human", "value": "<image>\nAnalyze the two consecutive images and provide a description of how the scene develops from the first to the second, highlighting any shifts or movements."}, {"from": "gpt", "value": "Over time, the outdoor setting with the wire fence separating the verdant foliage from the pathway remains largely unchanged. However, a subtle shift in the arrangement of objects on the ground becomes apparent. The backpack or bag has been moved slightly closer to the fence, while the blue plastic bottle has been repositioned nearer to the backpack. A white object, possibly debris or litter, can be spotted near the backpack, hinting at a change in the immediate surroundings. Despite these minor adjustments, the overall environment and background continue to mirror the previous scene, maintaining a sense of continuity and natural evolution."}],
#         "objects": [ 
#             {
#                 "image_name": [
#                     "Category": "Vehicle",
#                     "Status": "Moving",
#                     "Visual_description": "Brown SUV.",
#                     "2d_bbox": [
#                         966.6,
#                         403.3,
#                         1224.1,
#                         591.7
#                     ],
#                 ]   
#             }
#         ]
#     },
# ]
# '''

# loader = NuscenesLoader(version="v1.0-trainval", dataroot="/home/ec2-user/SafoLab-Amazon-Challenge/tmp/workspace/data/nuscenes")
    
# def calculate_iou(box1, box2):
#     """
#     Calculate the Intersection over Union (IoU) of two bounding boxes.

#     Parameters:
#         box1 (list): [x_min, y_min, x_max, y_max] for the first bounding box
#         box2 (list): [x_min, y_min, x_max, y_max] for the second bounding box

#     Returns:
#         float: IoU value
#     """
#     # Extract coordinates
#     x1_min, y1_min, x1_max, y1_max = box1
#     x2_min, y2_min, x2_max, y2_max = box2

#     # Calculate the coordinates of the intersection rectangle
#     inter_x_min = max(x1_min, x2_min)
#     inter_y_min = max(y1_min, y2_min)
#     inter_x_max = min(x1_max, x2_max)
#     inter_y_max = min(y1_max, y2_max)

#     # Compute the area of intersection rectangle
#     inter_width = max(0, inter_x_max - inter_x_min)
#     inter_height = max(0, inter_y_max - inter_y_min)
#     inter_area = inter_width * inter_height

#     # Compute the area of both bounding boxes
#     box1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     box2_area = (x2_max - x2_min) * (y2_max - y2_min)

#     # Compute the IoU
#     union_area = box1_area + box2_area - inter_area
#     iou = inter_area / union_area if union_area > 0 else 0

#     return iou


# def process_drivelm_qa_format(key_object_infos, key_object_names, frame_token, qa_list, question_type, frame_num=8):
#     sample = loader.nusc.get('sample', frame_token)
#     sample_data_token = sample['data']["CAM_FRONT"]
#     sd_record = loader.nusc.get('sample_data', sample_data_token)
#     filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])
#     visible_annotations_by_channel, filepath_by_channel = loader.get_annotations_for_sensors(sample)
#     anns = visible_annotations_by_channel["CAM_FRONT"]
#     filepath = filepath_by_channel["CAM_FRONT"]
#     object_descriptions = loader.get_object_description(anns)

#     for k, v in key_object_infos.items():
#         obj_info = key_object_infos[k]
#         bbox_2d = obj_info['2d_bbox']
#         key_object_infos[k]['instance_token'] = None
#         for obj_id, info in object_descriptions.items():
#             bounding_box = info['bounding_box']
#             iou = calculate_iou(bbox_2d, bounding_box)
#             for i in range(4):
#                 if i % 2 == 0:
#                     bounding_box[i] = max(0, bounding_box[i])
#                     bounding_box[i] = min(1600, bounding_box[i])
#                 else:
#                     bounding_box[i] = max(0, bounding_box[i])
#                     bounding_box[i] = min(900, bounding_box[i])

#             if iou > 0.8:
#                 key_object_infos[k]['instance_token'] = info['instance_token']
    
#     objects = [{frame_num - 1: [v for v in key_object_infos.values()]}]
#     image_list = [filepath]
#     prev_token = sample['prev']
#     for i in range(frame_num - 1):
#         if prev_token == "": break
#         sample = loader.nusc.get('sample', prev_token)
#         sample_data_token = sample['data']["CAM_FRONT"]
#         sd_record = loader.nusc.get('sample_data', sample_data_token)
#         filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])
#         visible_annotations_by_channel, filepath_by_channel = loader.get_annotations_for_sensors(sample)
#         anns = visible_annotations_by_channel["CAM_FRONT"]
#         filepath = filepath_by_channel["CAM_FRONT"]
#         object_descriptions = loader.get_object_description(anns)
#         image_list = [filepath] + image_list
#         object_list = []
#         for k, v in key_object_infos.items():
#             instance_token = key_object_infos[k]['instance_token']
#             if instance_token == None: continue
#             for obj_id, info in object_descriptions.items():
#                 if instance_token == info['instance_token']:
#                     obj_info = copy.deepcopy(key_object_infos[k])
#                     obj_info['2d_bbox'] = info['bounding_box']
#                     object_list.append(obj_info)
#         objects = [{frame_num - 2 - i:object_list}] + objects
#         prev_token = sample['prev']

#     # if "/home/ec2-user/SafoLab-Amazon-Challenge/tmp/workspace/data/nuscenes/samples/CAM_FRONT/n015-2018-08-03-15-00-36+0800__CAM_FRONT__1533280042112460.jpg" not in image_list:
#     #     return []
 
#     outputs = []
#     for i, qa in enumerate(qa_list):
#         question = qa['Q']
#         answer = qa['A']
    
#         if ",CAM_FRONT" not in question and ",CAM_FRONT" not in answer: continue
#         if ",CAM_BACK" in question or ",CAM_BACK" in answer: continue
#         if ",CAM_FRONT_" in question or ",CAM_FRONT_" in answer: continue

#         for object_id, key_object in enumerate(key_object_names):
#             object_description = key_object_infos[key_object]['Visual_description']
            
#             if key_object in question:
#                 question = question.replace(key_object, f"[{str(object_id + 1)}]")

#             if key_object in answer:
#                 answer = answer.replace(key_object, object_description.replace(".", " ") + f"[{str(object_id + 1)}]")

#         outputs.append({
#             "id": f"drivelm_{frame_token}_{i}",
#             "video": None,
#             "image": image_list,
#             "source": f"drivelm_{question_type}",
#             "conversations": [
#                 {"from": "human", "value": f"<image>\n{question}"}, 
#                 {"from": "gpt", "value": answer}
#             ],
#             "objects": objects
#         })

#     return outputs


# if __name__ == "__main__":
#     # with open("./workspace/data/init-it/inst_it_dataset_image_51k.json") as f:
#     #     inst_it_data = json.load(f)

#     # print(len(inst_it_data))
#     # print(inst_it_data[0])

#     # with open("./workspace/data/init-it/inst_it_dataset_video_21k.json") as f:
#     #     inst_it_video_data = json.load(f)

#     # print(len(inst_it_video_data))
#     # print(inst_it_video_data[0])


#     with open("./workspace/data/drivelm/v1_0_train_nus.json") as f:
#         drivelm_data = json.load(f)

#     print(len(drivelm_data))

#     cnt = 0
#     processed_qa_list = []
#     for scene_token in tqdm(drivelm_data.keys()):
#         scene_data = drivelm_data[scene_token]
#         for frame_token in scene_data['key_frames'].keys():
#             # if frame_token != "1b45a97a0e5e49fe9cd345dd4bd729c3": continue
#             frame_data = scene_data['key_frames'][frame_token]
#             key_object_infos = frame_data['key_object_infos']
#             key_object_infos = {k:v for k, v in key_object_infos.items() if "CAM_FRONT," in k}
#             key_object_names = [k for k in key_object_infos.keys()]

#             for object_id, key_object in enumerate(key_object_names):
#                 key_object_infos[key_object]['id'] = object_id + 1

#             qa_list = frame_data['QA']['perception'] + frame_data['QA']['prediction'] + frame_data['QA']['planning']
            
#             processed_qa_list = processed_qa_list + process_drivelm_qa_format(key_object_infos, key_object_names, frame_token, frame_data['QA']['perception'], 'perception')
#             processed_qa_list = processed_qa_list + process_drivelm_qa_format(key_object_infos, key_object_names, frame_token, frame_data['QA']['prediction'], 'prediction')
#             processed_qa_list = processed_qa_list + process_drivelm_qa_format(key_object_infos, key_object_names, frame_token, frame_data['QA']['planning'], 'planning')


#     print(len(processed_qa_list))
#     with open("vpd_drivelm200k_test.json", "w") as f:
#         f.write(json.dumps(processed_qa_list))
       
    
    
    
    # sample = loader.nusc.get('sample', "4a0798f849ca477ab18009c3a20b7df2")
    # sample_data_token = sample['data']["CAM_FRONT"]
    # sd_record = loader.nusc.get('sample_data', sample_data_token)
    # filepath = os.path.join(loader.nusc.dataroot, sd_record['filename'])


    # print(sample)
    # print(filepath)
  



# with open("./workspace/data/vpd_drivelm200k_test.json", "r") as f:
#     data = json.load(f)

# print(len(data))
# data_root = "/home/ec2-user/SafoLab-Amazon-Challenge/tmp/workspace/data/vpd-sft/drivelm"

# from tqdm import tqdm 
# cnt = 0
# data_v1 = []
# for line in tqdm(data):
#     image_name = line['image'][-1].split("/")[-1].split(".")[0].strip(" ")
#     image_path = os.path.join(data_root, image_name)
#     mark = False
#     for img_path in line['image']:
#         som_img_path = os.path.join(image_path, img_path.split("/")[-1])
#         try:
#             image = Image.open(som_img_path)
#         except:
#             mark = True

#             print(som_img_path)

#     if mark:
#         cnt += 1
#     else:
#         data_v1.append(line)
    
# print(cnt)
# print(len(data_v1))

# with open("vpd_drivelm200k.json", "w") as f:
#     f.write(json.dumps(data_v1))


# with open("./workspace/data/init-it/inst_it_dataset_image_51k.json") as f:
#     inst_it_data = json.load(f)

# print(len(inst_it_data))
# # print(inst_it_data[0])

# with open("./workspace/data/init-it/inst_it_dataset_video_21k.json") as f:
#     inst_it_video_data = json.load(f)

# print(len(inst_it_video_data))
# print(inst_it_video_data[0])


img_instance_prompts = [
    "Please describe the object [<id>].",
    "What details can you provide about object [<id>]?",
    "Provide a description of object [<id>], including its visual traits.",
    "Focus on object [<id>] and give a comprehensive description.",
    "Can you describe object [<id>] in detail?",
    "What can you tell me about the features of object [<id>]?",
    "Describe the object [<id>].",
    "Please give a detailed description of object [<id>].",
    "What observations can you make about object [<id>]?",
    "Provide an overview of the visual details of object [<id>]."
]

img_caption_prompts = [
    "List all tagged objects in the image.",
    "Please describe this image with tags, specifying the ID of each object.",
    "Please provide a description of this image with the specific number of each object.",
    "Describe this image by listing all the objects along with their tags.",
    "What are the objects in this image? Include their ID numbers in the description.",
    "Provide a detailed description of the image, including each object's tag.",
    "Describe this image with the IDs of all tagged objects and their details.",
    "What objects can you identify in this image? Include their tags in your response.",
    "Summarize the content of this image by mentioning the tags of all objects.",
    "Please list all objects in this image along with their specific ID numbers.",
    "Describe this image by associating each object with its tag and characteristics."
]

"""
[
    {
        "id": ,
        "video": , 
        "image": [],
        "source": ,
        "conversations": [{"from": "human", "value": "<image>\nAnalyze the two consecutive images and provide a description of how the scene develops from the first to the second, highlighting any shifts or movements."}, {"from": "gpt", "value": "Over time, the outdoor setting with the wire fence separating the verdant foliage from the pathway remains largely unchanged. However, a subtle shift in the arrangement of objects on the ground becomes apparent. The backpack or bag has been moved slightly closer to the fence, while the blue plastic bottle has been repositioned nearer to the backpack. A white object, possibly debris or litter, can be spotted near the backpack, hinting at a change in the immediate surroundings. Despite these minor adjustments, the overall environment and background continue to mirror the previous scene, maintaining a sense of continuity and natural evolution."}]
    }
]
"""


# image_vpt_data = []

# for line in inst_it_data:
#     image_path = os.path.join("workspace/data/init-it", line['image_path'])
#     image_id = "image_vpt_" + image_path.split("/")[-1].split(".")[0]
    
#     cnt = 0
#     for img_id, response in line['instance_level_caption'].items():
#         question = random.choice(img_instance_prompts)
#         if img_id.isdigit():
#             if cnt > 4: 
#                 break
#             question = question.replace("<id>", str(img_id))
#         elif len(img_id) > 2:
#             numbers = img_id.split(",")
#             if len(numbers) == 2:
#                 formatted_string = ", ".join(f"[{num}]" for num in numbers[:-1]) + f" and [{numbers[-1]}]"
#             else:
#                 formatted_string = ", ".join(f"[{num}]" for num in numbers[:-1]) + f", and [{numbers[-1]}]"
#             question = question.replace("[<id>]", formatted_string)
#             question = question.replace("object","objects").replace("its", "their")      
#         else:
#             continue

#         cnt += 1
#         unique_id = f"{image_id}_{str(cnt)}"
  

#         example = {}
#         example['id'] = unique_id
#         example['video'] = None
#         example['image'] = image_path
#         example['source'] = "init-it_vpt"
#         example['conversations'] = [
#             {
#                 "from": "human",
#                 "value": f"<image>\n{question}",
#             },
#             {
#                 "from": "gpt",
#                 "value": response
#             }
#         ]

#         image_vpt_data.append(example)

#     cnt += 1
#     unique_id = f"{image_id}_{str(cnt)}"
#     question = random.choice(img_caption_prompts)
#     # question = question.replace("<id>", str(img_id))

#     example = {}
#     example['id'] = unique_id
#     example['video'] = None
#     example['image'] = image_path
#     example['source'] = "init-it_vpt"
#     example['conversations'] = [
#         {
#             "from": "human",
#             "value": f"<image>\n{question}",
#         },
#         {
#             "from": "gpt",
#             "value": line['image_level_caption']
#         }
#     ]

#     image_vpt_data.append(example)

# with open("vpd_inst-it-img281k.json", "w") as f:
#     f.write(json.dumps(image_vpt_data))

# print(len(image_vpt_data))



# video_vpt_data = []
# for line in inst_it_video_data:
#     video_path = os.path.join("workspace/data/init-it", line['video_path'])
#     image_list = os.listdir(video_path)
#     image_list = sorted(image_list)

#     image_list = [os.path.join(video_path, item) for item in image_list]

#     if len(image_list) > 16: continue

#     video_id = line['video_path'].replace("train/", "").replace("/", "_")

#     random.shuffle(line['question_answer_pairs'])
#     for i, qas in enumerate(line['question_answer_pairs']):
#         if i >= 5: break
#         question = qas['question']
#         answer = qas['answer']


#         example = {}
#         example['id'] = f"{video_id}_{str(i + 1)}"
#         example['video'] = None
#         example['image'] = image_list
#         example['source'] = "init-it_vpt"
#         example['conversations'] = [
#             {
#                 "from": "human",
#                 "value": f"<image>\n{question}",
#             },
#             {
#                 "from": "gpt",
#                 "value": answer
#             }
#         ]

#         video_vpt_data.append(example)

# print(len(video_vpt_data))

# with open("vpd_inst-it-vid98k.json", "w") as f:
#     f.write(json.dumps(video_vpt_data))


# with open("workspace/data/vpd_drivelm91k.json", "r") as f:
#     data_1 = json.load(f)

# with open("workspace/data/vpd_inst-it-img281k.json", "r") as f:
#     data_2 = json.load(f)

# with open("workspace/data/vpd_inst-it-vid98k.json", "r") as f:
#     data_3 = json.load(f)


# from PIL import Image
# data = []
# cnt = 0
# for line in data_1:
#     line['image'] = [item[item.index("workspace"):] for item in line['image']]
#     for img_path in line['image']:
#         try:
#             image = Image.open(img_path)
#         except:
#             cnt += 1

#     data.append(line)


# print(cnt)


# for line in data_2:
 
#     try:
#         image = Image.open(line['image'])
#     except:
#         cnt += 1

#     data.append(line)

# print(cnt)

# max_frame = 0
# for line in data_3:
#     line['image'] = [item[item.index("workspace"):] for item in line['image']]
#     max_frame = max(len(line['image']), max_frame)
#     for img_path in line['image']:
#         try:
#             image = Image.open(img_path)
#         except:
#             cnt += 1
#     data.append(line)

# print(cnt)
# print(max_frame)
# print(len(data))


# with open("./workspace/data/vpd_mix470k_ins-it379k_drivelm91k.json", "w") as f:
#     f.write(json.dumps(data))


with open("./workspace/data/vpd_mix470k_ins-it379k_drivelm91k.json", "r") as f:
    data = json.load(f)

print(len(data))
print(data[0])

data_1 = []
from tqdm import tqdm
for line in tqdm(data):
    if "drivelm" not in line['source']: 
        data_1.append(line)
        continue
    
    image_name = line['image'][-1].split("/")[-1].split(".")[0]
    image_root = os.path.join("./workspace/data/vpd-sft/drivelm", image_name)

    assert os.path.exists(image_root), f"{image_root}"

    image_list = []
    for img_path in line['image']:
        new_img_path = os.path.join(image_root, img_path.split("/")[-1])
        assert os.path.exists(new_img_path), f"{new_img_path}"
        image_list.append(new_img_path)
        from PIL import  Image
        image = Image.open(new_img_path)

    line['image'] = image_list
    data_1.append(line)

print(len(data_1))
with open("./workspace/data/vpd_mix470k_ins-it379k_drivelm91k_V1.json", "w") as f:
    f.write(json.dumps(data_1))



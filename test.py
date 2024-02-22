#  OBJECT DETECTION
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# url = "https://upload.wikimedia.org/wikipedia/commons/2/2f/Prime_Minister_Lee_Kuan_Yew_of_Singapore_Making_a_Toast_at_a_State_Dinner_Held_in_His_Honor%2C_1975.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            + f"{round(score.item(), 3)} at location {box}"
    )

# ## VIDEO CLASSIFICATION
# import av
# import torch
# import numpy as np

# from transformers import AutoImageProcessor, TimesformerForVideoClassification
# from huggingface_hub import hf_hub_download

# np.random.seed(0)


# def read_video_pyav(container, indices):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     '''
#     Sample a given number of frame indices from the video.
#     Args:
#         clip_len (`int`): Total number of frames to sample.
#         frame_sample_rate (`int`): Sample every n-th frame.
#         seg_len (`int`): Maximum allowed index of sample's last frame.
#     Returns:
#         indices (`List[int]`): List of sampled frame indices
#     '''
#     converted_len = int(clip_len * frame_sample_rate)
#     end_idx = np.random.randint(converted_len, seg_len)
#     start_idx = end_idx - converted_len
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices


# # video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = "mancomp.mp4"
# container = av.open(file_path)

# # sample 8 frames
# indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container, indices)

# image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
# model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")

# inputs = image_processor(list(video), return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     logits = outputs.logits

# # model predicts one of the 400 Kinetics-400 classes
# predicted_label = logits.argmax(-1).item()
# print(model.config.id2label[predicted_label])
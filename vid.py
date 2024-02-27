import cv2
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

vidcap = cv2.VideoCapture('womanread.mp4')
success,frame = vidcap.read()
count = 0

# total number of frames
amount_of_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
print(amount_of_frames)
add = 0

while success:

  if count>5:  # temporary limiter
    print('bruh')
    break
  
  cv2.imwrite("frame.jpg", frame)     # save frame as JPEG file      
  success,frame = vidcap.read()
  print('Read a new frame: ', success)
  
  ## OBJECT DETECTION MODEL 
  url = 'frame.jpg'
  image = Image.open(url)
  # you can specify the revision tag if you don't want the timm dependency
  processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
  model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
  inputs = processor(images=image, return_tensors="pt")
  outputs = model(**inputs)
  # convert outputs (bounding boxes and class logits) to COCO API
  # let's only keep detections with score > 0.9
  target_sizes = torch.tensor([image.size[::-1]])
  results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
  reslist = []
  conflist = []
  for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    sent = f"{model.config.id2label[label.item()]}"
    reslist.append(sent)
    conflist.append(f"{round(score.item(), 3)}")
  print("Data:")
  print(reslist)
  
  
  
  # TO NEXT FRAME
  add += 24 # get a frame per second (if 24fps)
  vidcap.set(cv2.CAP_PROP_POS_FRAMES, add)
  count += 1

vidcap.release()
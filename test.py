#  OBJECT DETECTION
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

# http://images.cocodataset.org/val2017/000000039769.jpg
# https://upload.wikimedia.org/wikipedia/commons/2/2f/Prime_Minister_Lee_Kuan_Yew_of_Singapore_Making_a_Toast_at_a_State_Dinner_Held_in_His_Honor%2C_1975.jpg

def get_metadata(url):
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
    
    reslist = []
    conflist = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        sent = f"{model.config.id2label[label.item()]}"
        reslist.append(sent)
        conflist.append(f"{round(score.item(), 3)}")

    print("Data:")
    print(reslist)
    print("\nConfidence:")
    print(conflist)

    return reslist

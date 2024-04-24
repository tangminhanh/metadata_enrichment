try:
    import cv2
    from transformers import DetrImageProcessor, DetrForObjectDetection
    import torch
    from PIL import Image
    import requests
    import sys
    print("All imports ok ...")
except Exception as e:
    print("Error Imports : {} ".format(e))

def get_metadata(image):

    # you can specify the revision tag if you don't want the timm dependency
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50", revision="no_timm")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9)[0]

    reslist = []
    conflist = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        sent = f"{model.config.id2label[label.item()]}"
        reslist.append(sent)
        conflist.append(f"{round(score.item(), 3)}")

    return reslist


def lambda_handler(event, context):
    url = event.get('url')
    vidcap = cv2.VideoCapture(f'{url}')
    success, frame = vidcap.read()
    count = 0
    add = 0
    reslist = []

    while success:
        if count > 5:  # temporary limiter
            break

        success,frame = vidcap.read()
        img = Image.fromarray(frame[:, :, ::-1])
        ret = get_metadata(img)
        for val in ret:
            if val not in reslist:
                reslist.append(val)

        # TO NEXT FRAME
        add += 24  # get a frame per second (if 24fps)
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, add)
        count += 1

    vidcap.release()
    print(reslist)
    return reslist

# test_event = {
#     "url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerFun.mp4"
# }
# lambda_handler(test_event, '')
import cv2
import numpy as np
import torch
import torchvision

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = im.shape[:2] 
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """
    YOLOv8 Robust NMS implementation.
    Input: [Batch, 84, 8400]
    Output: List of detections
    """
    # Numpy -> Tensor
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction)

    # prediction = prediction.cpu() 

    # Shape Check: [Batch, 84, 8400] -> [Batch, 8400, 84] (Transpose)
    if prediction.shape[1] == 84:
        prediction = prediction.transpose(1, 2)

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]

    for xi, x in enumerate(prediction):  # image index, image inference
        # x shape: [8400, 84]
        
        # Sütunları ayır
        # İlk 4 sütun: bbox (cx, cy, w, h)
        # Kalan 80 sütun: class probabilities
        box = x[:, :4]
        cls = x[:, 4:]

        # Her kutu için en yüksek sınıf skorunu bul
        conf, j = cls.max(1, keepdim=True)
        
        # Confidence threshold altındakileri ele
        mask = conf.view(-1) > conf_thres
        x = torch.cat((box, conf, j.float()), 1)[mask]

        if not x.shape[0]:
            continue

        # Box convert (cx, cy, w, h) -> (x1, y1, x2, y2)
        x[:, :4] = xywh2xyxy(x[:, :4])

        # Batched NMS (Sınıfları birbirinden ayırarak çakışma kontrolü)
        c = x[:, 5:6] * 7680
        boxes, scores = x[:, :4] + c, x[:, 4]
        
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]

    return output

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
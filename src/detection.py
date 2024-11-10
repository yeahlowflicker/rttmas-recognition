import cv2
import torch
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
from imutils import paths
import imutils

# Model
vehicles_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=0)
license_plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/snmg/rttmas-recognition/src/weights/license.pt', verbose=0)


DESIRED_CLASSES = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]


def detect_vehicles(model, img):

  # Images
  # img = cv2.imread(img_path)[..., ::-1]  # OpenCV image (BGR to RGB)

  # Inference
  results = model([img], size=640) # batch of images

  detections = results.pred[0]

  crops = results.crop(save=False)

  final_results = []

  # crop is a struct: ['box', 'conf', 'cls', 'label', 'im']
  for crop in crops:
    label = crop['label'].split(' ')[0]
    if label in ['car', 'motorcycle', 'truck', 'bus']:
      final_results.append((crop['im'], crop['box'], crop['conf'], crop['cls']))

  del results
  del crops

  return final_results, detections



def detect_license_plate(model, img):

  # Images
  # img = cv2.imread(img_path)[..., ::-1]  # OpenCV image (BGR to RGB)

  # Inference
  results = model([img], size=640) # batch of images

  crops = results.crop(save=False)

  final_results = []

  for crop in crops:
    img = crop['im']
    h, w = img.shape[:2]
    zoomed_height = int(h * 1.2)
    zoomed_width = int(w * 1.2)
    zoomed_img = cv2.resize(img, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    crop_y = (zoomed_height - h) // 2
    crop_x = (zoomed_width - w) // 2
    zoomed_img_cropped = zoomed_img[crop_y:crop_y + h, crop_x:crop_x + w]
    final_results.append((zoomed_img_cropped, crop['box']))

  del results
  del crops

  return final_results



def preprocess_ocr_input(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Scale up the image by a factor of 5
    img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)

    return img



def perform_ocr(img):

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)  # You can change the language as needed

    # Convert the OpenCV image (BGR) to RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform OCR
    result = ocr.ocr(image_rgb, cls=True)

    # Extract text from the result
    text = None
    try:
        for line in result:
            text, confidence = line[0][1]
    except:
        pass
        
    return text



def detect_image(img):
    vehicles, detections = detect_vehicles(vehicles_model, img)

    plateCounter = 0
    plate_numbers = []

    for r in detections:
        x1, y1, x2, y2 = r[:4]
        w, h = x2 - x1, y2 - y1
        x, y, w, h = int(x1), int(y1), int(w), int(h)
        confidence = r[4]
        cls = int(r[5])

        if not cls in DESIRED_CLASSES:
            continue

        plate_recognized = False

        vxmin, vymin, vxmax, vymax = x, y, x+w, y+h

        vimg = img[vymin:vymax, vxmin:vxmax]

        vimgh, vimgw = vimg.shape[:2]
        if vimgh == 0 or vimgw == 0:
            continue


        plates = detect_license_plate(license_plate_model, vimg)



        for p in plates:
            
            pimg, box = p
            xmin, ymin, xmax, ymax = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
            cv2.rectangle(img, (xmin+vxmin, ymin+vymin), (xmax+vxmin, ymax+vymin), (0,0,255), 1)

            pimg = preprocess_ocr_input(pimg)

            plate_number = perform_ocr(pimg)

            if plate_number != None:
                plate_numbers.append(plate_number)
                plate_recognized = True
                cv2.putText(img, f'{plate_number}', (vxmin+10, vymax-10), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 255, 255), 1)
                if plateCounter == 0:
                    cv2.imwrite('pout.png', pimg)

            plateCounter+=1


        if plate_recognized:
            cv2.rectangle(img, (vxmin, vymin), (vxmax,vymax), (0, 255, 255), 1)
        else:
            cv2.rectangle(img, (vxmin, vymin), (vxmax,vymax), (0, 255, 0), 1)


    cv2.imwrite('output.png', img)

    return plate_numbers



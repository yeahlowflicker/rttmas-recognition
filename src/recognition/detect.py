from .ocr import *
from .postprocess import *
from .utils import *

import cv2
import torch
from PIL import Image

# Model
vehicles_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', verbose=0)
license_plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/snmg/rttmas-recognition/src/weights/license.pt', verbose=0)


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




from deep_sort_realtime.deepsort_tracker import DeepSort, Detection
tracker = DeepSort(max_age=20)

DESIRED_CLASSES = [
    2,  # car
    3,  # motorcycle
    5,  # bus
    7,  # truck
]

tracked_plate_data = {}
plate_data_clear_counter = 0

def full_detect(img, redis_server):
  global tracked_plate_data
  global plate_data_clear_counter

  response_payload = []

  if plate_data_clear_counter >= 50:
    tracked_plate_data = {}
    plate_data_clear_counter = 0
  else:
    plate_data_clear_counter += 1

  vehicles, detections = detect_vehicles(vehicles_model, img)


  tracking_results = []

  # xywhs = xyxy2xywh(detections[:, 0:4])
  # confs = detections[:, 4]
  # clss = detections[:, 5]

  # for i in range(len(detections)):
  #   box = xywhs[i].cpu().numpy()
  #   conf = confs[i].item()
  #   cls = clss[i].item()
  #   detection = [box, conf, cls]
  #   tracking_results.append(detection)

  for r in detections:
    x1, y1, x2, y2 = r[:4]
    w, h = x2 - x1, y2 - y1
    coordinates = list((int(x1), int(y1), int(w), int(h)))
    conf = r[4]
    cls = int(r[5])
    tracking_results.append((coordinates, conf, cls))

  trackers = tracker.update_tracks(tracking_results, frame=img)


  for track in trackers:
    # if not track.is_confirmed():
    #     continue

    confidence = track.get_det_conf()
    if not confidence or confidence < .5:
      continue

    class_index = track.get_det_class()
    if not class_index or not class_index in DESIRED_CLASSES:
      continue
    

    track_id = track.track_id

    xywh = track.to_ltrb()
    x, y, w, h = int(xywh[0]), int(xywh[1]), int(xywh[2]), int(xywh[3])

    vxmin, vymin, vxmax, vymax = x, y, w, h

    vimg = img[vymin:vymax, vxmin:vxmax]
    # cv2.imshow(vimg)

    vimgh, vimgw = vimg.shape[:2]
    if vimgh == 0 or vimgw == 0:
      continue

    if not track_id in tracked_plate_data:
      tracked_plate_data[track_id] = {}
      # r.hmset(track_id, {})

    # Find the most probable plate sequence
    try:
      # tracked_plate_data = redis_server.hgetall(track_id)
      most_probable_plate = max(tracked_plate_data[track_id], key=tracked_plate_data[track_id].get)
    except:
      most_probable_plate = ''


    cv2.rectangle(img, (vxmin, vymin), (vxmax,vymax), (0, 255, 0), 2)
    cv2.putText(img, f'{str(track.track_id)} {most_probable_plate}', (vxmin, vymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    payload = {
      'track_id': track_id,
      'car_box': [x, y, w, h],
      'plate_no': most_probable_plate,
    }

    plates = detect_license_plate(license_plate_model, vimg)


    for i, p in enumerate(plates):
      pimg, box = p

      tmp_path = f'/home/snmg/rttmas-recognition/src/tmp/p{i}.png'

      cv2.imwrite(tmp_path, pimg)

      result = detect_paddleocr(tmp_path)

      xmin, ymin, xmax, ymax = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())

      if result:
        s = proper_license_plate(result)
        pattern_regex, pattern_name = identify_license_plate_patterns(s)
        converted_value = format_license_plate(s, pattern_name)

        # Update frequency of this detected sequence
        if converted_value:
          if not converted_value in tracked_plate_data[track_id]:
          # if not r.hexists(str(track_id), str(converted_value)):
            tracked_plate_data[track_id][converted_value] = 1
          #   r.hset(track_id, converted_value, 1)
          else:
            tracked_plate_data[track_id][converted_value] += 1
            # redis_server.hincrby(track_id, converted_value, 1)


        # Draw the plate bounding box and sequence
        cv2.rectangle(img, (xmin+vxmin-10, ymin+vymin-10), (xmax+vxmin, ymax+vymin), (0,0,255), 2)

        payload['plate_box'] = [xmin+vxmin-10, ymin+vymin-10, xmax+vxmin, ymax+vymin]
        
        # cv2.putText(img, str(most_probable_plate), (xmin+vxmin,ymax+vymin+50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # if converted_value and len(converted_value) <= 8:
        #   cv2.rectangle(img, (xmin+vxmin-10, ymin+vymin-10), (xmax+vxmin, ymax+vymin), (0,0,255), 2)
        #   cv2.putText(img, str(converted_value), (xmin+vxmin,ymax+vymin+50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # elif pattern_name:
        #   cv2.rectangle(img, (xmin+vxmin-10, ymin+vymin-10), (xmax+vxmin, ymax+vymin), (0,130,255), 2)
        #   cv2.putText(img, str(result), (xmin+vxmin,ymax+vymin+50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 130, 255), 3)

    response_payload.append(payload)

  # for vimg, vbox, vconf, vcls in vehicles:

  #   plates = detect_license_plate(license_plate_model, vimg)

  #   vxmin, vymin, vxmax, vymax = int(vbox[0].item()), int(vbox[1].item()), int(vbox[2].item()), int(vbox[3].item())
  #   # cv2.rectangle(img, (vxmin, vymin), (vxmax, vymax), (0,255,0), 2)

  #   for i, p in enumerate(plates):
  #     pimg, box = p

  #     tmp_path = f'/tmp/p{i}.png'

  #     cv2.imwrite(tmp_path, pimg)

  #     result = detect_paddleocr(tmp_path)

  #     xmin, ymin, xmax, ymax = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())

  #     if result:
  #       s = proper_license_plate(result)
  #       pattern_regex, pattern_name = identify_license_plate_patterns(s)
  #       converted_value = format_license_plate(s, pattern_name)

  #       if converted_value and len(converted_value) <= 8:
  #         cv2.rectangle(img, (xmin+vxmin-10, ymin+vymin-10), (xmax+vxmin, ymax+vymin), (0,0,255), 2)
  #         cv2.putText(img, str(converted_value), (xmin+vxmin,ymax+vymin+50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
  #       elif pattern_name:
  #         cv2.rectangle(img, (xmin+vxmin-10, ymin+vymin-10), (xmax+vxmin, ymax+vymin), (0,130,255), 2)
  #         cv2.putText(img, str(result), (xmin+vxmin,ymax+vymin+50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 130, 255), 3)


  #     # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
  #     # cv2.rectangle(img, (xmin+vxmin, ymin+vymin), (vxmax-xmax, vymax-ymax), (0,255,255), 2)

  #     # print(f'=> {result}')

  #     # print(detect_paddleocr(img_path))
  #     # detect(p.astype(np.uint8))
  #     # cv2.imshow(im)

  del tracking_results
  del trackers

  return img, response_payload

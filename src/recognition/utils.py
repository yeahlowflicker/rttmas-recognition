import cv2
import numpy as np
import math
from scipy import ndimage

def is_tounching_image_boundary(imgw, imgh, x, y, w, h):
  if x <= 0 or y <= 0:
    return True
  if x+w >= imgw:
    return True
  if x+h >= imgh:
    return True
  return False



def subdivide(avgw, avgh, x,y,w,h):
  results = []
  if w >= avgw*.8:
    _w = int(w/2)
    x1, x2 = x, int(x+_w)
    b1 = [x1,y,_w,h]
    b2 = [x2,y,_w,h]
    results.extend(subdivide(avgw, avgh, x1,y,_w,h))
    results.extend(subdivide(avgw, avgh, x2,y,_w,h))
  else:
    results.append([x,y,w,h])

  return results




def filtering(im_gray):

  # Threshold the image in order to find contours
  ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


  imgh, imgw = im_th.shape[:2]

  # Invert black and white, to perform erosion and dilation correctly
  im_th = cv2.bitwise_not(im_th)

  # Scale up the image to perform fine-erosion
  im_th = cv2.resize(im_th, dsize=(imgw*5,imgh*5))

  # Perform erosion
  im_th = cv2.erode(im_th, np.ones((8,8), np.uint8))

  # Scale down the image back to the original size
  im_th = cv2.resize(im_th, dsize=(imgw, imgh))

  # Perform dilation
  im_th = cv2.dilate(im_th, np.ones((3,3), np.uint8))

  # Invert back the colors
  im_th = cv2.bitwise_not(im_th)

  # Perform threshold again, to darken the gray pixels
  ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

  return im_th




def find_boxes(im_th, cvt_gray=False, no_filter=False, allow_touch_boundary=False):

  if cvt_gray:
    im_th = cv2.cvtColor(im_th, cv2.COLOR_BGR2GRAY)

  if not no_filter:
    im_th = filtering(im_th)

  imgh, imgw = im_th.shape[:2]

  # Find contours in the image
  ctrs,hier = cv2.findContours(im_th, cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

  bboxes=[cv2.boundingRect(c) for c in ctrs]
  sorted_bboxes = sorted(bboxes,key=lambda b:b[0])


  avgh, avgw = 0, 0
  counth, countw = 0, 0


  ### STAGE 1: Capture all boxes which sizes are in the disired range ###
  boxes_stage1 = []
  mask = np.zeros_like(im_th)

  for num,i_bboxes in enumerate(sorted_bboxes):
      [x,y,w,h]=i_bboxes

      if is_tounching_image_boundary(imgw,imgh,x,y,w,h) and not allow_touch_boundary:
        continue

      if h>imgh*.4 and w < imgw*.4:
        avgh += h
        avgw += w
        counth += 1
        countw += 1
        boxes_stage1.append([x,y,w,h])

  # draw_bounding_boxes(im_th, boxes_stage1)

  avgh /= counth if counth > 0 else 1
  avgw /= countw if countw > 0 else 1


  ### STAGE 2: Subdivide boxes (if they stick together) ###
  boxes_stage2 = []

  # For each rectangular region, calculate HOG features and predict
  # the digit using our logistic regression model.
  plate_char=[]
  for num,i_bboxes in enumerate(sorted_bboxes):
      [x,y,w,h]=i_bboxes

      # If the box height is higher than 40% of the image height,
      # and is within 1.3x of the average box height
      if h > imgh*.4 and h <= avgh*1.3:

        # If the box width is within 20% of the image width:
        if w <= imgw*.2:
          boxes_stage2.append([x,y,w,h])

        # Otherwise, if the box width is within 1x and 2.5x of the average box width,
        # split it it half
        elif w >= avgw and w <= avgw*2.5:
          boxes_stage2.extend(subdivide(avgw, avgh, x, y, w, h))


  ### STAGE 3: Find and remove inscribed boxes ###
  box_indices_to_skip = []
  for j in range(len(boxes_stage2)):
    (x2, y2, w2, h2) = boxes_stage2[j]
    for i in range(len(boxes_stage2)):
      (x1, y1, w1, h1) = boxes_stage2[i]
      if x1>x2 and (x1+w1)<(x2+w2):
        box_indices_to_skip.append(i)

  digit_boxes = []

  ### STAGE 4: Filter boxes and augment the image ###
  for i, box in enumerate(boxes_stage2):
    if i in box_indices_to_skip:
      continue
    [x,y,w,h] = box
    digit_boxes.append(box)

  return digit_boxes, avgw, avgh




def augment(filepath, i, verbose=0):
  # Read the input image
  im = cv2.imread(filepath)

  if verbose == 1:
    cv2.imshow(im)

  # Convert to grayscale
  im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  imgh, imgw = im_gray.shape[:2]


  digit_boxes, avgw, avgh = find_boxes(im_gray, allow_touch_boundary=True)


  if len(digit_boxes) == 0:
    cv2.imwrite(f'/home/snmg/rttmas-recognition/src/tmp/{i}_nocrop.png', im)
    cv2.imwrite(f'/home/snmg/rttmas-recognition/src/tmp/{i}.png', im)
    return

  y2 = int(digit_boxes[-1][1] + digit_boxes[-1][3] / 2)
  y1 = int(digit_boxes[0][1] + digit_boxes[0][3] / 2)
  x2 = int(digit_boxes[-1][0] + digit_boxes[-1][2] / 2)
  x1 = int(digit_boxes[0][0] + digit_boxes[0][2] / 2)

  slope = (y2-y1) / (x2-x1) if (x2-x1 != 0) else 0
  rads = math.atan(slope)
  degrees = math.degrees(rads)
  if verbose == 1:
    print(f'Angle: {degrees}')

  targetW = imgw / 4
  scaleW = targetW / avgw

  result = ndimage.rotate(im_gray, degrees, reshape=False, cval=255)

  result_nocrop = result

  x1, y1, x2, y2 = imgw, imgh, 0, 0
  for d in digit_boxes:
    [x,y,w,h] = d
    x1 = min(x1, x)
    y1 = min(y1, y)
    x2 = max(x2, x+w)
    y2 = max(y2, y+h)

  result = result[y1:y2,x1:x2]
  sx = imgw

  if verbose == 1:
    print(f'Width scale: {scaleW}')
  if scaleW > 1.0:
    sx = int(imgw * scaleW)
    dim = (sx, imgh)
    result = cv2.resize(result, dim)

  # scale_factor = 800 / sx
  # finalW = 800
  # finalH = min(int(finalW / scale_factor), 500)
  # finalW = 800
  # finalH = 300
  # result = cv2.resize(result, (finalW, finalH))

  imgh, imgw = result_nocrop.shape[:2]
  result_nocrop = cv2.resize(result_nocrop, dsize=(imgw*2, imgh*2))

  imgh, imgw = result.shape[:2]
  result = cv2.resize(result, dsize=(imgw*2, imgh*2))

  result_nocrop = result_nocrop
  result = filtering(result)

  cv2.imwrite(f'/home/snmg/rttmas-recognition/src/tmp/{i}_nocrop.png', result_nocrop)
  cv2.imwrite(f'/home/snmg/rttmas-recognition/src/tmp/{i}.png', result)

  # draw_bounding_boxes(im, digit_boxes)
  # cv2.imshow(im)

  # border_size = 3
  # result = cv2.copyMakeBorder(
  #   result,
  #   top=border_size,
  #   bottom=border_size,
  #   left=border_size,
  #   right=border_size,
  #   borderType=cv2.BORDER_CONSTANT,
  #   value=[255,255,255]
  # )

  # digit_boxes, avgw, avgh = find_boxes(result, cvt_gray=True)

  # draw_bounding_boxes(result, digit_boxes)
  if verbose == 1:
    cv2.imshow(result_nocrop)
    cv2.imshow(result)




def detect(filepath):
  # Read the input image
  im = cv2.imread(filepath)
  # cv2.imshow(im)

  # Convert to grayscale
  im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  im_th = im_gray


  imgh, imgw = im_th.shape[:2]

  avgh, avgw = 0, 0
  counth, countw = 0, 0

  # # For each rectangular region, calculate HOG features and predict
  # # the digit using our logistic regression model.
  plate_char=[]


  boxes_stage2, avgw, avgh = find_boxes(im_th, cvt_gray=False, no_filter=False, allow_touch_boundary=True)

  draw_bounding_boxes(im, boxes_stage2)

  ### STAGE 4: Perform OCR prediction ###
  for i, box in enumerate(boxes_stage2):

    [x,y,w,h] = box
    # Make the rectangular region around the digit
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
    roi=im_gray[y:y+h,x:x+w]
    # Resize the image
    roi = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
    # Calculate the HOG features
    # use the same parameters used for training
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1))
    nbr = clf.predict(np.array([roi_hog_fd]))

    cv2.putText(im, str((nbr[0])), (x,y+h),cv2.FONT_HERSHEY_SIMPLEX,
                .5, (0, 0, 255), 1)
    plate_char.append(str(nbr[0]))


  print('=> ' + ''.join(plate_char) if len(plate_char) > 0 else '(None)')

  # im = cv2.resize(im, dsize=(imgw*2,imgh*2))
  # cv2.imshow(im)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y
# PaddleOCR (Preferred)

import os
import time
from .utils import *
from paddleocr import PaddleOCR, draw_ocr

# enable_mkldnn: Hardware acceleration (multi-core CPU)
ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False, enable_mkldnn=False)


def _do_paddleocr(path):
  try:
    result = ocr.ocr(path, cls=True)
    result = result[0]
    if result:
      txts = [line[1][0] for line in result]
      final = txts[0]
      return final
  except:
    return None
  return None

def detect_paddleocr(src_path):

  # augment(f'{folder_path}/{filename}', i, verbose=0)
  i = 0
  tmp_path = f'/home/snmg/rttmas-recognition/src/tmp/{i}_nocrop.png'

  augment(src_path, i, verbose=0)
  result_augmented = _do_paddleocr(tmp_path)

  if result_augmented:
    if len(result_augmented) < 6:
      result_original = _do_paddleocr(src_path)
      if result_original:
        return result_original
    return result_augmented

  return None

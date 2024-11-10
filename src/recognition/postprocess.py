import re

def proper_license_plate(value):
  result = value.strip().upper()

  result = result.replace('I', '1').replace('|', '1')

  result = re.sub(r'[^\w]', '', result)

  length = len(result)

  # Remove first digit, if first digit is number but the next two are letters
  # e.g. 3AE6071E
  if length >= 7 and result[0].isdigit() and result[1:3].isalpha():
    result = result[1:]

  # Remove first digit, if first digit is letter but the next three are digits
  # e.g. A331KQ
  if length == 7 and result[0].isalpha() and result[1:4].isdigit():
    result = result[1:]

  # Remove the last digit, if there is an extra digit at the end
  # e.g. CA-63351 (extra '1')
  if length >= 7 and result[-5:].isdigit():
    result = result[:-1]

  return result



import re

def identify_license_plate_patterns(value):
  # Define the regular expressions for the patterns along with their names
  patterns = [
    (r'^[A-Z]{3}\d{4}$', 'ABC-1234'),
    (r'^[A-Z]{2}\d{4}$', 'AB-1234'),
    (r'^\d{4}[A-Z]{2}$', '1234-AB'),
    (r'^[A-Z]{3}\d{3}$', 'ABC-123'),
    (r'^[A-Z]{2}\d{3}$', 'AB-123'),
    (r'^\d{3}[A-Z]{2}$', '123-AB'),
    (r'^[A-Z]\d{5}$', 'A1-1234'),
    (r'^\d{1}[A-Z]\d{4}$', '1A-1234'),
    (r'^\d{4}[A-Z]\d{1}$', '1234-A1'),
    (r'^\d{5}[A-Z]$', '1234-1A')
  ]

  # Remove characters from the start and end and check each variation
  for pattern, pattern_name in patterns:
    for i in range(len(value) + 1):
      variation = value[i:]
      if re.match(pattern, variation):
        return pattern, pattern_name

  return None, None



def format_license_plate(value, pattern):
  if pattern in ['ABC-1234', 'ABC-123', '123-AB']:
    return value[:3] + '-' + value[3:]

  if pattern in ['AB-1234', 'AB-123', 'A1-1234', '1A-1234']:
    return value[:2] + '-' + value[2:]

  if pattern in ['1234-AB', '1234-A1', '1234-1A']:
    return value[:4] + '-' + value[4:]

  return None



def pick_proper_license_plate(val1, val2, pattern1, pattern2):

  score1, score2 = 0, 0

  if not val1:
    return val2
  if not val2:
    return val1

  if len(val1) < 5 or len(val1) > 7:
    score1 -= 1
  elif len(val2) < 5 or len(val2) > 7:
    score2 -= 1

  if not pattern1:
    score1 -= 1

  if not pattern2:
    score2 -= 1

  if score1 > score2:
    return val1
  else:
    return val2




# def postprocess(i, easyocr_reader, verbose=1):

#   pattern_name_nocrop, pattern_name_cropped = None, None
#   converted_value_nocrop, converted_value_cropped = None, None

#   result_nocrop = reader.readtext(f'/tmp/{i}_nocrop.png', detail=0)
#   if len(result_nocrop) > 0:
#     s = proper_license_plate(result_nocrop[0])
#     pattern_regex, pattern_name_nocrop = identify_license_plate_patterns(s)
#     converted_value_nocrop = format_license_plate(s, pattern_name_nocrop)

#   result_cropped = reader.readtext(f'/tmp/{i}.png', detail=0)
#   if len(result_cropped) > 0:
#     s = proper_license_plate(result_cropped[0])
#     pattern_regex, pattern_name_cropped = identify_license_plate_patterns(s)
#     converted_value_cropped = format_license_plate(s, pattern_name_cropped)

#   final = pick_proper_license_plate(converted_value_nocrop, converted_value_cropped, pattern_name_nocrop, pattern_name_cropped)

#   if verbose == 1:
#     print(f'1: {result_nocrop}    {converted_value_nocrop}    {pattern_name_nocrop}')
#     print(f'2: {result_cropped}    {converted_value_cropped}    {pattern_name_cropped}')
#     print(final)

#   return final


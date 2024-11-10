# import context  # Ensures paho is in PYTHONPATH

import paho.mqtt.client as mqtt
import ssl
import json
import cv2

import numpy as np
from PIL import Image
from io import BytesIO
import base64

from detection import detect_image
from report import make_grpc_report


def on_connect(mqttc, obj, flags, reason_code, properties):
    print("reason_code: " + str(reason_code))


def on_message(mqttc, obj, msg):
    topic = msg.topic
    payload = msg.payload

    if not topic.startswith('uplink/report/'):
        return

    print("[01] Raw image received via MQTT.")


    reporter_uid = topic.split('/')[2]
    payload = json.loads(payload)

    b64_encoded_image = payload['raw_image']
    # img = Image.open(BytesIO(base64.b64decode(b64_encoded_image)))

    print("[02] Begin object detection and plate recognition...")

    nparr = np.frombuffer(base64.b64decode(b64_encoded_image), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    plate_numbers = detect_image(img)
    
    print("[03] Detection complete.")

    if len(plate_numbers) > 0:
        del payload['report_type']
        del payload['raw_image']
        payload['plate_numbers'] = plate_numbers
        mqttc.publish("recognition_result", json.dumps(payload))

        report_time = payload['report_time']
        lat = payload['lat']
        lon = payload['lon']
        speed = payload['gps_speed_kmh']
        heading = payload['gps_heading']

        make_grpc_report(report_time, reporter_uid, lat, lon, speed, heading, plate_numbers)
        print("[04] Recognition results emitted via MQTT and gRPC.")
        print("\n\n\n")


def on_subscribe(mqttc, obj, mid, reason_code_list, properties):
    print("Subscribed: " + str(mid) + " " + str(reason_code_list))


def on_log(mqttc, obj, level, string):
    print(string)



def run_mqtt():
    # detect_test()
    # return

    mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, protocol=mqtt.MQTTv5, client_id='recognition', transport='tcp')
    mqttc.username_pw_set(username='test', password='test')
    mqttc.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)

    mqttc.on_message = on_message
    mqttc.on_connect = on_connect
    mqttc.on_subscribe = on_subscribe
    mqttc.on_log = on_log

    mqttc.connect(host="rttmas.mwnl.ce.ncu.edu.tw", port=8883, keepalive=60)
    mqttc.subscribe("#")

    mqttc.loop_forever()
    

# def detect_test():
#     img = cv2.imread("input.png")
#     plate_numbers = detect_image(img)
#     print(plate_numbers)

#     make_grpc_report(1730710951, "test-uid", 37.7749, 122.4194, 0, 0, plate_numbers)
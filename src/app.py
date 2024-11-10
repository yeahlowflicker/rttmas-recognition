from flask import Flask, render_template
import os
from recognition.detect import *
from recognition import mqtt
import gc


app = Flask(__name__)

# IMAGE_FOLDER = os.path.join('media', 'images')

@app.route('/')
def show():
    full_filename = 'static/images/output.png'
    return render_template("app.html", user_image = full_filename)

@app.route('/imshow')
def show_index():
    full_filename = 'static/images/output.png'
    return render_template("imshow.html", user_image = full_filename)


from flask_socketio import SocketIO
from flask_socketio import send, emit

MAX_BUFFER_SIZE = 50 * 1000 * 1000  # 50 MB
socketio = SocketIO(app, max_http_buffer_size=MAX_BUFFER_SIZE)

import base64

import redis
redis_server = redis.Redis(host='localhost', port=6379, db=0)

@socketio.on('upload')
def handle_client_report(data):
    # emit('user-report-ack', {  })
    img = data['img']
    seq_number = data['seq_number']
    imgdata = base64.b64decode(str.encode(img))
    filename = 'static/uploads/input.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    img = cv2.imread(filename)
    final_image, detection_results = full_detect(img, redis_server)
    cv2.imwrite('static/images/output.png', final_image)
    enc = cv2.imencode('.png', final_image)
    # emit("recognition-complete", base64.b64encode(enc[1]).decode('utf-8'))
    emit('recognition-complete', {
        'seq_number': seq_number,
        'results': detection_results,
    })
    gc.collect()

if __name__ == '__main__':
    socketio.run(app, host="localhost", debug=True)

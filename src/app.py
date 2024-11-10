from flask import Flask, render_template
import mqtt
# import os
# from recognition.detect import *
# from recognition import mqtt
# import gc


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

    
if __name__ == "__main__":
    mqtt.run_mqtt()


# if __name__ == '__main__':
#     socketio.run(app, host="localhost", debug=True)

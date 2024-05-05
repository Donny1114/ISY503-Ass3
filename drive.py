import os
import glob
import argparse
import base64
from io import BytesIO

import socketio
from flask import Flask
import eventlet.wsgi
import numpy as np
from keras.models import load_model
from PIL import Image

from model import crop, cropped_width, cropped_height, origin_colours, save_autonomous_image, grayscale, equalize

grayscale_model: bool = False

# Path of the trained model file r".\model.h5"

# Initializing server and wrapping Flask with socketio.
# :param model: Model that will be passed used to predict steering angle

server = socketio.Server()
app = Flask(import_name="ISY503 Assessment 3")

cli_opts = argparse.ArgumentParser()
cli_opts.add_argument('--debug', default=True, action='store_true', help='Debug mode')
cli_opts.add_argument('--file', type=str, default='', help='Saved model path')
cli_opts.add_argument('--save-image-to', type=str, default='', help=' Saved autonomous data directory')
cli_opts.add_argument('--speed-limit', type=float, default=12.0, help='Possible maximum speed')
options = cli_opts.parse_args()


# Instead of using hard-coded values for throttle calculations, these values can be calculated dynamically
# based on the speed limit.
def select_throttle(speed: float, speed_limit: float) -> float:
    delta = 0.25
    throttle = 0.0

    # Calculate the throttle based on the difference between the current speed and the speed limit
    throttle = ((1.0 - speed / speed_limit) * 1.5)

    # Clamp the throttle value between 0 and 1
    throttle = max(0, min(1, throttle))

    # Ensure the throttle is not too low
    if abs(throttle) < 0.09:
        throttle = 0.09 if throttle > 0 else -0.09

    return round(throttle, 3)

def connection_event_handler(sid: str, _: dict):
    print('Connected ' + sid)


def telemetry_event_handler(sig: str, msg: dict):
    global options

    mode: str = "manual"
    control: dict[str, str] = {}

    if msg:
        mode = "steer"

        # JSON Read input data
        img_center = msg["image"]
        speed = float(msg["speed"])
        # Conversion from base64-encoded image to PIL.Image object
        img = Image.open(BytesIO(base64.b64decode(img_center)))
        origin = img

        # Preprocessing
        img = crop(img)
        img = equalize(img)

        # If the model was trained on grayscale images, convert the image to a grayscale palette.
        if grayscale_model:
            img = grayscale(img)

        if options.debug is True and np.random.rand() < 0.1:
            img.save("debug_driving_autonomous.jpg")

        # The layer and input 0 of the "sequential" layer are incompatible: anticipated shape = (None, 80, 320, 3)
        # found shape = (None, 320, 3)
        image_array = np.asarray(img).reshape([1, cropped_height(), cropped_width(), origin_colours])

        # Obtaining the trained model's steering angles
        predicted = model.predict(image_array, verbose=0)
        steering = round(float(predicted[0][0]), 3)

        control = {
            "steering_angle": steering.__str__(),
            "throttle": select_throttle(speed, options.speed_limit).__str__(),
        }

        if options.debug is True:
            print(control)

        # Saving autonomous photos, so you can train on them again.
        if options.save_image_to != "":
            save_autonomous_image(options.save_image_to, origin, steering)

    # Return the response to the simulator.
    server.emit(mode, data=control, skip_sid=True)


# Associate Socket.IO's frames with their handlers
server.on('connect', connection_event_handler)
server.on('telemetry', telemetry_event_handler)

if options.file == "" and os.path.exists("model.h5"):
    options.file = "model.h5"

# Choose the last saved model on disk
if options.file == "":
    models = glob.glob("./model-2023-*")
    models.sort(reverse=False)
    options.file = os.path.basename(models[-1])

print("Model %s is chosen to be used as a target" % options.file)


# Saving model loading. If safe_mode=False is not set, error will occur.
model = load_model(options.file, safe_mode=False)

model_config = model.get_config()
input_shape = model_config['layers'][0]['config']['batch_input_shape']

# It will also be necessary to transform the input photos to grayscale if we're using a grayscale model.
if input_shape[3] == 1:
    origin_colours = input_shape[3]
    grayscale_model = True
    print("This model is trained on grayscale images. Colour setting have been adjusted...")


# Starting the web server on port 4567.
app = socketio.Middleware(server, app)
eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

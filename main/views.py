import atexit
import sys
from datetime import datetime

import psutil as psutil
from django.http import HttpResponse
from django.shortcuts import render

import argparse
import os
import random
import shutil
import time
import signal
import time
import pickle

import requests
import cv2
from .Arducam import *
from .ImageConvert import *

exit_ = False

## START
verbose_imshow = False
clear = True
url = "http://34.93.90.203:8501/fetch_accuracy/"

static_region_of_interest = {

    'target': [(5737, 2527), (6979, 5475), 17.361, 3.5615],  # target

}

# Length
calibration_constant_length = 0.008367788461538461

# Thread dia
calibration_constant_thread = 0.008495226730310262


def show_image(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)


def inference_paths(target_path, calibration_path, target_length, calibration_length, thread_length, random_id):
    print("Doing Inference!")
    payload = {
        'calibration_length': calibration_length,
        'target_length': target_length,
        'thread_length': thread_length,
        'file_type': 'png',
        'pair_status': '123',
        'calib_const': '123'
    }
    files = [
        ('target_file', ('file', open(target_path, 'rb'), 'image/png')),
        ('calibration_file', ('file', open(calibration_path, 'rb'), 'image/png'))
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.text)


def inference_objects(target_obj, calibration_obj, target_length, calibration_length, thread_length, random_id):
    print("Doing Inference objects!")

    payload = {'calibration_length': calibration_length,
               'target_length': target_length,
               'thread_length': thread_length,
               'file_type': 'png',
               'pair_status': 'False',
               'calib_const': str(calibration_constant_length),
               'calib_const_thread': str(calibration_constant_thread)}

    t5 = time.time()

    # target_bytes = BytesIO()
    # target_obj.save(target_bytes, format='png')

    # calibration_bytes = BytesIO()
    # calibration_obj.save(calibration_bytes, format='png')

    # target_obj.save(os.path.join(random_id, 'target_cropped.png'))
    # save target_obj with cv2
    cv2.imwrite(os.path.join(random_id, 'target_cropped.png'), target_obj)
    print('Time taken for encoding bytes :', time.time() - t5)

    files = [
        ('target_file', ('target_cropped.png', open(os.path.join(random_id, 'target_cropped.png'), 'rb'), 'image/png')),
        ('live_ui_image',
         ('ui_image.png', open(os.path.join(random_id, 'live_ui_image_resized.jpeg'), 'rb'), 'image/jpeg'))
        # ('target_file', ('file', target_bytes.getvalue(), 'image/png')),
        # ('calibration_file', ('file', calibration_bytes.getvalue(), 'image/png'))
    ]

    headers = {}

    t6 = time.time()
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    print(response.text)
    print('Time taken for server :', time.time() - t6)
    data = eval(eval(response.text)["Data"])
    print("Length Error :", data["Screw"]["Length_Error"])
    print("Thread Error :", data["Screw"]["Thread_Error"])
    return True


def white_balance_loops(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 2.5)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 2.5)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def convert(img):
    t6 = time.time()
    image = Image.fromarray(img)
    # image = image.convert("RGB")
    image = image.rotate(180)
    # image.save('temp.png')
    print('Time taken for pillow :', time.time() - t6)

    return image


def resize_image_jpeg(converted_image, random_id):
    t7 = time.time()

    # compress image using cv2

    converted_image = white_balance_loops(cv2.resize(converted_image, None, fx=0.5, fy=0.5))
    cv2.imwrite(os.path.join(random_id, 'live_ui_image_resized.jpeg'), converted_image, [cv2.IMWRITE_JPEG_QUALITY, 20])
    print('Time taken for resizing :', time.time() - t7)


## END
def sigint_handler(signum, frame):
    global exit_
    exit_ = True


signal.signal(signal.SIGINT, sigint_handler)
signal.signal(signal.SIGTERM, sigint_handler)


def display_fps(index):
    display_fps.frame_count += 1

    current = time.time()
    if current - display_fps.start >= 1:
        print("fps: {}".format(display_fps.frame_count))
        display_fps.frame_count = 0
        display_fps.start = current


def start_camera(request):
    display_fps.start = time.time()
    display_fps.frame_count = 0
    play = False
    global camera
    camera = ArducamCamera()
    try:

        config_file = os.path.join(os.path.dirname(__file__), 'ardu.cfg')

        if not camera.openCamera(config_file):
            raise RuntimeError("Failed to open camera.")

        camera.start()
    except Exception as e:
        camera.stop()
        camera.closeCamera()

    return HttpResponse("Camera started you can start taking images")


def capture_image(request):
    try:

        time_start = time.time()
        scale_width = 1280
        random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
        print('Processing!', random_id)
        os.mkdir(random_id)
        time_camera_started = time.time()
        print('time for camera opening and preview:', time_camera_started - time_start)
        play = False
        cwd = os.path.dirname(os.path.realpath(__file__))
        exit_ = False
        while not exit_:
            ret, data, cfg = camera.read()

            display_fps(0)

            if ret:
                image = convert_image(data, cfg, camera.color_mode)

                time_after_image = time.time()

                print('image capturing, converting(bytes to image)', time_after_image - time_camera_started)
                if play:
                    if scale_width != -1:
                        scale = scale_width / image.shape[1]
                        image = cv2.resize(image, None, fx=scale, fy=scale)
                    cv2.imshow("Arducam", image)
                else:
                    # new code START Firday, March 3
                    # image = white_balance_loops(image)

                    time_white = time.time()
                    print('white balance', time_white - time_after_image)

                    converted_image = cv2.rotate(image, cv2.ROTATE_180)

                    thread1 = threading.Thread(target=resize_image_jpeg, args=(converted_image, random_id,))
                    thread1.start()
                    time_rotation_conv = time.time()
                    print('time for image rotation and jpeg conversion', time_rotation_conv - time_white)
                    target_image_object = None
                    calibration_image_object = None

                    for object in static_region_of_interest:
                        # points = (x1, y1, x2, y2)
                        t4 = time.time()
                        # save converted_image
                        # 111cv2.imwrite('converted_image.png', converted_image)
                        # crop the converted_image using cv2
                        cropped_image = converted_image[
                                        static_region_of_interest[object][0][1]:static_region_of_interest[object][1][1],
                                        static_region_of_interest[object][0][0]:static_region_of_interest[object][1][0]]
                        # crop the converted_image using cv2

                        print('cropping in loop ', time.time() - t4)

                        # cropped_image = converted_image.crop((static_region_of_interest[object][0][0],
                        #                                       static_region_of_interest[object][0][1],
                        #                                       static_region_of_interest[object][1][0],
                        #                                       static_region_of_interest[object][1][1]))

                        if object == 'target':
                            target_image_object = cropped_image
                        else:
                            calibration_image_object = cropped_image
                    time_cropping = time.time()
                    print('time for cropping image', time_cropping - time_rotation_conv)
                    thread1.join()  # Waiting for the thread to save the raw image that will be sent to the UI
                    try:

                        '''inference_objects(target_image_object,
                                          calibration_image_object,
                                          static_region_of_interest['target'][2],
                                          '123',
                                          static_region_of_interest['target'][3],
                                          random_id)
'''
                    except Exception as a:
                        return HttpResponse('Failed to inference Object with exception : {}'.format(str(a)))
                    # if clear:
                    #     shutil.rmtree(random_id)
                    # 111cv2.imwrite('New_image.png', cropped_image)
                    # cropped_image=cropped_image.save('New_image.png')

                    # 111cv2.imwrite('random1st.png', target_image_object)
                    # new code END Firday, March 3
                    # 111cv2.imwrite('random.png', image)
                    exit_ = True
                    time_after_write = time.time()
                    print('time for inferencer ', time_after_write - time_cropping)
            else:
                print("timeout")
        time_end = time.time()

        print('', time_end)
        print('elapsed', time_end - time_start)
        finalstring = f"Image Captured at {random_id}\n <br>" \
                      f"time for camera opening and preview:, {time_camera_started - time_start}\n <br>" \
                      f"'image capturing, converting(bytes to image)', {time_after_image - time_camera_started}\n <br>" \
                      f"'white balance', {time_white - time_after_image}\n <br>" \
                      f"'time for image rotation and jpeg conversion', {time_rotation_conv - time_white}\n <br>" \
                      f"'time for cropping image', {time_cropping - time_rotation_conv}\n <br>" \
                      f"'time for inference Object ', {time_after_write - time_cropping}\n <br>" \
                      f"'elapsed', {time_end - time_start} <br>"
        return HttpResponse(finalstring)
    except Exception as e:
        return HttpResponse(f"Failed to capture image : {str(e)}")


# Create your views here.


def stop_django_server(request, port=8000):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'runserver' in proc.info['cmdline'] and f':{port}' in proc.info['cmdline']:
                os.kill(proc.pid, 9)
                print(f"Django server running on port {port} has been stopped.")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    else:
        print(f"No Django server found running on port {port}.")

    camera.stop()
    camera.closeCamera()
    return HttpResponse("Camera Closed now you can close the terminal")

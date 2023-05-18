import atexit
import sys
from datetime import datetime
from django.http import JsonResponse
import psutil as psutil
from django.http import HttpResponse
from django.shortcuts import render
import threading
import argparse
import os
import random
import traceback
import json
import shutil
import time
import signal
import time
import pickle
from . import config
import requests
import cv2
from .Arducam import *
from .ImageConvert import *
from threading import Thread, Lock
_db_lock = Lock()

exit_ = False
import base64
## START
verbose_imshow = False
clear = True
url ="http://34.93.168.221:8501/calibration/"
static_region_of_interest = {

    'target': [(5952, 3312), (6912, 5712), 17.361, 3.5615],  # target

}

# Length
calibration_constant_length = 0.008367788461538461

# Thread dia
calibration_constant_thread = 0.008495226730310262



class ThreadWithReturnValue(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

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


import json
import base64
def inference_objects(target_obj, calibration_obj, target_length, calibration_length, thread_length, random_id,idx):
    print("Doing Inference objects!")

    payload = config.calibration_payloads.get(str(idx))

    t5 = time.time()

    cv2.imwrite(os.path.join(random_id, f'target_cropped{str(idx)}.png'), target_obj)
    print('Time taken for encoding bytes :', time.time() - t5)

    # contig=np.ascontiguousarray(target_obj)
    # target_obj.tofile(os.path.join(random_id, f'target_cropped{str(idx)}.npy'))
    # np.save(os.path.join(random_id, f'target_cropped{str(idx)}.npy'), target_obj)

    #
    # image_string=base64.b64encode(contig).decode('utf-8')


    files = [
        ('target_file', ('target_cropped.png', open(os.path.join(random_id, f'target_cropped{str(idx)}.png'), 'rb'))),
        ('live_ui_image',
         ('ui_image.png', open(os.path.join(random_id, 'live_ui_image_resized.jpeg'), 'rb'), 'image/jpeg'))
    ]
    # payload['target_file']=image_string
    headers = {}
    print(payload)
    t6 = time.time()
    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print('total time elapsed',response.elapsed.total_seconds())
    print(response.text)
    print('Time taken for server :', time.time() - t6)
    data = eval(eval(response.text)["Data"])
    print("Length Error :", data["Screw"]["Length_Error"])
    print("Thread Error :", data["Screw"]["Thread_Error"])
    return response


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


def stop_django_server(request, port=8000):
    try:
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
    except:
        try:
            camera.stop()
            camera.closeCamera()
            return HttpResponse("Camera Closed now you can close the terminal")
        except:
            return HttpResponse("Camera Already Closed")


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


def final_result(images,random_id):
    idx = 0
    threads = []

    result = {}
    time_main = time.time()
    for i in images:
        idx += 1
        one_image = time.time()
        print(f'Starting process for ')
        target_image_object = i
        calibration_image_object = i
        if object == 'target':
            target_image_object = i
        else:
            calibration_image_object = i
        # Waiting for the thread to save the raw image that will be sent to the UI
        try:
            # response = inference_objects(target_image_object,
            #                   calibration_image_object,
            #                   static_region_of_interest['target'][2],
            #                   '123',
            #                   static_region_of_interest['target'][3],
            #                   random_id, idx=4)
            # result.append(response)
            threads.append(ThreadWithReturnValue(target=inference_objects, args=(target_image_object,
                                                                                 calibration_image_object,
                                                                                 static_region_of_interest['target'][2],
                                                                                 '123',
                                                                                 static_region_of_interest['target'][3],
                                                                                 random_id, idx,)))

        except Exception as a:
            traceback.print_exc()
            return HttpResponse('Failed to inference Object with exception : {}'.format(str(traceback.format_exc(a))))

        print("time for each image inference ", time.time() - one_image)



    for t in threads:
        t.start()
    screw_pos = 1
    for t in threads:
        thread_time = time.time()

        result[f'screw{screw_pos}'] = (t.join()).content.decode("utf-8")
        screw_pos += 1

        print(f'joining threads', time.time() - thread_time)
    return result

def capture_image(request):
    try:

        time_start = time.time()
        scale_width = 1280
        random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
        print('Processing!', random_id)
        os.mkdir(random_id)
        result = {}
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
                    converted_image = cv2.flip(converted_image, 1)
                    center_image = converted_image[1984:4480, 5154:5936]
                    target_image_object = center_image
                    time_cropping = time.time()
                    images=[target_image_object]
                    print('time for cropping image', time_cropping - time_rotation_conv)
                    thread1.join()
                    idx=3
                    threads = []

                    time_main = time.time()
                    for i in images:
                        idx += 1
                        one_image = time.time()
                        print(f'Starting process for ')
                        target_image_object = i
                        calibration_image_object = i
                        if object == 'target':
                            target_image_object = i
                        else:
                            calibration_image_object = i
                        # Waiting for the thread to save the raw image that will be sent to the UI
                        try:

                            threads.append(ThreadWithReturnValue(target=inference_objects, args=(target_image_object,
                                                                                                 calibration_image_object,
                                                                                                 static_region_of_interest[
                                                                                                     'target'][2],
                                                                                                 '123',
                                                                                                 static_region_of_interest[
                                                                                                     'target'][3],
                                                                                                 random_id, idx,)))

                        except Exception as a:
                            traceback.print_exc()
                            return HttpResponse(
                                'Failed to inference Object with exception : {}'.format(str(traceback.format_exc(a))))

                        print("time for each image inference ", time.time() - one_image)

                    for t in threads:
                        t.start()
                    screw_pos = 1
                    for t in threads:
                        thread_time = time.time()

                        result[f'screw{screw_pos}'] = (t.join()).content.decode("utf-8")
                        screw_pos += 1
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
                      f"'elapsed', {time_end - time_start} <br>"
        print(result)
        abs = os.path.abspath(random_id)
        response_dict=result
        response_dict['ui_image'] = os.path.join(abs, 'live_ui_image_resized.jpeg')
        response_dict['random_id'] = random_id
        print(response_dict)
        return JsonResponse(response_dict)
    except Exception as e:
        camera.stop()
        camera.closeCamera()
        return HttpResponse(f"Failed to capture image : {str(e)}")


# Create your views here.

def inference_3_screw(request):
    try:

        time_start = time.time()
        scale_width = 1280
        random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
        print('Processing!', random_id)
        os.mkdir(random_id)
        result = {}
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
                    converted_image = cv2.flip(converted_image, 1)
                    # cv2.imwrite('New_image.png', converted_image)
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
                        #croppping[y1:y2,x1:x2]
                        left_image = converted_image[1984:4480, 4370:5154]
                        center_image = converted_image[1984:4480, 5154:5936]
                        right_image = converted_image[1984:4480, 5936:6720]


                        images=[left_image,center_image,right_image]
                        print('cropping in loop ', time.time() - t4)

                    time_cropping = time.time()
                    print('time for cropping image', time_cropping - time_rotation_conv)
                    thread1.join()

                    threads = []
                    idx=3
                    time_main = time.time()
                    for i in images:
                        idx += 1
                        one_image = time.time()
                        print(f'Starting process for ')
                        target_image_object = i

                        calibration_image_object = i
                        if object == 'target':
                            target_image_object = i
                        else:
                            calibration_image_object = i
                        # Waiting for the thread to save the raw image that will be sent to the UI
                        try:
                            # response = inference_objects(target_image_object,
                            #                   calibration_image_object,
                            #                   static_region_of_interest['target'][2],
                            #                   '123',
                            #                   static_region_of_interest['target'][3],
                            #                   random_id, idx=4)
                            # result.append(response)
                            threads.append(ThreadWithReturnValue(target=inference_objects, args=(target_image_object,
                                                                                                 calibration_image_object,
                                                                                                 static_region_of_interest[
                                                                                                     'target'][2],
                                                                                                 '123',
                                                                                                 static_region_of_interest[
                                                                                                     'target'][3],
                                                                                                 random_id, idx,)))

                        except Exception as a:
                            traceback.print_exc()
                            return HttpResponse(
                                'Failed to inference Object with exception : {}'.format(str(traceback.format_exc(a))))

                        print("time for each image inference ", time.time() - one_image)

                    for t in threads:
                        t.start()
                    screw_pos = 1
                    for t in threads:
                        thread_time = time.time()

                        result[f'screw{screw_pos}'] = (t.join()).content.decode("utf-8")
                        screw_pos += 1

                        print(f'joining threads', time.time() - thread_time)

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
                      f"'elapsed', {time_end - time_start} <br>"
        abs = os.path.abspath(random_id)
        response_dict = result
        response_dict['ui_image'] = os.path.join(abs, 'live_ui_image_resized.jpeg')
        response_dict['random_id'] = random_id
        print(response_dict)
        return JsonResponse(response_dict)
    except Exception as e:
        return HttpResponse(f"Failed to capture image : {str(e)}")


def temp_crop(request):
    t1=time.time()
    random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    print('Processing!', random_id)
    os.mkdir(random_id)
    converted_image = cv2.imread('/home/techolution/Desktop/Techolution/django_globus/client_server/final_crop_calibration.png')
    time_convert_image = time.time()
    print("time for creating directory", time_convert_image - t1)

    left_image = converted_image[3392:5664, 4448:5264]
    center_image = converted_image[3392:5664, 5264:6048]
    right_image = converted_image[3392:5664, 6048:6832]

    images = [left_image, center_image, right_image]
    # images = [center_image]
    time_after_crop=time.time()
    print('cropping 1 image',time_after_crop - time_convert_image)
    thread1 = threading.Thread(target=resize_image_jpeg, args=(converted_image, random_id,))
    thread1.start()
    thread1.join()
    result=final_result(images,random_id)

    print("TOTAL TIME TAKEN " ,time.time() - t1)
    # response=JsonResponse({'headers':'','result':result})
    # response.status_code=200
    response_dict=result


    abs=os.path.abspath(random_id)
    response_dict['ui_image']=os.path.join(abs, 'live_ui_image_resized.jpeg')
    response_dict['random_id']=random_id
    print(response_dict)
    return JsonResponse(response_dict)


def manual_inference(request,**kwargs):
    print(kwargs)
    print(kwargs.get('from'))
    print(kwargs.get('to'))
    from_stud=int(kwargs.get('from'))
    to_stud=int(kwargs.get('to'))

    number_of_screws=int(kwargs.get('from'))-int(kwargs.get('to')) + 1
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

                    time_white = time.time()
                    print('white balance', time_white - time_after_image)

                    converted_image = cv2.rotate(image, cv2.ROTATE_180)
                    thread1 = threading.Thread(target=resize_image_jpeg, args=(converted_image, random_id,))
                    thread1.start()
                    time_rotation_conv = time.time()
                    print('time for image rotation and jpeg conversion', time_rotation_conv - time_white)
                    target_image_object = None
                    calibration_image_object = None
                    converted_image = cv2.flip(converted_image, 1)
                    x4_start=5264
                    x4_end=6048
                    y4_top=3392
                    y4_bottom=5664
                    threads = []
                    thread1.join()
                    images=[]
                    thread_generation_time=time.time()
                    for stud_pos in range(from_stud,to_stud+1):
                        if stud_pos < 4:
                            start_x_coordinate= (x4_start - ((4-stud_pos)*784))
                            end_x_coordinate= (x4_end - ((4-stud_pos)*784))
                        elif stud_pos >4:
                            start_x_coordinate = (x4_start + (abs(4 - stud_pos) * 784))
                            end_x_coordinate = (x4_end + (abs(4 - stud_pos) * 784))
                        else:
                            start_x_coordinate=x4_start
                            end_x_coordinate=x4_end
                        roi=converted_image[y4_top:y4_bottom, start_x_coordinate:end_x_coordinate]

                        try:
                            idx=stud_pos
                            target_image_object=roi
                            calibration_image_object=roi
                            threads.append(threading.Thread(target=inference_objects, args=(target_image_object,
                                                                                            calibration_image_object,
                                                                                            static_region_of_interest[
                                                                                                'target'][2],
                                                                                            '123',
                                                                                            static_region_of_interest[
                                                                                                'target'][3],
                                                                                            random_id, idx)))

                        except Exception as a:
                            return HttpResponse('Failed to inference Object with exception : {}'.format(str(a)))


                    print('Initialize all image threads', time.time() - thread_generation_time)

                    time_threading_start = time.time()


                    for t in threads:
                        t.start()
                    print('All threads Started', time.time() - time_threading_start)
                    time_thread_joining=time.time()
                    for t in threads:
                        thread_time = time.time()
                        t.join()
                        print(f'joining threads', time.time() - thread_time)
                    print('all thread joined',time.time() - time_thread_joining)
                    exit_ = True
                    time_after_write = time.time()
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
                      f"'time for cropping image', {time.time() - time_rotation_conv}\n <br>" \
                      f"'elapsed', {time_end - time_start} <br>"
        return HttpResponse(finalstring)
    except Exception as e:
        return HttpResponse(f"Failed to capture image : {str(e)}")


def request_visualizer(request):
    print(request)
    print(request.body)
    return HttpResponse(request.body)

from django.core.files import File
from django.core.files.storage import FileSystemStorage
def calib(request):
    t1 = time.time()
    random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    print('Processing!', random_id)
    os.mkdir(random_id)
    converted_image = cv2.imread(
        '/home/techolution/Desktop/Techolution/django_globus/client_server/final_crop_calibration.png')
    time_convert_image = time.time()
    print("time for creating directory", time_convert_image - t1)

    left_image = converted_image[3392:5664, 4448:5264]
    center_image = converted_image[3392:5664, 5264:6048]
    right_image = converted_image[3392:5664, 6048:6832]

    #images = [left_image, center_image, right_image]
    images = [center_image]
    time_after_crop = time.time()
    print('cropping 1 image', time_after_crop - time_convert_image)
    thread1 = threading.Thread(target=resize_image_jpeg, args=(converted_image, random_id,))
    thread1.start()
    thread1.join()
    result = final_result(images, random_id)


    print("TOTAL TIME TAKEN ", time.time() - t1)
    # response=JsonResponse({'headers':'','result':result})
    # response.status_code=200
    response_dict = result

    abs = os.path.abspath(random_id)

    print("**************************************************************")
    json_data = result.get('screw1')
    jsonResponse = json.loads(json_data)
    print(jsonResponse)

    calib_const = eval(jsonResponse.get('Data')).get('Screw').get('mm/px')
    thread_calib_const = eval(jsonResponse.get('Data')).get('Screw').get('thread_calib_const')
    neck_calib_const = eval(jsonResponse.get('Data')).get('Screw').get('neck_calib_const')
    head_calib_const = eval(jsonResponse.get('Data')).get('Screw').get('head_calib_const')

    print(neck_calib_const)

    string = f'{{"calib_const" : {calib_const}, \n "thread_calib_const" : {thread_calib_const}, \n "head_calib_const":{head_calib_const}, \n "neck_calib_const": {neck_calib_const}   }}'
    final = "calib_values = " + string
    print(final)
    storage = FileSystemStorage()

    # Open the file in write mode
    with storage.open('calib.py', 'w+') as f:
        # Write the final string to the file
        f.write(final)

    return HttpResponse(str(final))

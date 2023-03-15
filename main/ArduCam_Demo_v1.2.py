import argparse
import os
import random
import shutil
import time
import signal
import time
import pickle

import PIL
from PIL import Image
import requests
from datetime import datetime
import cv2
from Arducam import *
from ImageConvert import *

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

    #target_obj.save(os.path.join(random_id, 'target_cropped.png'))
    #save target_obj with cv2
    cv2.imwrite(os.path.join(random_id, 'target_cropped.png'), target_obj)
    print('Time taken for encoding bytes :', time.time() - t5)

    files = [
        ('target_file', ('target_cropped.png', open(os.path.join(random_id, 'target_cropped.png'), 'rb'), 'image/png')),
        ('live_ui_image', ('ui_image.png', open(os.path.join(random_id, 'live_ui_image_resized.jpeg'), 'rb'), 'image/jpeg'))
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

    # Using the Numpy array
    # t6 = time.time()
    # (h, w) = image_array.shape[:2]
    # (cX, cY) = (w // 2, h // 2)
    # rotation_matrix = cv2.getRotationMatrix2D((cX, cY), 180, 1.0)
    # image = cv2.warpAffine(image_array, rotation_matrix, (w, h))
    # print('Time taken for numpy :', time.time() - t6)

    return image


def resize_image_jpeg(converted_image, random_id):
    t7 = time.time()

    #compress image using cv2
    converted_image = cv2.resize(converted_image, None, fx=0.5, fy=0.5)
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


display_fps.start = time.time()
display_fps.frame_count = 0


if __name__ == "__main__":
    time_start = time.time()
    print('start',time_start)
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--config-file', type=str, required=True, help='Specifies the configuration file.')
    parser.add_argument('-v', '--verbose', action='store_true', required=False, help='Output device information.')
    parser.add_argument('--preview-width', type=int, required=False, default=-1, help='Set the display width')
    parser.add_argument('-n', '--nopreview', action='store_true', required=False, help='Disable preview windows.')
    parser.add_argument('-play','--play', action='store_true',default=False, required=False, help='Preview the video')



    args = parser.parse_args()
    config_file = args.config_file
    verbose = args.verbose
    preview_width = args.preview_width
    no_preview = args.nopreview
    play=args.play


    camera = ArducamCamera()
    if not camera.openCamera(config_file):
        raise RuntimeError("Failed to open camera.")

    if verbose:
        camera.dumpDeviceInfo()

    camera.start()
    # camera.setCtrl("setFramerate", 2)
    # camera.setCtrl("setExposureTime", 20000)
    #camera.setCtrl("setAnalogueGain", 800)
    scale_width = preview_width

    cwd=os.path.dirname(os.path.realpath(__file__))
    # if not os.path.isdir(os.path.join(cwd,'output')):
    #     os.makedirs(os.path.join(cwd,'output'))
    # os.chdir(os.path.join(cwd,'output'))

    random_id = str(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    print('Processing!',random_id)
    os.mkdir(random_id)
    time_camera_started=time.time()
    print('time for camera opening and preview:', time_camera_started - time_start)
    while not exit_:
        ret, data, cfg = camera.read()

        display_fps(0)
        if no_preview:
            continue

        if ret:
            image = convert_image(data, cfg, camera.color_mode)

            time_after_image = time.time()

            print('image capturing and converting into white balance',time_camera_started-time_after_image)
            if play:
                if scale_width != -1:
                    scale = scale_width / image.shape[1]
                    image = cv2.resize(image, None, fx=scale, fy=scale)
                cv2.imshow("Arducam", image)
            else:
                # new code START Firday, March 3
                image=white_balance_loops(image)
                camera.stop()
                camera.closeCamera()
                # converted_image = convert(image)

                converted_image=cv2.rotate(image, cv2.ROTATE_180)


                thread1 = threading.Thread(target=resize_image_jpeg, args=(converted_image, random_id,))
                thread1.start()
                time_rotation_conv=time.time()
                print('time for image rotation and jpeg conversiown',time_rotation_conv-time_after_image)
                target_image_object = None
                calibration_image_object = None

                for object in static_region_of_interest:
                    # points = (x1, y1, x2, y2)
                    t4 = time.time()
                    #save converted_image
                    #111cv2.imwrite('converted_image.png', converted_image)
                    # crop the converted_image using cv2
                    cropped_image = converted_image[static_region_of_interest[object][0][1]:static_region_of_interest[object][1][1],
                                    static_region_of_interest[object][0][0]:static_region_of_interest[object][1][0]]
                    #crop the converted_image using cv2

                    print('cropping in loop ',time.time()-t4)

                    # cropped_image = converted_image.crop((static_region_of_interest[object][0][0],
                    #                                       static_region_of_interest[object][0][1],
                    #                                       static_region_of_interest[object][1][0],
                    #                                       static_region_of_interest[object][1][1]))

                    if object == 'target':
                        target_image_object = cropped_image
                    else:
                        calibration_image_object = cropped_image
                time_cropping=time.time()
                print('time for cropping image',time_cropping-time_rotation_conv)
                thread1.join()  # Waiting for the thread to save the raw image that will be sent to the UI
                inference_objects(target_image_object,
                                  calibration_image_object,
                                  static_region_of_interest['target'][2],
                                  '123',
                                  static_region_of_interest['target'][3],
                                  random_id)

                # if clear:
                #     shutil.rmtree(random_id)
                #111cv2.imwrite('New_image.png', cropped_image)
                #cropped_image=cropped_image.save('New_image.png')

                #111cv2.imwrite('random1st.png', target_image_object)
                # new code END Firday, March 3
                #111cv2.imwrite('random.png', image)
                exit_ = True
                time_after_write=time.time()
                print('time for inferencer ',time_after_write-time_cropping)
        else:
            print("timeout")

        key = cv2.waitKey(1)
        if key == ord('q'):
            exit_ = True
        elif key == ord('s'):
            np.array(data, dtype=np.uint8).tofile("image.raw")



    time_end = time.time()

    print('', time_end)
    print('elapsed',time_end-time_start)

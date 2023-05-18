from flask import Flask, jsonify, request

app = Flask(__name__)

import logging
import threading
import paddlehub as hub
from skimage.transform import probabilistic_hough_line
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data
import matplotlib.pyplot as plt
from matplotlib import cm
import json, random, operator, math, requests, argparse, traceback, datetime, os, sys, cv2, shutil, time
from functools import reduce
import numpy as np
import datetime
from google.cloud import storage
import firebase_admin
from firebase_admin import db
from head_and_neck import *


def signedurl(name, bucket='globus_screw'):
    storage_client = storage.Client.from_service_account_json(
        "/home/jupyter/globus/21_Feb/faceopen-techolution-firebase-adminsdk-mq18c-0bde0112e8.json")
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(name)
    url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(hours=5), method="GET")
    return url


def upload_image(gcs_path, local_path, bucket='globus_screw'):
    storage_client = storage.Client.from_service_account_json(
        "/home/jupyter/globus/21_Feb/storage@faceopen-techolution.iam.gserviceaccount.com.json")
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return "Upload Successful"


databaseURL = "https://globus-screw.firebaseio.com/"
cred_obj = firebase_admin.credentials.Certificate("faceopen-techolution-firebase-adminsdk-mq18c-0bde0112e8.json")
default_app = firebase_admin.initialize_app(cred_obj, {
    'databaseURL': databaseURL
})

# Have logs that will log the required calibration values
formatter = logging.Formatter('%(message)s')

def setup_logger(log_file, level=logging.INFO):
    """Function to create a logger file for each user"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger = logging.getLogger('logs')
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

logger = setup_logger("log.log")


ref = db.reference("/")
# ref.set({'screw': {'liveImages': {'image': {'url': 'imageUrl'},
#    'part': 184.156,
#    'qaBatch': 1,
#    'qaFail': 1,
#    'qaPass': 3,
#    'qaResults': 'Pass'},
#   'reportImage': {'url': ''},
#   'reports': [{'currentReading': 17.3,
#     'parameter': 'Screw Length',
#     'threshold': {'min': 17.2, 'max': 17.5}},
#    {'currentReading': 3.57,
#     'parameter': 'Thread Diameter',
#     'threshold': {'min': 3.55, 'max': 3.60}},
#    {'currentReading': 3.84,
#     'parameter': 'Neck Diameter',
#     'threshold': {'min': 3.82, 'max': 3.88}},
#    {'currentReading': 4.480,
#     'parameter': 'Head Diameter',
#     'threshold': {'min': 4.475, 'max': 4.500}}],
#   'stationInfo': {'alertIndicators': 'All OK',
#    'alertStatus': 'Good (No Problems)',
#    'needsMaintenance': 'no maintenance required',
#    'stationNo': 1},
#                   'current_time':f"{datetime.datetime.now()}"}})

highContrastingColors = ['rgba(0,255,81,1)', 'rgba(255,219,0,1)', 'rgba(255,0,0,1)', 'rgba(0,4,255,1)',
                         'rgba(227,0,255,1)']


# ==  ==  Actual Measurements == ==

# calib_orig = 19.910
# screw_orig = 17.430
# calib_orig = 25.092
# calib_orig = 17.395
# screw_orig = 16.655
# thread_orig = 3.566
# screw_orig = 17.395
# calib_orig = 16.655
# thread_orig = 3.552
class Reques_Data:
    def __init__(self, calib_path, screw_path, calib_orig, screw_orig, thread_orig, pair_status, calib_const,
                 calib_const_thread, raw_image_path, trigger_time, One_Above_Thread, Two_Above_Thread, Orig_Thread,
                 thread_p, neck_orig, head_orig, calib_const_neck, calib_const_head):
        self.calib_path = calib_path
        self.screw_path = screw_path
        self.calib_orig = calib_orig
        self.screw_orig = screw_orig
        self.thread_orig = thread_orig
        self.neck_orig = neck_orig
        self.head_orig = head_orig
        self.pair_status = pair_status
        self.calib_const = calib_const
        self.calib_const_thread = calib_const_thread
        self.calib_const_neck = calib_const_neck
        self.calib_const_head = calib_const_head
        self.raw_image_path = raw_image_path
        self.trigger_time = trigger_time
        self.thread_p = thread_p


status = 'calib'

model = hub.Module(name='U2Net')


def json_creater(inputs, closed):
    data = []
    color = random.sample(highContrastingColors, 1)[0]
    for index, input in enumerate(inputs):
        # JSON Object for the metadata and vertices
        json_id = random.randint(1, 100)
        sub_json_data = {}
        sub_json_data['id'] = json_id
        sub_json_data['name'] = json_id
        sub_json_data['color'] = color
        sub_json_data['isClosed'] = closed
        sub_json_data['selectedOptions'] = [{"id": "0", "value": "root"},
                                            {"id": str(random.randint(10, 20)), "value": inputs[input]}]

        # Sorting the vertices clockwise or anti-clockwise
        points = eval(input)
        if len(points) > 0:
            center = tuple(
                map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), points), [len(points)] * 2))
            sorted_coords = sorted(points, key=lambda coord: (-135 - math.degrees(
                math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
        else:
            sorted_coords = []

        # JSON Object for the vertices
        vertices = []
        is_first = True
        for vertex in sorted_coords:
            vertex_json = {}
            if is_first:
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
                is_first = False
            else:
                json_id = random.randint(10, 20)
                vertex_json['id'] = json_id
                vertex_json['name'] = json_id
            vertex_json['x'] = vertex[0]
            vertex_json['y'] = vertex[1]
            vertices.append(vertex_json)
        sub_json_data['vertices'] = vertices
        data.append(sub_json_data)
    return json.dumps(data)


def send_to_autoai_annotation(status, csv, model, predicted_label, tag, confidence_score, prediction, model_type,
                              filename, url, imageAnnotations, label_tag):
    try:
        print('Sending to AutoAI - Filename : ', filename)

        print('Sending API Call #1 : Upload details')
        payload = {'status': status,
                   'csv': csv,
                   'model': model,
                   'label': str(predicted_label),
                   'tag': tag,
                   'confidence_score': confidence_score,
                   'prediction': prediction,
                   'imageAnnotations': str(imageAnnotations),
                   'model_type': model_type,
                   'appShouldNotUploadResourceFileToGCS': 'true',
                   'resourceFileName': f"{label_tag}.png",
                   'resourceContentType': 'image/png'
                   }
        print(payload['resourceFileName'])
        headers = {}
        response = requests.request('POST', url, headers=headers, data=payload, verify=False)
        print("First Result", tag)
        print("=" * 20)
        # print(response.json())
        resourceFileSignedUrlForUpload = response.json()['resourceFileSignedUrlForUpload']
        resource_id = response.json()['_id']

        print('Sending to AutoAI #2 : Upload file', tag)
        with open(filename, mode='rb') as file:
            fileContent = file.read()
        print("Removing File : ", filename)
        payload = fileContent
        headers = {
            'Content-Type': 'image/png'
        }
        response = requests.request("PUT", resourceFileSignedUrlForUpload, headers=headers, data=payload)

        print('Sending to AutoAI #3 : ', tag)
        response = requests.request('POST',
                                    'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/%s/resourceDirectUploadToGCSCompleted' % str(
                                        resource_id), headers={}, verify=False)
        os.remove(filename)
        return True
    except Exception as e:
        print(traceback.format_exc())
        print('Error while sending data to Auto AI : ', e)
        return False


# def send_to_autoai_annotation(status, csv, model, predicted_label, tag, confidence_score, prediction, model_type,
#                               filename, url, imageAnnotations, label_tag):

#     try:
#         if True:
#             print('Sending to AutoAI - Filename : ', filename)
#         payload = {'status': status,
#                    'csv': csv,
#                    'model': model,
#                    'label': str(predicted_label),
#                    'tag': tag,
#                    'confidence_score': confidence_score,
#                    'prediction': prediction,
#                    'imageAnnotations': str(imageAnnotations),
#                    'model_type': model_type}

#         files = [('resource', (f"{label_tag}.png",
#                                open(filename, 'rb'),
#                                'image/png'))]
#         headers = {}
#         print(payload)
#         response = requests.request('POST', url, headers=headers, data=payload, files=files, verify=False)
#         if True:
#         # if response.status_code == 200:
#             if True:
#                 print('Successfully sent to AutoAI')
#             return True
#         else:
#             if True:
#                 print('Error while sending to AutoAI')
#                 print(response.text)
#                 print(response)
#             return False
#     except Exception as e:
#         print(traceback.format_exc())
#         if True:
#             print('Error while sending data to Auto AI : ', e)
#         return False

def send_lines(model, filename, datad, label_tag, preds, orig_size, csv, tagd):
    try:
        annotations = datad
        tag = str(datetime.date.today()) + "_" + tagd
        label = orig_size
        filename = filename
        model_id = model
        # autoai_url = "https://autoai.techolution.com/"
        autoai_url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/"
        response = send_to_autoai_annotation('backlog', csv, model_id, label, tag, 0, preds,
                                             'image', filename, autoai_url + 'resource', annotations, label_tag)

    except Exception as exception:
        if True:
            print('Error while sending linegrids to Auto AI : ', exception)
            print('Detailed error: {}'.format(traceback.print_exc()))
        try:
            os.remove(new_filename)
        except:
            pass
        return False
    return response


def filter_background(img_path):
    st = time.time()
    imgx = cv2.imread(img_path)
    # print(imgx)
    # print(imgx.shape)
    # print(img_path)
    fin = cv2.hconcat([imgx[:, :50] for x in range(round(abs(imgx.shape[0] - imgx.shape[1]) / 50) - 5)])
    fin = cv2.hconcat([fin, imgx])
    print("Shape : ", fin.shape, round(abs(imgx.shape[0] - imgx.shape[1]) / 50))
    result = model.Segmentation(
        images=[fin],
        paths=None,
        batch_size=1,
        input_size=640,
        visualization=False)
    print("Background Subtraction : ", time.time() - st)
    return result[0]['front'][:, :, ::-1][:, (round(abs(imgx.shape[0] - imgx.shape[1]) / 50) - 5) * 50:], result[0][
        'mask']


def calculate_edges(req_rec, img_obj, target):
    st = time.time()
    image = img_obj.copy()
    if target == "Screw":
        image_th = cv2.imread(req_rec.screw_path)
        image_th = cv2.cvtColor(image_th, cv2.COLOR_BGR2GRAY)
        edges_th = canny(image_th, 2, 1, 25)
        y_co_th, x_co_th = np.where(edges_th * 1.0)
        cols_th = [(y, x) for x, y in zip(x_co_th, y_co_th)]
        targ_th = np.expand_dims(edges_th * 255.0, axis=-1)
        imgx_th = np.concatenate((targ_th, targ_th, targ_th), axis=-1)


    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = canny(image, 2, 1, 25)
    # t_lower = 2
    # t_upper = 150
    # edges = cv2.Canny(image, t_lower, t_upper).astype("bool")
    # lines = probabilistic_hough_line(edges, threshold=10, line_length=10,
    #                                  line_gap=20)
    # x_val = []
    # for l in lines:
    #     x_val.append(l[0][1])
    #     x_val.append(l[1][1])
    y_co, x_co = np.where(edges * 1.0)
    cols_bg_sep = [(y, x) for x, y in zip(x_co, y_co)]
    print("Length Calc : ", time.time() - st)
    st1 = time.time()
    targ = np.expand_dims(edges * 255.0, axis=-1)
    imgx = np.concatenate((targ, targ, targ), axis=-1)
    # imgx = edges.astype(np.uint8)  #convert to an unsigned byte
    # imgx*=255
    # print(imgx)
    if target == "Screw":
        min_th, max_th, imgy = process_for_thread(req_rec, edges_th, cols_th, 255 - imgx_th, cols_bg_sep)
        img_head, img_neck, head_coord, neck_coord = head_and_neck_diameter(image, edges, y_co, x_co)
    print("Thread Calc : ", time.time() - st1)
    # # for line in lines:
    # #     imgx = cv2.line(imgx, line[0], line[1], (255, 0, 0), 1)
    # if status!='calib':
    #     cv2.imwrite("out_data/screw.png", imgx)
    # else:
    #     cv2.imwrite("out_data/calib.png", imgx)
    # imgz = image.copy()
    # imgz = cv2.line(imgz, (500, min(x_val)), (9000, min(x_val)), (255, 0, 0), 1)
    # imgz = cv2.line(imgz, (500, max(x_val)), (9000, max(x_val)), (0, 0, 255), 1)
    # if status!='calib':
    #     cv2.imwrite("out_data/screw_marked_meas.png", imgz)
    # else:
    #     cv2.imwrite("out_data/calib_marked_meas.png", imgz)
    if target == "Screw":
        return (max(y_co), min(y_co)), imgx, (min_th, max_th), imgy, img_head, img_neck, head_coord, neck_coord
    else:
        return (max(y_co), min(y_co)), imgx


def send_results_to_autoai_concat(status, results_at, model, img_obj_1, img_obj_2, label_tag, preds, orig_size, csv=' ',
                                  meta_data=[]):
    if str(type(img_obj_1)) == "<class 'str'>":
        img_obj_1 = cv2.imread(img_obj_1)
    if str(type(img_obj_2)) == "<class 'str'>":
        img_obj_2 = cv2.imread(img_obj_2)
    # print("=."*40)
    # print(preds)
    # print("=."*40)
    data = {}
    print(meta_data)
    if len(meta_data) != 0:
        for d in meta_data:
            data[str([(50, d[0][0]), (9000, d[0][0])])] = d[1]
            data[str([(50, d[0][1]), (9000, d[0][1])])] = d[1]
    data = json_creater(data, False)
    combined = np.concatenate((img_obj_1, img_obj_2), axis=1)
    # combined = img_obj_2.copy()
    idx = random.randint(1, 99999999)
    if status != 'calib':
        cv2.imwrite(f"{idx}_marked_screw.png", combined)
        img_path = f"{idx}_marked_screw.png"

    tagd = "Screw_Length"
    csv = results_at
    th = threading.Thread(target=send_lines, args=(model, img_path, data, label_tag, preds, orig_size, csv, tagd,))
    # send_lines(model, img_path, data, label_tag, preds, orig_size, csv)
    th.start()


def send_results_to_autoai(model, img_obj, label_tag, orig_size, preds, target_process, csv=' ', meta_data=[]):
    idx = random.randint(1, 99999999)
    if str(type(img_obj)) != "<class 'str'>":
        img_obj = cv2.imwrite(f"{idx}_temp.png", img_obj)
        img_obj = f"{idx}_temp.png"

    data = {}
    print(meta_data)
    if target_process == "screw_length":
        tagd = "Screw_Length"
        if len(meta_data) != 0:
            for d in meta_data:
                data[str([(50, d[0][0]), (9000, d[0][0])])] = d[1]
                data[str([(50, d[0][1]), (9000, d[0][1])])] = d[1]
    if target_process == "screw_thread_diameter":
        tagd = "Screw_Thread_Diameter"
        if len(meta_data) != 0:
            for d in meta_data:
                data[str([(d[0][0], 50), (d[0][0], 9000)])] = d[1]
                data[str([(d[0][1], 50), (d[0][1], 9000)])] = d[1]

    if target_process == "screw_thread_diameter_prs":
        tagd = "Screw_Thread_Diameter"
        data = {}
    if target_process == "screw_head_diameter_prs":
        tagd = "Screw_Head_Diameter"
        data = {}
    if target_process == "screw_neck_diameter_prs":
        tagd = "Screw_Neck_Diameter"
        data = {}
    if target_process == "raw":
        tagd = "Screw_Raw_Image"

        data = {}
    # print("=."*40)
    # print(data)
    # print("=."*40)
    data = json_creater(data, False)

    th1 = threading.Thread(target=send_lines, args=(model, img_obj, data, label_tag, preds, orig_size, csv, tagd,))
    # data = {}
    # print(thread_plot)
    # if thread_plot!=' ':
    #     print("Running")
    #     print("=."*40)
    #     plot_path = "thread_plot_autoai.png"
    #     cv2.imwrite(plot_path, thread_plot)
    # th2 = threading.Thread(target = send_lines, args=(model, img_obj, data, label_tag, preds, orig_size, csv, tagd, ))
    # th2.start()
    th1.start()


def main(calib_path, screw_path):
    global status
    # === === Background Seperation

    calib_obj, calib_mask = filter_background(calib_path)
    screw_obj, screw_mask = filter_background(screw_path)
    screw_obj = cv2.cvtColor(screw_obj, cv2.COLOR_BGR2RGB)
    calib_obj = cv2.cvtColor(calib_obj, cv2.COLOR_BGR2RGB)
    cv2.imwrite("bg_sep_screw.png", screw_obj)
    # cv2.imwrite("bg_sep_calib.png", calib_obj)

    # === === Send Background Filter Results
    status = "screw"
    send_results_to_autoai_concat(status, screw_path, screw_obj, f"Screw_{datetime.datetime.now()}_Background",
                                  orig_size=str(screw_orig))


    # === === Edge Detection

    status = "screw"
    screw_edges, screw_lines = calculate_edges(screw_obj)

    # === === Measurement Calculation

    screw_vald = screw_orig / (screw_edges[0] - screw_edges[1])


    results = {"Screw": {
        "mm/px": screw_vald,
        "Detected y co-ord": screw_edges,
        "Detected_size": calib_vald * (screw_edges[0] - screw_edges[1]),
        "Error": (screw_orig - calib_vald * (screw_edges[0] - screw_edges[1]))
    },
    }

    # === === Send Edge Detection Results
    status = "screw"
    send_results_to_autoai_concat(screw_obj, screw_lines, f"Screw_{datetime.datetime.now()}_Edge",
                                  orig_size=str(screw_orig))


    # === === Send Measurements  Results
    status = "screw"
    send_results_to_autoai(screw_obj, f"Screw_{datetime.datetime.now()}_Edge_Marked", str(screw_orig), str(results),
                           [[screw_edges, "Detected"]])



# parser = argparse.ArgumentParser(description='CMD Arguments')

# parser.add_argument("--calib", default="/home/jupyter/shivansh/inspection_res/globus/golden.png",
#                     help="Calibration Object Path")
# parser.add_argument("--screw", default="/home/jupyter/shivansh/inspection_res/globus/blue.png",
#                     help="Screw Path")

# args = parser.parse_args()

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


def background_process(req_rec):
    # if str(type(calisb_path))!= "<class 'str'>":
    #     cv2.imwrite("calib_obj.png", calib_path)
    #     calib_path = "calib_obj.png"
    # if str(type(screw_path))!= "<class 'str'>":
    #     cv2.imwrite("screw_obj.png", screw_path)
    #     screw_path = "screw_obj.png"
    calib_path, screw_path = req_rec.calib_path, req_rec.screw_path

    th2 = ThreadWithReturnValue(target=filter_background, args=(screw_path,))
    th2.start()

    screw_obj, screw_mask = th2.join()
    # calib_obj, calib_mask = filter_background(calib_path)
    # screw_obj, screw_mask = filter_background(screw_path)
    # screw_obj = cv2.cvtColor(screw_obj, cv2.COLOR_BGR2RGB)
    # calib_obj = cv2.cvtColor(calib_obj, cv2.COLOR_BGR2RGB)
    cv2.imwrite("bg_sep_screw.png", screw_obj)

    # === === Send Background Filter Results

    #     model = "63ce4c0f8d150c4a96375b9f"
    #     send_results_to_autoai_concat(model, screw_path, screw_obj, f"Screw_{datetime.datetime.now()}_Background", orig_size=str(screw_orig))
    #     send_results_to_autoai_concat(model, calib_path, calib_obj, f"Calib_{datetime.datetime.now()}_Background", orig_size=str(calib_orig))


    return screw_obj, screw_path


def edge_detection(req_rec, screw_obj, calib_obj=""):
    # status = "calib"
    # calib_edges, calib_lines = calculate_edges(calib_obj)
    # status = "screw"
    # screw_edges, screw_lines = calculate_edges(screw_obj)

    target = "Screw"
    th2 = ThreadWithReturnValue(target=calculate_edges, args=(req_rec, screw_obj, target,))
    th2.start()

    screw_edges, screw_lines, thread_coords_screw, thread_plot, img_head, img_neck, head_coord, neck_coord = th2.join()
    # cv2.imwrite("verify.png", screw_obj)
    # === === Send Edge Detection Results

    #     model = "63d0f9a24a604211ab8e38e2"
    #     send_results_to_autoai_concat(model, screw_obj, screw_lines, f"Screw_{datetime.datetime.now()}_Edge", orig_size=str(screw_orig))

    #     send_results_to_autoai_concat(model, calib_obj, calib_lines, f"Calib_{datetime.datetime.now()}_Edge", orig_size=str(calib_orig))


    return screw_edges, screw_lines, thread_coords_screw, thread_plot, img_head, img_neck, head_coord, neck_coord


def measurements(req_rec, screw_edges, screw_obj, screw_lines, thread_edges, thread_plot, head_coord, neck_coord,
                 calib_edges="", calib_obj="", calib_lines=""):
    screw_vald = req_rec.screw_orig / (screw_edges[0] - screw_edges[1])
    print("=.=" * 50)
    print("Target Calibration Constant : ", screw_vald)
    print("=.=" * 20)
    thread_calib_const = req_rec.thread_orig / abs(thread_edges[1] - thread_edges[0])
    neck_calib_const = req_rec.neck_orig / abs(neck_coord[1][0] - neck_coord[0][0])
    head_calib_const = req_rec.head_orig / abs(head_coord[1][0] - head_coord[0][0])
    print("Target Calibration Constant Thread : ", thread_calib_const)
    print("Target Calibration Constant Neck : ", neck_calib_const)
    print("Target Calibration Constant Head : ", head_calib_const)
    logger.info("%s %s %s %s" % (str(screw_vald),
                                  str(req_rec.thread_orig / abs(thread_edges[1] - thread_edges[0])),
                                  str(req_rec.neck_orig / abs(neck_coord[1][0] - neck_coord[0][0])),
                                  str(req_rec.head_orig / abs(head_coord[1][0] - head_coord[0][0]))))

    print("Target from Request Thread : ", req_rec.calib_const_thread)
    print("=.=" * 50)
    print("Screw Nums : ", req_rec.screw_orig / (screw_edges[0] - screw_edges[1]))
    if req_rec.pair_status:
        print("Screw Nums : ", req_rec.calib_orig / (calib_edges[0] - calib_edges[1]))
    print("Screw Orig : ", req_rec.screw_orig)



    calib_vald = req_rec.calib_const
    print("=.=" * 50)
    print("Selected Calibration Constant : ", calib_vald, req_rec.calib_const)
    print("=.=" * 50)
    results = {"Screw": {
        "mm/px": screw_vald,
        "Y Coord Screw_Length": screw_edges,
        "Detected_Screw_Length": round(calib_vald * (screw_edges[0] - screw_edges[1]), 4),
        "Length_Error": (req_rec.screw_orig - calib_vald * (screw_edges[0] - screw_edges[1])),
        "Thread_Edges": "Min X - " + str(thread_edges[0]) + "  Max X - " + str(thread_edges[1]),
        "Thread_Error": req_rec.thread_orig - abs(thread_edges[1] - thread_edges[0]) * req_rec.calib_const_thread,
        "Thread_Diameter": round(abs(thread_edges[1] - thread_edges[0]) * req_rec.calib_const_thread, 4),
        "Orig_Thread": req_rec.Orig_Thread,
        "One_Above_Thread": req_rec.One_Above_Thread,
        "Two_Above_Thread": req_rec.Two_Above_Thread,
        "Length_Top": req_rec.Top,
        "Length_Bottom": req_rec.Bottom,
        "Thread_Left": req_rec.Left_Th,
        "Thread_Right": req_rec.Right_Th,
        "Peaks_Left": str(req_rec.thread_p[0]),
        "Peaks_Right": str(req_rec.thread_p[1]),
        "Thread_(mm/px)": req_rec.calib_const_thread,
        "Neck_Edges": "Min X - " + str(neck_coord[0][0]) + "  Max X - " + str(neck_coord[1][0]),
        "Neck_Error": req_rec.neck_orig - abs(neck_coord[1][0] - neck_coord[0][0]) * req_rec.calib_const_neck,
        "Neck_Diameter": round(abs(neck_coord[1][0] - neck_coord[0][0]) * req_rec.calib_const_neck, 4),
        "Head_Edges": "Min X - " + str(head_coord[0][0]) + "  Max X - " + str(head_coord[1][0]),
        "Head_Error": req_rec.head_orig - abs(head_coord[1][0] - head_coord[0][0]) * req_rec.calib_const_head,
        "Head_Diameter": round(abs(head_coord[1][0] - head_coord[0][0]) * req_rec.calib_const_head, 4),
        "thread_calib_const":thread_calib_const,
        "neck_calib_const":neck_calib_const,
        "head_calib_const":head_calib_const,
    },

        "Calib": {
            "mm/px": calib_vald,
        }
    }
    result_alt = f"""<b>Screw<b>:<br><br>
                     mm/px:{screw_vald}<br>
                     Y Coord Screw_Length:{screw_edges}<br>
                     Detected_Screw_Length:{round(calib_vald * (screw_edges[0] - screw_edges[1]), 4)}<br>
                     Length_Error:{(req_rec.screw_orig - calib_vald * (screw_edges[0] - screw_edges[1]))}<br>
                     Thread_Edges:{thread_edges}<br>
                     Thread_Error: {round(req_rec.thread_orig - abs(thread_edges[1] - thread_edges[0]) * req_rec.calib_const_thread, 3)}<br>
                     Thread_Diameter:{round(abs(thread_edges[1] - thread_edges[0]) * req_rec.calib_const_thread, 4)}<br>
                     Orig_Thread:{req_rec.Orig_Thread}<br>
                     One_Above_Thread:{req_rec.One_Above_Thread}<br>
                     Two_Above_Thread:{req_rec.Two_Above_Thread}<br>
                     Length_Top:{req_rec.Top}<br>
                     Length_Bottom:{req_rec.Bottom}<br>
                     Thread_Left:{req_rec.Left_Th}<br>
                     Thread_Right:{req_rec.Right_Th}<br>
                     Peaks_Left : {str(req_rec.thread_p[0])}<br>
                     Peaks_Right : {str(req_rec.thread_p[1])}<br>
                     Thread_(mm/px):{req_rec.calib_const_thread}<br>
                     Neck_Edges : {"Min X - " + str(neck_coord[0][0]) + "  Max X - " + str(neck_coord[1][0])}<br>
                     Neck_Error : {req_rec.neck_orig - abs(neck_coord[1][0] - neck_coord[0][0]) * req_rec.calib_const_neck}<br>
                     Neck_Diameter :{round(abs(neck_coord[1][0] - neck_coord[0][0]) * req_rec.calib_const_neck, 4)}<br>
                     Head_Edges : {"Min X - " + str(head_coord[0][0]) + "  Max X - " + str(head_coord[1][0])}<br>
                     Head_Error : {req_rec.head_orig - abs(head_coord[1][0] - head_coord[0][0]) * req_rec.calib_const_head}<br>
                     Head_Diameter : {round(abs(head_coord[1][0] - head_coord[0][0]) * req_rec.calib_const_head, 4)}<br>
                     Camera : Arducam 108MP<br>
                     Setup : Test bed V2<br><br>
    

               <b>Calib</b>:<br><br>
                     mm/px:{calib_vald}<br><br>
                """

    print(results)
    # === === Send Measurements  Results

    return results, result_alt


def send_raw_data(req_rec, results, screw_path, calib_path, target_process, result_alt):
    model = "63cf835c119468083691e9ea"
    screw_obj = cv2.imread(req_rec.screw_path)


    send_results_to_autoai(model, screw_obj, f"Screw_{req_rec.trigger_time}_Raw_&_Results", str(req_rec.screw_orig),
                           str(results['Screw']['Detected_Screw_Length']), target_process)
    os.remove(req_rec.screw_path)
    try:
        os.remove(req_rec.calib_path)
    except:
        print("Calib Failed")
        pass


def send_data(req_rec, results, screw_obj, result_alt, screw_lines, screw_edges, thread_edges, thread_plot, img_head,
              img_neck, head_coord, neck_coord, calib_edges='', calib_lines='', calib_obj=''):
    model = "63ce4c0f8d150c4a96375b9f"
    send_results_to_autoai_concat("screw", str(result_alt), model, req_rec.screw_path, screw_obj,
                                  f"Screw_{req_rec.trigger_time}_Background",
                                  str(results['Screw']['Detected_Screw_Length']), orig_size=str(req_rec.screw_orig))

    model = "63d0f9a24a604211ab8e38e2"
    send_results_to_autoai_concat("screw", str(result_alt), model, screw_obj, screw_lines,
                                  f"Screw_{req_rec.trigger_time}_Edge", str(results['Screw']['Detected_Screw_Length']),
                                  orig_size=str(req_rec.screw_orig))

    model = "63d0f9b24a60425b858e38e9"

    send_results_to_autoai(model, screw_obj, f"Screw_{req_rec.trigger_time}_Edge_Marked", str(req_rec.screw_orig),
                           str(results['Screw']['Detected_Screw_Length']), "screw_length", str(result_alt),
                           [[screw_edges, "Detected"]])

    model = "63e6543a75f59ba87776c7db"
    send_results_to_autoai(model, screw_obj, f"Screw_{req_rec.trigger_time}_Thread_Edge_Marked",
                           str(req_rec.thread_orig), str(results['Screw']['Thread_Diameter']), "screw_thread_diameter",
                           str(result_alt), [[thread_edges, "Detected"]])
    model = "63eb57522578f31b0e4d7b58"
    send_results_to_autoai(model, thread_plot, f"Screw_{req_rec.trigger_time}_Thread_Edge_Plot",
                           str(req_rec.thread_orig), str(results['Screw']['Thread_Diameter']),
                           "screw_head_diameter_prs", str(result_alt), [[thread_edges, "Detected"]])
    model = "641d8d81e30dcf3dd5b4b786"
    send_results_to_autoai(model, img_neck, f"Screw_{req_rec.trigger_time}_Head_Plot", str(req_rec.thread_orig),
                           str(results['Screw']['Head_Diameter']), "screw_neck_diameter_prs", str(result_alt),
                           [[head_coord, "Detected"]])
    model = "641d8d94e30dcf3f6db4b78d"
    send_results_to_autoai(model, img_head, f"Screw_{req_rec.trigger_time}_Neck_Plot", str(req_rec.thread_orig),
                           str(results['Screw']['Neck_Diameter']), "screw_thread_diameter_prs", str(result_alt),
                           [[neck_coord, "Detected"]])


from scipy.signal import find_peaks
from skimage.draw import line
import numpy as np


def fetch_thread(lc, rc, x_val, y_val):
    top = (list(x_val)[list(y_val).index(min(y_val))], min(y_val))
    bottom = (list(x_val)[list(y_val).index(min(y_val))], max(y_val))
    mid_p = [top[0], round((top[1] + bottom[1]) / 2)]
    sel_pair = ""
    min_d = 10000000
    for p, q in zip(lc, rc):
        x_p, x_q = p[1], q[1]
        dis = (distance.euclidean(mid_p, [list(p)[1], list(p)[0]]) + distance.euclidean(mid_p,
                                                                                        [list(q)[1], list(q)[0]])) / 2
        print("Pair : ", p, q, dis)
        if dis < min_d:
            min_d = dis
            sel_pair = [p, q]
    print("Mid Point ", mid_p)
    print("Sel Point ", sel_pair)
    lc = list(lc)
    rc = list(rc)
    # if sel_pair[0][0]<sel_pair[1][0]:
    #     print()
    if sel_pair[0][0] < sel_pair[1][0]:
        print("Condition SATS ")
        for g in rc[::-1]:
            if g[0] < sel_pair[0][0]:
                sel_pair = [sel_pair[0], g]
                break
    print("P1 : ", sel_pair)
    sel_pair_alt = (
    [sel_pair[0], list(rc)[list(rc).index(sel_pair[1]) - 1]], [sel_pair[0], list(rc)[list(rc).index(sel_pair[1]) - 2]])
    print("Alt_Pairs", sel_pair_alt)
    pair_up = [lc[lc.index(sel_pair[0]) - 1], rc[rc.index(sel_pair[1]) - 1]]
    pair_down = [lc[lc.index(sel_pair[0]) + 1], rc[rc.index(sel_pair[1]) + 1]]

    return sel_pair, pair_up, pair_down, sel_pair_alt, sel_pair, (
    (list(x_val)[list(y_val).index(min(y_val))], min(y_val)), (list(x_val)[list(y_val).index(max(y_val))], max(y_val)))


def process_for_thread(req_rec, edges, x_val, imgy, x_val_bg):
    catl_min = {}
    catl_max = {}
    for p in x_val:
        try:
            catl_min[p[0]] = min(catl_min[p[0]], p[1])
        except:
            catl_min[p[0]] = p[1]
        try:
            catl_max[p[0]] = max(catl_max[p[0]], p[1])
        except:
            catl_max[p[0]] = p[1]
    t_x = max(list(catl_min.values())) - np.array(list(catl_min.values()))
    peaks, _ = find_peaks(t_x, distance=100, prominence=(40, None))
    lc = [(list(catl_min.keys())[y], list(catl_min.values())[y]) for y in peaks]

    t_x = np.array(list(catl_max.values()))
    peaks, _ = find_peaks(t_x, distance=100, prominence=(40, None))
    rc = [(list(catl_max.keys())[y], list(catl_max.values())[y]) for y in peaks]
    if lc[0][0] > rc[0][0]:
        print("Case 1 ")
        for idx, d in enumerate(rc):
            print(idx)
            if lc[0][0] > d[0]:
                continue
            else:
                break
        rc = rc[list(rc).index(d) - 1:]
    if lc[0][0] < rc[0][0]:
        print("Case 2 ")
        for idx, d in enumerate(lc):
            print(idx)
            if rc[0][0] > d[0]:
                continue
            else:
                break
        lc = lc[list(lc).index(d) - 1:]
    #     catl = {}
    #     for j in x_val:
    #         if j[0] not in catl:
    #             catl[j[0]]=[]
    #         catl[j[0]].append(j[1])

    #     final_d = dict(sorted(catl.items()))

    #     max_sel = []
    #     maxi = -1
    #     for key, value in final_d.items():
    #             max_sel.append((key, min(value)))

    #     t_x = [max([x[1] for x in max_sel])-x[1] for x in max_sel]
    #     t_x = np.array(t_x)
    #     peaks, _ = find_peaks(t_x, distance = 100, prominence=(40, None))
    #     lc = [(max_sel[y][0],max_sel[y][1])  for y in peaks]
    # ===================
    #     max_sel = []
    #     maxi = -1
    #     for key, value in final_d.items():
    #             max_sel.append((key, max(value)))

    #     t_x = [x[1] for x in max_sel]
    #     t_x = np.array(t_x)
    #     peaks, _ = find_peaks(t_x, distance = 100, prominence=(40, None))
    #     rc = [max_sel[y] for y in peaks]
    (sti, edi), up, dow, sel_pair_alt, sel_pair_orig, length_cod = fetch_thread(lc, rc, [x[1] for x in x_val_bg],
                                                                                [x[0] for x in x_val_bg])
    print("Threads : ", (sti, edi), up, dow)
    print("Diameter : ", round(req_rec.calib_const_thread * abs(sti[1] - edi[1]), 4),
          round(req_rec.calib_const_thread * abs(up[0][1] - up[1][1]), 4),
          round(req_rec.calib_const_thread * abs(dow[0][1] - dow[1][1]), 4))
    print("Error : ", round((req_rec.thread_orig - req_rec.calib_const_thread * abs(sti[1] - edi[1])), 4),
          round((req_rec.thread_orig - req_rec.calib_const_thread * abs(up[0][1] - up[1][1])), 4),
          round((req_rec.thread_orig - req_rec.calib_const_thread * abs(dow[0][1] - dow[1][1])), 4)
          )

    req_rec.One_Above_Thread = round(
        (req_rec.thread_orig - req_rec.calib_const_thread * abs(sel_pair_alt[0][0][1] - sel_pair_alt[0][1][1])), 4)
    req_rec.Two_Above_Thread = round(
        (req_rec.thread_orig - req_rec.calib_const_thread * abs(sel_pair_alt[1][0][1] - sel_pair_alt[1][1][1])), 4)
    # if sel_pair[0][0]<sel_pair[1][0]:
    #     req_rec.Orig_Thread = round((req_rec.thread_orig- req_rec.calib_const_thread*abs(sel_pair_orig[0][1]-sel_pair_orig[1])), 4)
    # else:
    #     req_rec.Orig_Thread = round((req_rec.thread_orig- req_rec.calib_const_thread*abs(sti[1]-edi[1])), 4)
    req_rec.Orig_Thread = round(
        (req_rec.thread_orig - req_rec.calib_const_thread * abs(sel_pair_orig[0][1] - sel_pair_orig[1][1])), 4)
    req_rec.Top = length_cod[0]
    req_rec.Bottom = length_cod[1]
    # x_val = [x[1] for x in lc][2:-2]
    # st = min(x_val)
    # x_val = [x[1] for x in rc][2:-2]
    # ed = max(x_val)
    req_rec.thread_p = [[x[::-1] for x in lc], [x[::-1] for x in rc]]
    st, ed = sti[1], edi[1]

    for idz, line in enumerate(lc):
        imgy = cv2.circle(imgy, (line[1], line[0]), radius=3, color=(255, 0, 255), thickness=2)
        # imgy = cv2.line(imgy, (line[1], line[0]), (rc[idz][1], rc[idz][0]), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_4)
    for line in rc:
        imgy = cv2.circle(imgy, (line[1], line[0]), radius=3, color=(255, 0, 255), thickness=2)
    imgy = cv2.circle(imgy, (st, lc[[x[1] for x in lc].index(st)][0]), radius=5, color=(0, 0, 255), thickness=3)
    imgy = cv2.circle(imgy, (ed, rc[[x[1] for x in rc].index(ed)][0]), radius=5, color=(0, 0, 255), thickness=3)
    # imgy = cv2.line(imgy, (st, lc[[x[1] for x in lc].index(st)][0]), (ed ,lc[[x[1] for x in lc].index(st)][0]), color=(255, 0, 0), thickness=2)
    # imgy = cv2.line(imgy, (st, lc[[x[1] for x in lc].index(st)][0]), (ed ,lc[[x[1] for x in lc].index(st)][0]), color=(255, 0, 0), thickness=2)
    # imgy = cv2.line(imgy, (ed, rc[[x[1] for x in rc].index(ed)][0]), (rc[idz][1], rc[idz][0]), color=(255, 0, 0), thickness=2)
    # imgy = cv2.line(imgy, (ed, rc[[x[1] for x in rc].index(ed)][0]), (ed, lc[[x[1] for x in lc].index(st)][0]), color=(255, 0, 0), thickness=2)
    # cv2.imwrite("out_image_autoai.png", imgy)
    print("L1 : ", (st, lc[[x[1] for x in lc].index(st)][0]), (ed, lc[[x[1] for x in lc].index(st)][0]))
    print("L2 : ", (ed, rc[[x[1] for x in rc].index(ed)][0]), (ed, lc[[x[1] for x in lc].index(st)][0]))
    print("Cords : ", sti, edi)
    imgy = cv2.line(imgy, (sti[1], sti[0]), (edi[1], sti[0]), color=(255, 0, 0), thickness=2)
    imgy = cv2.line(imgy, (edi[1], edi[0]), (edi[1], sti[0]), color=(255, 0, 0), thickness=2)
    x_val = [x[1] for x in x_val_bg]
    y_val = [x[0] for x in x_val_bg]
    top = (list(x_val)[list(y_val).index(min(y_val))], min(y_val))
    bottom = (list(x_val)[list(y_val).index(min(y_val))], max(y_val))
    mid_p = [top[0], round((top[1] + bottom[1]) / 2)]
    imgy = cv2.circle(imgy, (mid_p[0], mid_p[1]), radius=8, color=(0, 0, 255), thickness=3)
    # st, ed = sel_pair_alt[0][0][1], sel_pair_alt[0][1][1]
    req_rec.Left_Th = sti
    req_rec.Right_Th = edi
    return st, ed, imgy


def prepare_url(gcs_path, local_path):
    upload_image(gcs_path, local_path)
    signed_url = signedurl(gcs_path)

    return signed_url


def main_call(req_rec):
    sti = time.time()


    screw_obj, scrw_path = background_process(req_rec)
    gcs_path_raw, local_path_raw = f"{datetime.datetime.now()}_Raw.png", req_rec.raw_image_path
    gcs_path_report, local_path_report = f"{datetime.datetime.now()}_Report.png", "bg_sep_screw.png"
    # upload_image(gcs_path_raw, local_path_raw)
    # upload_image(gcs_path_report, local_path_report)
    # signed_url_raw,  signed_url_report = signedurl(gcs_path_raw), signedurl(gcs_path_report)
    th1 = ThreadWithReturnValue(target=prepare_url, args=(gcs_path_raw, local_path_raw,))
    th2 = ThreadWithReturnValue(target=prepare_url, args=(gcs_path_report, local_path_report,))
    th1.start()
    th2.start()

    st = time.time() - sti
    print("=." * 40)
    print("Background Time : ", st)
    print("=." * 40)
    st1 = time.time()


    screw_edges, screw_lines, thread_edges, thread_plot, img_head, img_neck, head_coord, neck_coord = edge_detection(
            req_rec, screw_obj)
    # cv2.imwrite("verify.png", screw_edges)
    # print(screw_edges)
    # print(screw_edges.shape)
    st = time.time() - sti
    print("=." * 40)
    print("Edge Time : ", st)
    print("=." * 40)

    mst = time.time()
    results, result_alt = measurements(req_rec, screw_edges, screw_obj, screw_lines, thread_edges, thread_plot,
                                       head_coord, neck_coord)
    print("Measurement Time : ", time.time() - mst)
    th = threading.Thread(target=send_data, args=(
    req_rec, results, screw_obj, result_alt, screw_lines, screw_edges, thread_edges, thread_plot, img_head,
    img_neck, head_coord, neck_coord,))
    th.start()

    th = threading.Thread(target=send_raw_data,
                          args=(req_rec, results, req_rec.screw_path, req_rec.calib_path, "raw", result_alt,))
    th.start()
    # send_raw_data(results, screw_path, calib_path)
    print("=." * 40)
    print("Complete Time : ", time.time() - sti)
    print("=." * 40)
    signed_url_raw = th1.join()
    signed_url_report = th2.join()
    return results, signed_url_raw, signed_url_report


import numpy as np
from PIL import Image
import os
from scipy.spatial import distance


def process_raw_file(input_path, output_path):
    def convert(input_path, output_path):
        # Open the raw data as a binary file
        with open(input_path, "rb") as f:
            raw_data = f.read()

        # Specify the width and height of the image
        width = 12000
        height = 9000

        # Decode the raw data into a 2D numpy array
        image_array = np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width)

        # Convert the numpy array to a PIL Image object
        image = Image.fromarray(image_array)

        # Convert the image to 8-bit RGB
        image = image.convert("RGB")

        # Rotate image by 180 degree
        image = image.rotate(180)

        # Save the image as a PNG file
        image.save(output_path)

    files = os.listdir(input_folder)

    for index, file in enumerate(files):
        convert(input_path, output_paths)


@app.route("/", methods=["GET"])
def status():
    return "Server is Up"


@app.route("/fetch_accuracy/", methods=["POST"])
def run():
    # global calib_orig, screw_orig, thread_orig
    # data = eval(request.files['data'].read().decode())
    inference_seed = random.randint(1, 99999999)
    inference_start_time = datetime.datetime.now().isoformat()
    print("Entered")
    calib_orig = float(request.form['calibration_length'])
    screw_orig = float(request.form['target_length'])
    thread_orig = float(request.form['thread_length'])
    neck_orig = float(request.form['neck_length'])
    head_orig = float(request.form['head_length'])
    screw_path = f"screw_p{inference_seed}.png"
    calib_path = f"calib_p{inference_seed}.png"
    pair_status = eval(request.form['pair_status'])
    calib_const = eval(request.form['calib_const'])
    calib_const_thread = eval(request.form['calib_const_thread'])
    calib_const_head = eval(request.form['calib_const_head'])
    print("Recieved Head: ", calib_const_head)
    calib_const_neck = eval(request.form['calib_const_neck'])
    raw_file = request.files['live_ui_image']
    raw_file.save("raw_screw.png")
    raw_image_path = "raw_screw.png"
    trigger_time = str(datetime.datetime.now())
    One_Above_Thread = -999.0
    Two_Above_Thread = -999.0
    Orig_Thread = -999.0
    thread_p = ""
    req_rec = Reques_Data(calib_path, screw_path, calib_orig, screw_orig, thread_orig, pair_status, calib_const,
                          calib_const_thread, raw_image_path, trigger_time, One_Above_Thread, Two_Above_Thread,
                          Orig_Thread, thread_p, neck_orig, head_orig, calib_const_neck, calib_const_head)
    if request.form['target_length'] == 'raw':
        calib = request.files['calibration_file']
        calib.save("calib_p.raw")
        screw = request.files['target_file']
        screw.save("screw_p.raw")
        process_raw_file("calib_p.raw", "calib_p.png")
        process_raw_file("screw_p.raw", "screw_p.png")

    else:

        screw = request.files['target_file']
        screw.save(screw_path)
    result, signed_url_raw, signed_url_report = main_call(req_rec)
    # print(data)
    status_c = 0

    ################################################ FIXED OFFSET ###################################################
    # Thread diameter = +0.01
    # Head diameter = +0.01
    # Length = +0.01
    result['Screw']['Detected_Screw_Length'] = result['Screw']['Detected_Screw_Length'] + 0.01
    result['Screw']['Thread_Diameter'] = result['Screw']['Thread_Diameter'] + 0.01
    # result['Screw']['Head_Diameter'] = result['Screw']['Head_Diameter'] + 0.01

    if round(result['Screw']['Detected_Screw_Length'], 3) <= 17.5 and round(result['Screw']['Detected_Screw_Length'],
                                                                            3) >= 17.2:
        status_c += 1
    if round(result['Screw']['Thread_Diameter'], 3) <= 3.60 and round(result['Screw']['Thread_Diameter'], 3) >= 3.55:
        status_c += 1
    if (result['Screw']['Thread_Diameter'] >= 3.55 and result['Screw']['Thread_Diameter'] <= 3.60) and (
            result['Screw']['Detected_Screw_Length'] >= 17.2 and result['Screw']['Detected_Screw_Length'] <= 17.5) and (
            result['Screw']['Neck_Diameter'] >= 3.82 and result['Screw']['Neck_Diameter'] <= 3.88) and (
            result['Screw']['Head_Diameter'] >= 4.47 and result['Screw']['Head_Diameter'] <= 4.5):
        status = 'Pass'
    else:
        status = 'Fail'
    inference_completion_time = datetime.datetime.now().isoformat()

    ref.set({'screw': {'liveImages': {'image': {'url': signed_url_raw},
                                      'part': str(184.156),
                                      'qaBatch': str(1),
                                      'qaFail': str(2 - status_c),
                                      'qaPass': str(status_c),
                                      'qaResults': status},
                       'reportImage': {'url': signed_url_report},
                       'reports': [{'currentReading': '.'.join(
                           str(result['Screw']['Detected_Screw_Length']).split('.')[:-1] + [
                               str(result['Screw']['Detected_Screw_Length']).split('.')[-1][:3]]),
                                    'parameter': 'Screw Length',
                                    'threshold': {'min': "17.2", 'max': "17.5"}},
                                   {'currentReading': '.'.join(
                                       str(result['Screw']['Thread_Diameter']).split('.')[:-1] + [
                                           str(result['Screw']['Thread_Diameter']).split('.')[-1][:3]]),
                                    'parameter': 'Thread Diameter',
                                    'threshold': {'min': "3.55", 'max': "3.60"}},
                                   {'currentReading': '.'.join(str(result['Screw']['Neck_Diameter']).split('.')[:-1] + [
                                       str(result['Screw']['Neck_Diameter']).split('.')[-1][:3]]),
                                    'parameter': 'Neck Diameter',
                                    'threshold': {'min': "3.82", 'max': "3.88"}},
                                   {'currentReading': '.'.join(str(result['Screw']['Head_Diameter']).split('.')[:-1] + [
                                       str(result['Screw']['Head_Diameter']).split('.')[-1][:3]]),
                                    'parameter': 'Head Diameter',
                                    'threshold': {'min': "4.47", 'max': "4.5"}}],
                       'stationInfo': {'alertIndicators': 'All OK',
                                       'alertStatus': 'Good (No Problems)',
                                       'needsMaintenance': 'no maintenance required',
                                       'stationNo': 1},
                       'current_time': f"{datetime.datetime.now()}",
                       'captureTime': inference_start_time,
                       'inferenceTime': inference_completion_time,
                       }})

    return jsonify({"Status": "Successful", "Data": str(result)})

@app.route("/calibration/", methods=["POST"])
def calibration():

    inference_seed = random.randint(1, 99999999)
    inference_start_time = datetime.datetime.now().isoformat()
    print("Entered")
    calib_orig = float(request.form['calibration_length'])
    screw_orig = float(request.form['target_length'])
    thread_orig = float(request.form['thread_length'])
    neck_orig = float(request.form['neck_length'])
    head_orig = float(request.form['head_length'])
    screw_path = f"screw_p{inference_seed}.png"
    calib_path = f"calib_p{inference_seed}.png"
    pair_status = eval(request.form['pair_status'])
    calib_const = eval(request.form['calib_const'])
    calib_const_thread = eval(request.form['calib_const_thread'])
    calib_const_head = eval(request.form['calib_const_head'])
    print("Recieved Head: ", calib_const_head)
    calib_const_neck = eval(request.form['calib_const_neck'])
    raw_file = request.files['live_ui_image']
    raw_file.save("raw_screw.png")
    raw_image_path = "raw_screw.png"
    trigger_time = str(datetime.datetime.now())
    One_Above_Thread = -999.0
    Two_Above_Thread = -999.0
    Orig_Thread = -999.0
    thread_p = ""
    req_rec = Reques_Data(calib_path, screw_path, calib_orig, screw_orig, thread_orig, pair_status, calib_const,
                          calib_const_thread, raw_image_path, trigger_time, One_Above_Thread, Two_Above_Thread,
                          Orig_Thread, thread_p, neck_orig, head_orig, calib_const_neck, calib_const_head)
    if request.form['target_length'] == 'raw':
        calib = request.files['calibration_file']
        calib.save("calib_p.raw")
        screw = request.files['target_file']
        screw.save("screw_p.raw")
        process_raw_file("calib_p.raw", "calib_p.png")
        process_raw_file("screw_p.raw", "screw_p.png")

    else:

        screw = request.files['target_file']
        screw.save(screw_path)
    result, signed_url_raw, signed_url_report = main_call(req_rec)
    # print(data)
    status_c = 0

    ################################################ FIXED OFFSET ###################################################
    # Thread diameter = +0.01
    # Head diameter = +0.01
    # Length = +0.01
    result['Screw']['Detected_Screw_Length'] = result['Screw']['Detected_Screw_Length'] + 0.01



    return jsonify({"Status": "Successful", "Data": str(result)})



if __name__ == '__main__':
    app.run(host='10.160.15.195', port=8501, threaded=True)

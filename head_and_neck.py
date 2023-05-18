import glob, cv2, time
import numpy as np
from scipy.signal import find_peaks
from skimage.feature import canny
from scipy.spatial import distance
from collections import Counter
from tqdm import tqdm


def head_and_neck_diameter(image_th, edges_th, y_co_th, x_co_th):
    st = time.time()
    # top1, bottom1 = (x_co[y_co.index(min(y_co))], min(y_co)), (x_co[y_co.index(max(y_co))], max(y_co))
    targ_th = np.expand_dims(edges_th * 255.0, axis=-1)
    imgx_th = np.concatenate((targ_th, targ_th, targ_th), axis=-1)
    x_val = [(y, x) for x, y in zip(x_co_th, y_co_th)]
    # imgy, lc, rc = process_thread(x_val, 255-imgx_th)
    x_val = [(y, x) for x, y in zip(x_co_th, y_co_th)]
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

    lcd = [(x, y) for y, x in catl_min.items()]
    rcd = [(x, y) for y, x in catl_max.items()]

    left = {}
    for pk in lc[::-1]:
        if pk[0] + 50 >= max(y_co_th):
            continue
        pk = (pk[1], pk[0])
        val_d = []
        temp = pk[0]
        sum_d = 0
        for idx in range(lcd.index(pk), lcd.index(pk) + 50):
            val_d.append(lcd[idx][0] - temp)
            sum_d += lcd[idx][0] - temp
            temp = lcd[idx][0]
        left[pk] = abs(sum_d / 50) * 100
        # print(pk, lcd.index(pk), abs(sum_d/50)*100, val_d)

    right = {}
    for pk in rc[::-1]:
        if pk[0] + 50 >= max(y_co_th):
            continue
        pk = (pk[1], pk[0])
        val_d = []
        temp = pk[0]
        sum_d = 0
        for idx in range(rcd.index(pk), rcd.index(pk) + 50):
            val_d.append(rcd[idx][0] - temp)
            sum_d += rcd[idx][0] - temp
            temp = rcd[idx][0]
        right[pk] = abs(sum_d / 50) * 100
        # print(pk, rcd.index(pk), abs(sum_d/50)*100, val_d)
    past = list(left.items())[0][1]
    l_t = ""
    for key, value in left.items():
        if value - past > 50:
            l_t = key
            break
    past = list(left.items())[0][1]
    r_t = ""
    for key, value in right.items():
        if value - past > 50:
            r_t = key
            break
    vald = dict(Counter([x[0] for x in rcd[rcd.index(r_t):]]))
    vald = sorted(vald.items(), key=lambda x: x[1], reverse=True)
    vald1 = vald
    vald = dict(Counter([x[0] for x in lcd[lcd.index(l_t):]]))
    vald = sorted(vald.items(), key=lambda x: x[1], reverse=True)
    vald2 = vald
    head_l = (min([x[0] for x in lcd]), [x[1] for x in lcd if x[0] == min([x[0] for x in lcd])][0])
    head_r = (max([x[0] for x in rcd]), [x[1] for x in rcd if x[0] == max([x[0] for x in rcd])][0])
    # print(head_l, head_r)

    min_diff = 99999
    sel_ri = ""
    for z in vald1:
        if z[0] > 20 and [x[1] for x in rcd[rcd.index(r_t):] if x[0] == z[0]][0] < head_r[1] and (
                head_r[1] - [x[1] for x in rcd[rcd.index(r_t):] if x[0] == z[0]][0]) > 200:
            # print((head_r[1], [x[1] for x in rcd[rcd.index(r_t):] if x[0]==z[0]][0]))
            diff = head_r[1] - [x[1] for x in rcd[rcd.index(r_t):] if x[0] == z[0]][0]
            if diff > 0 and diff < min_diff:
                sel_ri = z
                min_diff = diff

    min_diff = 99999
    sel_le = ""
    for z in vald2:
        if z[0] > 20 and [x[1] for x in lcd[lcd.index(l_t):] if x[0] == z[0]][0] < head_l[1] and (
                head_l[1] - [x[1] for x in lcd[lcd.index(l_t):] if x[0] == z[0]][0]) > 200:
            diff = head_l[1] - [x[1] for x in lcd[lcd.index(l_t):] if x[0] == z[0]][0]
            if diff > 0 and diff < min_diff:
                sel_le = z
                min_diff = diff

    imgy = (255 - imgx_th).copy()
    for line in lc:
        imgy = cv2.circle(imgy, (line[1], line[0]), radius=3, color=(255, 0, 255), thickness=2)
    for line in rc:
        imgy = cv2.circle(imgy, (line[1], line[0]), radius=3, color=(255, 0, 255), thickness=2)
    vald1, vald2 = sel_ri, sel_le
    count = 0
    for d in rcd[rcd.index(r_t):]:
        if d[0] == vald1[0]:
            count += 1
            # if count==8:
            if True:
                mark1 = d
                break
    count = 0
    for d in lcd[lcd.index(l_t):]:
        if d[0] == vald2[0]:
            count += 1
            # if count==8:
            if True:
                mark2 = d
                break
    # print(mark1, mark2)
    # print(head_l, head_r)
    imgyn = imgy.copy()
    imgyn = cv2.circle(imgyn, mark1, radius=7, color=(0, 0, 255), thickness=2)
    imgyn = cv2.circle(imgyn, mark2, radius=7, color=(0, 0, 255), thickness=2)
    imgyn = cv2.circle(imgyn, mark1, radius=1, color=(0, 0, 255), thickness=-1)
    imgyn = cv2.circle(imgyn, mark2, radius=1, color=(0, 0, 255), thickness=1)
    img_neck = cv2.line(imgyn, mark1, (mark2[0], mark1[1]), color=(255, 0, 0), thickness=2)
    img_neck = cv2.line(img_neck, mark2, (mark2[0], mark1[1]), color=(255, 0, 0), thickness=2)

    imgyh = imgy.copy()
    imgyh = cv2.circle(imgyh, head_l, radius=7, color=(255, 0, 0), thickness=2)
    imgyh = cv2.circle(imgyh, head_r, radius=7, color=(255, 0, 0), thickness=2)
    imgyh = cv2.circle(imgyh, head_l, radius=1, color=(255, 0, 0), thickness=-1)
    imgyh = cv2.circle(imgyh, head_r, radius=1, color=(255, 0, 0), thickness=-1)
    img_head = cv2.line(imgyh, head_l, (head_r[0], head_l[1]), color=(0, 0, 255), thickness=2)
    img_head = cv2.line(img_head, head_r, (head_r[0], head_l[1]), color=(0, 0, 255), thickness=2)
    # plt.figure(figsize=(100, 100))
    # plt.imshow(imgy)
    # cv2.imwrite(f"plot.png", imgy)
    return img_neck, img_head, (head_l, head_r), (mark2, mark1)
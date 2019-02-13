import numpy as np
import matplotlib.pyplot as plt
import scipy as s
import imageio as iio
from skimage import io
import time
from operator import itemgetter

def load_data(faces_dir, background_dir, num_images):
    images = []
    types = []
    for i in range(0, num_images):
        img = io.imread(faces_dir +'face' + str(i) +'.jpg', as_grey=True)
        images.append(img)
        types.append(1)
    for i in range(0, num_images):
        img = io.imread(background_dir + str(i) +'.jpg', as_grey=True)
        images.append(img)
        types.append(-1)
    return (np.array(images), np.array(types))

def compute_integral_image(imgs):
    int_imgs = []
    for img in imgs:
        ii = [0]*64
        i_img = []
        for row in img:
            row_sum = 0
            i_row = []
            for x in range(0, 64):
                pixel = row[x]
                row_sum += pixel
                i_row.append(row_sum + ii[x])
                for j in range(0, x+1):
                    ii[j]+=pixel
            i_img.append(np.array(i_row))
        int_imgs.append(np.array(i_img))
    return np.array(int_imgs)

def r(p):
    return (p[1], p[0])

def feature_list(dim):
    feats = []
    for height in range(1, dim+1):
        min = 1
        if height == 1:
            min = 2
        for width in range(min, int(dim/2)+1):
            for ux in range(0, dim+1-2*width):
                for uy in range(0, dim+1-height):
                    a = (ux, uy)
                    b = (ux+width-1, uy+height-1)
                    c = (ux+width, uy)
                    d = (ux+width+width-1, uy+height-1)
                    feats.append((a,b,c,d))
                    if (height != width):
                        feats.append((r(a), r(b), r(c), r(d)))
    return feats

def calculate_avg_intensity(int_img, ul, br):
    (u, l) = ul
    (b, r) = br
    intensity = 0
    area = 0
    if u == 0 and l == 0:
        intensity = int_img[b][r]
        area = (b+1)*(r+1)
    elif u == 0:
        intensity = int_img[b][r]-int_img[b][l-1]
        area = (b+1)*(r-l+1)
    elif l == 0:
        intensity = int_img[b][r]-int_img[u-1][r]
        area = (b-u+1)*(r+1)
    else:
        intensity = int_img[b][r]-int_img[b][l-1]-int_img[u-1][r]+int_img[u-1][l-1]
        area = (b-u+1)*(r-l+1)
    return float(intensity/area)


def compute_feature(int_img_rep, feat):
    (a,b,c,d) = feat
    feat_values = []
    for img in int_img_rep:
        shaded = calculate_avg_intensity(img, a, b)
        unshaded = calculate_avg_intensity(img, c, d)
        feat_values.append(shaded-unshaded)
    return np.array(feat_values)

def opt_p_theta(int_img_rep, feat, weights, y_true, sorted):
    min_error = float('inf')
    p = 0
    theta = 0
    T_plus = 0
    T_minus = 0
    for (y, i) in zip(y_true, range(0, len(y_true))):
        if y == 1:
            T_plus+=weights[i]
        if y == -1:
            T_minus+=weights[i]
    S_plus = 0
    S_minus = 0
    for i in range(0, len(sorted)):
        (f_j, j) = sorted[i]
        e_1 = S_plus + T_minus - S_minus
        e_2 = S_minus + T_plus - S_plus
        e = min(e_1, e_2)
        if e < min_error:
            min_error = e
            if i == len(sorted) - 1:
                theta = f_j + 0.01
            else:
                theta = float((f_j+sorted[i+1][1])/2)
            if e_1 < e_2:
                p = -1
            else:
                p = 1
        y = y_true[j]
        if y == 1:
            S_plus+=weights[j]
        if y == -1:
            S_minus+=weights[j]
    return (p, theta)

def error_rate(values, feat, weights, p, theta, y_true):
    total_error = 0.0
    N = len(values)
    for i in range(0, N):
        pred = p*(values[i]-theta)
        pred = np.sign(pred)
        if pred != y_true[i]:
            total_error+=weights[i]
    return float(total_error/N)

def opt_weaklearner(int_img_rep, weights, feat_list, y_true):
    best_error_rate = float('inf')
    best_feat = -1
    best_p = 0
    best_theta = 0
    for i in range(0, len(feat_list)):
        feat = feat_list[i]
        values = compute_feature(int_img_rep, feat)
        to_sort = [(v, j) for v, j in zip(values, range(0, len(values)))]
        to_sort.sort(key=itemgetter(0))
        (p, theta) = opt_p_theta(int_img_rep, feat, weights, y_true, to_sort)
        e = error_rate(values, feat, weights, p, theta, y_true)
        if e < best_error_rate:
            best_error_rate = e
            best_feat = i
            best_p = p
            best_theta = theta
    return (best_error_rate, best_feat, best_p, best_theta)



def viola_jones():
    start = time.time()
    (imgs, y_true) = load_data('faces/', 'background/', 2)
    print(time.time()-start)
    int_img_rep = compute_integral_image(imgs)
    print(time.time()-start)
    feat_list = feature_list(64)
    print(len(feat_list))
    print(time.time()-start)
    N = len(int_img_rep)
    weights = np.array([float(1/N)]*N)
    print(time.time()-start)
    print(opt_weaklearner(int_img_rep, weights, feat_list, y_true))
    print(time.time()-start)

viola_jones()

import cv2 # opencv 3.4.2+ required
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageChops

def colorizeImage(image_path,input_image,bg_save_path):
    proto = 'Colorization/models/colorization_deploy_v2.prototxt'
    weights = 'Colorization/models/colorization_release_v2.caffemodel'

    # load cluster centers
    pts_in_hull = np.load('Colorization/models/pts_in_hull.npy')
    pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

    # load model
    net = cv2.dnn.readNetFromCaffe(proto, weights)
    # net.getLayerNames()

    # populate cluster centers as 1x1 convolution kernel
    net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
    # scale layer doesn't look work in OpenCV dnn module, we need to fill 2.606 to conv8_313_rh layer manually
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

    # Preprocessing===========================
    # img_path = 'Colorization/img/Object_image.png'
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_input = img.copy()

    # convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_rgb = img.copy()

    # normalize input
    img_rgb = (img_rgb / 255.).astype(np.float32)

    # convert RGB to LAB
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
    # only L channel to be used
    img_l = img_lab[:, :, 0]

    input_img = cv2.resize(img_l, (256, 256))
    input_img -= 50  # subtract 50 for mean-centering

    plt.axis('off')
    plt.imshow(input_img, cmap='gray')

    # Prediction======================================
    net.setInput(cv2.dnn.blobFromImage(input_img))
    pred = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize to original image shape
    pred_resize = cv2.resize(pred, (img.shape[1], img.shape[0]))

    # concatenate with original image L
    pred_lab = np.concatenate([img_l[:, :, np.newaxis], pred_resize], axis=2)

    # convert LAB to RGB
    pred_rgb = cv2.cvtColor(pred_lab, cv2.COLOR_Lab2RGB)
    pred_rgb = np.clip(pred_rgb, 0, 1) * 255
    pred_rgb = pred_rgb.astype(np.uint8)

    # plot prediction result
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(1, 2, 1).axis('off')
    plt.imshow(img_l, cmap='gray')
    fig.add_subplot(1, 2, 2).axis('off')
    plt.imshow(pred_rgb)
    # plt.savefig(output_filename)

    # save result image file
    filename, ext = os.path.splitext(image_path)
    input_filename = '%s_input%s' % (filename, ext)
    output_filename = '%s_output%s' % (filename, ext)
    combined_filename = '%s_combined%s' % (filename, ext)

    pred_rgb_output = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    mask_save_path = 'images/save_color_mask.png'
    cv2.imwrite(mask_save_path, pred_rgb_output)
    cv2.imwrite(input_filename, img_input)

    dst1 = cv2.addWeighted(input_image, 0.5, pred_rgb_output, 1, 0)

    cv2.imwrite(combined_filename, dst1)
    cv2.imshow('colorizedImg',dst1)
    cv2.imwrite(output_filename, pred_rgb_output)
    input_img = cv2.resize(img_l, (256, 256))
    input_img -= 50  # subtract 50 for mean-centering

    print(output_filename)



def save_colorized_image(mask, input_img_path):
    print(input_img_path)
    background_image = np.array(Image.open(input_img_path))

    mask_c = np.array(Image.open(mask).resize(background_image.shape[1::-1], Image.BILINEAR))
    print(mask_c.dtype, mask_c.min(), mask_c.max())
    # uint8 0 255

    mask_c = mask_c / 255

    print(mask_c.dtype, mask_c.min(), mask_c.max())
    # float64 0.0 1.0
    mask_re = mask_c.reshape(*mask_c.shape, 1)

    dst = background_image * mask_re

    image_name_save = 'images/Object_image_colorized.png'
    print(image_name_save)

    Image.fromarray(dst.astype(np.uint8)).save(image_name_save)

    cv2.imshow('colorizedImg', 'images/Object_image_colorized.png')

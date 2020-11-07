"""
"""
import cv2
import numpy as np


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def normalize_img(origin_img, scale, mean, val):
    """Normalize image

    Args
        :origin_img: (ndarray) The given input image.
        :scale: Scale value.
        :mean: Mean value.
        :val: Val value.
    
    Returns
        :img: Normalized image.
    """
    img = np.array(origin_img.copy(), np.float32)/scale
    if len(img.shape) == 4:
        for j in range(img.shape[0]):
            for i in range(len(mean)):
                img[j,:,:,i] = (img[j,:,:,i]-mean[i])*val[i]
        return img
    else:
        for i in range(len(mean)):
            img[:,:,i] = (img[:,:,i]-mean[i])*val[i]
        return img


def resize_padding(image, out_shape, pad_value=0):
    """Resize image with padding

    Args
        :image: (ndarray) The given image.
        :out_shape: Output shape.
        :pad_value: Padding value.
    
    Returns
        :out_img: Padded image.
        :bbx:
    """
    height, width = image.shape[:2]
    ratio = float(width) / height # ratio = (width:height)
    dst_width = int(min(out_shape[1]*ratio, out_shape[0]))
    dst_height = int(min(out_shape[0]/ratio, out_shape[1]))
    origin = [int((out_shape[1] - dst_height)/2), int((out_shape[0] - dst_width)/2)]
    if len(image.shape)==3:
        image_resize = cv2.resize(image, (dst_width, dst_height))
        out_img = np.zeros(shape = (out_shape[1], out_shape[0], image.shape[2]), dtype = np.uint8) + pad_value
        out_img[origin[0]:origin[0]+dst_height, origin[1]:origin[1]+dst_width, :] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    else:
        image_resize = cv2.resize(image, (dst_width, dst_height),  interpolation = cv2.INTER_NEAREST)
        out_img = np.zeros(shape = (out_shape[1], out_shape[0]), dtype = np.uint8)
        out_img[origin[0]:origin[0]+height, origin[1]:origin[1]+width] = image_resize
        bbx = [origin[1], origin[0], origin[1]+dst_width, origin[0]+dst_height] # x1,y1,x2,y2
    return out_img, bbx


def generate_input(args, img, prior=None):
    """Generate input for prediction

    Args:
        :args: Given arguments.
        :img: (ndarray) Input image.
        :prior:
    
    Returns
        A ready-to-predict numpy array
    """
    img_norm = normalize_img(img, scale=args['img_scale'], mean=args['img_mean'], val=args['img_val'])
    
    if args['video'] == True:
        if prior is None:
            prior = np.zeros((args['input_height'], args['input_width'], 1))
            img_norm = np.c_[img_norm, prior]
        else:
            prior = prior.reshape(args['input_height'], args['input_width'], 1)
            img_norm = np.c_[img_norm, prior]
       
    img = np.transpose(img_norm, (2, 0, 1))
    return np.array(img, dtype=np.float32)
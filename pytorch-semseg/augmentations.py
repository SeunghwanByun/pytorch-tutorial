import cv2
import random
import numpy as np

# Augmentation
# Image
# - Noise : N(mean=1, std=0.05) * image
# - Gray : alpha(u(0.0, 1.0) * image + (1 - alpha) * gray
# - Brightness : (image + u(0, 32)) * u(0.5, 1.5)
# - Contrast : (image - mean(iamge)) * u(0.5, 1.5) + mean(image)
# - Color : (image - gray) * u(0.0, 1.0) + image
# - Equalization : convert to YCbCr -> Equalize Histogram of Y Channel
# - Sharpness : (image - f(image)) * u(0.0, 1.0) + image f : filter(1, 1, 1, 1, 5, 1, 1, 1, 1, ) / 13
# - Power-Law(Gamma) Transformation : gamma: u(0.8, 1.2) image = (image / 255)^gamma * 255
# - JPEG Compression Artifact : quality u(0, 100) save->load

# Random Rotation -15~+15
# Random Translation -5%~+5%
# Random Flipping 50%
# Random Rescaling -15%~+15%
# Random Gaussian Blur 50%, 5x5 Kernel sigma = 1, occlusion (50%, 16x16 Cutout)


def Rotation(img, lbl, degree):
    new_degree = random.uniform(-degree, degree)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), new_degree, 1)
    img = cv2.warpAffine(img, M, (w, h))
    lbl = cv2.warpAffine(lbl, M, (w, h))

    return img, lbl

def Scale(img, lbl):
    factor = random.uniform(0.9, 1.0)
    h, w = img.shape[:2]
    img = cv2.resize(img, ((int)(w * factor), (int)(h * factor)), interpolation=cv2.INTER_LINEAR)
    lbl = cv2.resize(lbl, ((int)(w * factor), (int)(h * factor)), interpolation=cv2.INTER_LINEAR)


    scaled_h, scaled_w = img.shape[:2]
    if factor <= 1:
        udp = (int)((h - scaled_h ) / 2)
        lrp = (int)((w - scaled_w) / 2)
        img = cv2.copyMakeBorder(img, udp, udp, lrp, lrp, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        img = img[(int)(scaled_w/2 - w/2):(int)(scaled_w/2 + w/2), (int)(scaled_h/2 - h/2):(int)(scaled_w/2 + w/2)]
        lbl = lbl[(int)(scaled_w/2 - w/2):(int)(scaled_w/2 + w/2), (int)(scaled_h/2 - h/2):(int)(scaled_w/2 + w/2)]

    return img, lbl

def Translate(img, lbl, tx, ty):
    new_tx = random.uniform(-tx, tx)
    new_ty = random.uniform(-ty, ty)

    h, w = img.shape[:2]

    M = np.float32([[1, 0, new_tx], [0, 1, new_ty]])

    img = cv2.warpAffine(img, M, (w, h))
    lbl = cv2.warpAffine(lbl, M, (w, h))

    return img, lbl

def Flip(img, lbl, p):
    if p < random.random():
        img = cv2.flip(img, 1)
        lbl = cv2.flip(lbl, 1)
        return img, lbl
    else:
        return img, lbl

def White_Noise(img):
    h, w, c = img.shape
    mean = 1
    sigma = 0.1
    gauss = np.random.normal(mean, sigma, (h, w, c))
    gauss = gauss.reshape(h, w, c)

    noisy = img * gauss
    noisy = np.clip(noisy, 0, 255.0)
    noisy = noisy.astype('uint8')

    return noisy

def Gray(img):
    alpha = random.random()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = alpha * img + (1 - alpha) * gray
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Brightness(img):
    img = img.astype('float32')
    random_brightness = random.randint(0, 32)
    random_saturation = random.uniform(0.5, 1.5)

    img = (img + random_brightness) * random_saturation
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Contrast(img):
    mean = np.mean(img)
    random_contrast = random.uniform(0.5, 1.5)

    img = (img - mean) * random_contrast + mean
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Color(img):
    img = img.astype('float32')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    img = (img - gray) * random.random() + img
    img = np.clip(img, 0, 255.0)
    img = img.astype('uint8')

    return img

def Equalization(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_cbcr = cv2.split(img)
    hist, bins = np.histogram(y_cbcr[0], 256, [0, 256])

    cdf = hist.cumsum()

    cdf_m = np.ma.masked_equal(cdf, 0)

    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    y_cbcr[0] = cdf[y_cbcr[0]]
    img = cv2.merge(y_cbcr)
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img

def Shapness(img):
    kernel_shapen = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]]) / 5.0
    k_img = cv2.filter2D(img, -1, kernel_shapen)
    img = (img - k_img) * random.random() + img
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img

def Power_Law(img):
    gamma = random.uniform(0.8, 1.2)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255.0)

    img = img.astype('uint8')

    return img
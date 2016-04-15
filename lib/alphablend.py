import numpy as np

def blend_background(image, alpha, background):
    s = image.shape
    image, alpha, background = [np.ravel(x) for x in [image, alpha, background]]
    return np.clip(np.array(image, float) * alpha + background * (1.0-alpha),0.0,255.0).reshape(s)

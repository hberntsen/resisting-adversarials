import numpy as np

def range_extend_min_max(renders, alphas):
    rr = np.ravel(renders)
    ar = np.ravel(alphas)
    sorted = np.sort(rr[ar > 0] / ar[ar > 0])
    min = float(sorted[len(sorted) * 0.15])
    max = float(sorted[len(sorted) * 0.85])
    return (min,max)

def range_extend(renders, alphas, min, max):
    alphas = alphas.reshape(renders.shape)
    rnoalpha = renders
    rnoalpha[alphas > 0] = renders[alphas > 0] / alphas[alphas > 0]
    return np.array(np.clip((rnoalpha - min) * (alphas * (255.0 / float(max - min))), 0 , 255), dtype='uint8')

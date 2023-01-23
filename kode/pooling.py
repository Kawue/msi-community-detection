import numpy as np

def mean_pooling(img, weight=None):
    return np.average(img, weight)

# Source: "Mean Deviation Similarity Index: Efficient and Reliable Full-Reference Image Quality Evaluator"
# moc: Measure of Central Tendency. This can be either a map, e.g. local mean map or a scalar, e.g. mean.
# p: p=1: Manhatten Distance, p=2: Eucledian Distance, normalized by sample size or the total weight.
# q: Integer or weighting map. Adjusts the emphasis of the values in the image
# MCT=Mean, p=2, q=1 is equal to (weighted) standard deviation
# MCT=Mean, p=1 is equal to the mean absolute deviation
# Attention: Consider that the computation of moc should incorparate the same q weighting.
def deviation_pooling(img, moc, weight=None, p=2, q=1):
    if not (isinstance(moc, int) or isinstance(moc, float)):
        if not (moc.shape == img.shape):
            raise ValueError("moc needs to be a scalar or a map with local centrality measures that equals the shape of the image!")

    if weight:
        return ((1/weight.sum()) * (np.abs(weight*(img**q - moc))**p).sum())**(1/p)
    else:
        return ((1/img.size) * (np.abs(img**q - moc)**p).sum())**(1/p)
    
    
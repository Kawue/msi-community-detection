import numpy as np
from skimage import img_as_float
from skimage.morphology import binary_opening, dilation, erosion, opening, binary_closing
from skimage.filters import try_all_threshold, threshold_otsu, threshold_isodata
from skimage.morphology import square, rectangle, diamond, disk, octagon, star

# m,n: in case only m is given its the radius, otherwise its width and height
def morphology_filter(img, selem, mode, m, n=None):
    img = img.copy()

    selem_dict = {
        "square": square,
        "rectangle": rectangle,
        "diamond": diamond,
        "disk": disk,
        "octagon": octagon,
        "star": star
    }
    s = selem_dict[selem](m, n) if n else selem_dict[selem](m)

    if mode == "open":
        return binary_opening(img, s)
    elif mode == "close":
        return binary_closing(img, s)
    else:
        raise ValueError("mode must be 'open' or 'close' for binary opening or binary closing, respectively.")


def residual_image(img1, img2):
    return np.abs(img1 - img2)


def difference_images(img1, img2):
    return img1-img2, img2-img1


# Exclude information of img2 from img1
def exclusion_image(img1, img2):
    ex = img1 - img2
    ex[ex<0] = 0
    return ex


# Minimum amount of "information" that is equal in both images
def coincidence_image(img1, img2):
    return np.amin(np.dstack((img1,img2)),axis=2)

def binarize(img, mask=None):
    # Current preference is Otsus, but other methods could be applied as well, see threshold methods from skimage.filters
    if mask:
        t = threshold_otsu(img[mask])
    else:
        t = threshold_otsu(img)
    binary = img_as_float(img > t, force_copy=True)
    return binary
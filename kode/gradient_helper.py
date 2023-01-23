import numpy as np
from skimage import filters
from scipy.ndimage import convolve

# sobel_v / sobel_h finds vertical / horizontal edges. Vertical / horizontal edges are found by the derivative of direction x / y, i.e. dx / dy.
def gradient_map(img):
    dy = filters.sobel_h(img)
    dx = filters.sobel_v(img)   
    return dy, dx


def gradient_map_diag(img):
    # Paper: Image Quality Assessment Based on Gradient Similarity
    # changed direction of one weight matrix in comparison to the original paper
    # bltr ~ dx, tlbr ~ dy, for quiver
    tlbr_weights = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
    bltr_weights = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
    tlbr = convolve(img, tlbr_weights)
    bltr = convolve(img, bltr_weights)
    return tlbr, bltr


def magnitude_map(dy, dx):
    magnitude = np.sqrt(dy**2 + dx**2)
    return magnitude


def magnitude_map_four_directions(dy, dx, tlbr, bltr):
    magnitude = np.sqrt(dy**2 + dx**2 + tlbr**2 + bltr**2)
    return magnitude


# arctan2 = arctan(x/y) but works even if y == 0
# Be careful when plotting this map. Different gradient directions, 
# e.g. 0° and 180° or 90° and 270° (each are opposite directions when horizontal axis equals 0°) might produce confusing heatmaps since 0 < 180 and 90 < 270.
# Better use a quiver plot to plot vector arrows.
def direction_map(dy, dx, degrees=True, to360=False):
    # Returns angle in radians
    direction = np.arctan2(dy, dx)
    
    if degrees:
        # Convert radians to degree default range [-180,180]
        # to360 converts [-180,180] to [0,360]
        if to360:
            return np.degrees(direction) + 180
        else:
            return np.degrees(direction)
    else:
        if to360:
            raise ValueError("to360 can only be used with degrees on True.")
        return direction
    

	
# Vectormap example: np.dstack((dx,dy))
# Gradients of (0,0) in X and Y score to 1, because both images show the same behaviour in form of a "accumulation" (aka no gradient).
# Gradients of (0,0) in one image and !(0,0) in the other image will score as 0. The behaviour is interpreted similar to orthogonality.
# If mark_homogeneous is a specified value, all pixels with (0,0) vector in either map are set to this value.
# If mark_homogeneous is True, (0,0) vectors of the first, second and both maps are inf, -inf and nan, respectively.
def dot_product_map(vectormap1, vectormap2, mark_homogeneous=False):
    if vectormap1.shape != vectormap2.shape:
        raise ValueError("Vectormaps must be of same shape!")

    if vectormap1.ndim != 3:
        raise ValueError("Vectormaps have to be three dimensional. Example: np.dstack((dx,dy)), with dx and dy beeing gradient maps.")

    # Norm vectors to unit length so that the dot product is highest if both vectors have the same direction.
    normed1 = np.sqrt((vectormap1 * vectormap1).sum(axis=2))
    vectormap1 = vectormap1 / normed1[:,:,None]
    vectormap1[np.isnan(vectormap1)] = 0

    normed2 = np.sqrt((vectormap2 * vectormap2).sum(axis=2))
    vectormap2 = vectormap2 / normed2[:,:,None]
    vectormap2[np.isnan(vectormap2)] = 0

    if mark_homogeneous:
        hr_map1 = (vectormap1[:,:,0] == 0) * (vectormap1[:,:,1] == 0)
        idx_hr1 = np.where(hr_map1)
        hr_map2 = (vectormap2[:,:,0] == 0) * (vectormap2[:,:,1] == 0)
        idx_hr2 = np.where(hr_map2)
        idx_hr12 = np.where(hr_map1 * hr_map2)
        # To solve numerical instabilities, most likely due to rounding.
        dpm = np.around((vectormap1*vectormap2).sum(axis=2), 5)
        if type(mark_homogeneous) == bool:
            if mark_homogeneous:
                dpm[idx_hr1] = np.inf
                dpm[idx_hr2] = -np.inf
                dpm[idx_hr12] = np.nan
        else:
            dpm[idx_hr1] = mark_homogeneous
            dpm[idx_hr2] = mark_homogeneous
            dpm[idx_hr12] = mark_homogeneous
    else:
        # To solve numerical instabilities, most likely due to rounding.
        dpm = np.around((vectormap1*vectormap2).sum(axis=2), 5)    
    return dpm


# Paper: Image Quality Assessment Based on Gradient Similarity
# Excluded mean operator in comparison to the original paper, introduces squares to include opposed directions
# Here there might be the addition of other rotational variants needed
def max_grad_map(img):
    dy, dx = gradient_map(img)
    bltr, tlbr = gradient_map_diag(img)
    return np.max(np.array([dy**2, dx**2, bltr**2, tlbr**2]), axis=0)
import copy

from .common import Image, ImageException
from .processing import ImageProcess, ImageIO
import numpy as np
import cv2 as cv
from sklearn.mixture import GaussianMixture
from skimage import segmentation, color
import skimage.feature as skfeature
from skimage.future import graph
from scipy import ndimage
import skimage.morphology as skmorph
from typing import Optional
from skimage.filters.rank import entropy
from skimage.morphology import disk
from pathlib import Path
from typing import Union
from scipy.ndimage import maximum_filter
import imutils
import imutils.object_detection

"""
General algorithms
"""


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gmmMeanFrequencies(gmm: GaussianMixture):
    freqs = []
    for mean in gmm.means_:
        freqs.append(np.exp(gmm.score_samples(np.array(mean).reshape(-1, 1))))
    return np.array(freqs).ravel()


def weightMeanSort(gmm: GaussianMixture):
    return gmm.means_[np.argsort(gmm.weights_.ravel()), :]


def weightCovarSort(gmm: GaussianMixture):
    return gmm.covariances_[np.argsort(gmm.weights_.ravel()), :]


def cannyThresh(image, sigma=0.33):
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 - sigma) * median))
    return lower, upper


def image_auto_crop(image: np.ndarray, tolerance: int = 5, out_path: Union[Path, str] = None):
    """
    Function to auto-crop an image based on a white background

    :param image: input image as ndarray
    :param out_path: if set, image is saved to the path. should include filename and extension
    :param tolerance: white level tolerance
    :return: the cropped image
    """

    white_level = 255 - tolerance
    temp = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    temp = maximum_filter(temp, 7)
    h_mean = np.mean(temp, axis=0)
    v_mean = np.mean(temp, axis=1)
    udlr = []

    for i in range(len(v_mean)):
        if v_mean[i] < white_level:
            udlr.append(i)
            break

    for i in reversed(range(len(v_mean))):
        if v_mean[i] < white_level:
            udlr.append(i)
            break

    for i in range(len(h_mean)):
        if h_mean[i] < white_level:
            udlr.append(i)
            break

    for i in reversed(range(len(h_mean))):
        if h_mean[i] < white_level:
            udlr.append(i)
            break

    crop = image[udlr[0]:udlr[1], udlr[2]:udlr[3]]

    if out_path is not None:
        cv.imwrite(f'{out_path}', crop)

    return crop


def labels2rgb(labels):
    """
    Convert a label image to an rgb image using a lookup table
    :Parameters:
      labels : an image of type np.uint8 2D array
      lut : a lookup table of shape (256, 3) and type np.uint8
    :Returns:
      colorized_labels : a colorized label image
    """

    def tobits(x, o):
        return np.array(list(np.binary_repr(x, 24)[o::-3]), np.uint8)

    def gen_lut():
        """
        Generate a label colormap compatible with opencv lookup table, based on
        Rick Szelski algorithm in `Computer Vision: Algorithms and Applications`,
        appendix C2 `Pseudocolor Generation`.
        :Returns:
          color_lut : opencv compatible color lookup table
        """
        arr = np.arange(256)
        r = np.concatenate([np.packbits(tobits(x, -3)) for x in arr])
        g = np.concatenate([np.packbits(tobits(x, -2)) for x in arr])
        b = np.concatenate([np.packbits(tobits(x, -1)) for x in arr])
        return np.concatenate([[[b]], [[g]], [[r]]]).T

    return cv.LUT(cv.merge((labels, labels, labels)).astype(np.uint8), gen_lut())


def gmmBackroundMask(
    image: Optional[Image] = None,
    resizeImage: tuple = None,
    nComponents=5,
    verbose=False,
    plotResults=False,
    name: str = None
    ) -> ImageProcess:
    """
    Creates a background mask using a Gaussian Mixture model.  Currently assumes 3 components

    :param name:
    :param image:
    :param resizeImage: a tuple of scaling factors (xdir, ydir) between 0 and 1
    :param nComponents:
    :param verbose:
    :param plotResults:
    """
    n = name if name is not None else "GMM Background Subtraction"
    process = ImageIO(_func=gmmBackroundMask, _setParams=locals()).initProcess(n)

    if resizeImage is not None:
        def resize(io: ImageIO):
            _resizeImage = process.input['resizeImage']
            out = cv.resize(io.image,
                            (0, 0),  # set fx and fy, not the final size
                            fx=_resizeImage[0],
                            fy=_resizeImage[1],
                            interpolation=cv.INTER_AREA)
            return ImageIO(image=out)

        process.addFunction(resize)
    else:
        img = np.array(process.image)

    # Remove most common color (background)
    def create_mask(_image: Image) -> ImageIO:
        # Fit GMM
        hsvIm = cv.cvtColor(_image, cv.COLOR_RGB2HSV)
        # data = np.array(hsv[:, :, 0]).ravel()  # hsv.reshape(hsv.shape[0] * hsv.shape[1], 3)  # h[s>0].ravel()

        _gmm = GaussianMixture(n_components=nComponents, verbose_interval=1, verbose=verbose)
        _gmm = _gmm.fit(X=hsvIm.reshape(hsvIm.shape[0] * hsvIm.shape[1], 3))

        def color_bounds_from(_gmm):
            __maxColor = weightMeanSort(_gmm)[-1, :]
            __maxVar = weightCovarSort(_gmm)[-1, :]
            __maxVar = np.diag(__maxVar)
            __pm1 = max(np.sqrt(__maxVar[0]), 15)
            __lower_h = np.array([__maxColor[0] - __pm1, 0, 0])
            __upper_h = np.array([__maxColor[0] + __pm1, 255, 255])
            __lower_h[__lower_h < 0] = 0
            __upper_h[__upper_h > 255] = 255
            return __lower_h, __upper_h, __maxColor, __maxVar

        _lower_h, _upper_h, _maxColor, _maxVar = color_bounds_from(_gmm)
        maxHueMask = cv.inRange(hsvIm, _lower_h, _upper_h)
        maxHueData = cv.bitwise_and(hsvIm, hsvIm, mask=maxHueMask)
        _idxs = np.argwhere(maxHueMask)

        maxHueData = maxHueData[_idxs[:, 0], _idxs[:, 1], :]
        maxHueGmm = GaussianMixture(n_components=nComponents, verbose_interval=1, verbose=0)
        maxHueGmm.fit(maxHueData)

        h_lower_h, h_upper_h, h_maxColor, h_maxVar = color_bounds_from(maxHueGmm)
        hsv_im = cv.cvtColor(process.image, cv.COLOR_RGB2HSV)
        # cv.inRange(cv.cvtColor(np_image, cv.COLOR_RGB2HSV), _lower_h, _upper_h)
        _mask = cv.bitwise_not(cv.inRange(hsv_im, _lower_h, _upper_h))
        return ImageIO(image=_mask,
                       gmm=maxHueGmm,
                       maxColor=_maxColor,
                       maxVar=_maxVar,
                       lower_h=_lower_h,
                       upper_h=_upper_h)

    process.addFunction(create_mask)

    # morphops
    process.addProcess(skeleton())

    # median filter mask
    process.addProcess(medianBlurProcess(ksize=9))

    # Perform Mask
    def mask_image(_image, _mask=np.ndarray) -> ImageIO:
        masked_image = cv.bitwise_and(_image, _image, mask=_mask)
        return ImageIO(image=masked_image)

    process.addFunction(mask_image)

    return process


# Canny Detection
def canny(_image) -> ImageIO:
    canny_params = cannyThresh(_image)
    res = cv.Canny(_image, *canny_params)
    return ImageIO(image=res, canny_params=canny_params)


def cannyProcess(image: Image, blurSize=5) -> ImageProcess:
    process = ImageIO(_func=cannyProcess, _setParams=locals()).initProcess("Canny")

    try:

        # Blurring
        def blur(_image: Image, _blurSize=blurSize):
            _img = _image.asGrayScale()
            res = cv.medianBlur(_img, ksize=_blurSize)
            return ImageIO(image=res)

        process.addFunction(blur)
        process.addFunction(canny)

    except ImageException:
        return process

    return process


# skeleton
def skeleton(image: Optional[Image] = None) -> ImageProcess:
    proc = ImageIO(_func=skeleton, _setParams=locals()).initProcess("Skeletonization")

    def _skeleton(_image):
        size = np.size(_image)
        el = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        skel = np.zeros(_image.shape, np.uint8)
        done = False
        while not done:
            eroded = cv.erode(_image, el)
            temp = cv.dilate(eroded, el)
            temp = cv.subtract(_image, temp)
            skel = cv.bitwise_or(skel, temp)
            _image = eroded.copy()
            zeros = size - cv.countNonZero(_image)
            done = zeros == size

        return ImageIO(image=skel)

    proc.addFunction(_skeleton)

    return proc


# Blurring
def medianBlurProcess(image: Optional[Image] = None, ksize=5):
    proc = ImageIO(_func=medianBlurProcess, _setParams=locals()).initProcess("Median Blur")

    def median_blur(_image: Image, _ksize=ksize) -> ImageIO:
        return ImageIO(image=cv.medianBlur(_image, ksize=_ksize))

    proc.addFunction(median_blur)
    return proc


def bilateral_filter(_image, _d=-1, _sigmaColor=41, _sigmaSpace=31):
    return ImageIO(image=cv.bilateralFilter(_image, d=_d, sigmaColor=_sigmaColor, sigmaSpace=_sigmaSpace))


# Bilateral filter
def bilateralFilter(image: Optional[Image] = None, d=-1, sigmaColor=41, sigmaSpace=31):
    proc = ImageIO(_func=bilateralFilter, _setParams=locals()).initProcess("Bilateral Filter")
    proc.addFunction(bilateral_filter)
    return proc


# Morphology
def morphologyExProcess(morph_op, image: Image = None, ksize=5, iterations=1) -> ImageProcess:
    io = ImageIO(_func=morphologyExProcess, _setParams=locals())

    morph_type = None
    if morph_op == cv.MORPH_OPEN:
        morph_type = 'open'
    elif morph_op == cv.MORPH_CLOSE:
        morph_type = 'close'
    elif morph_op == cv.MORPH_DILATE:
        morph_type = 'dilate'
    elif morph_op == cv.MORPH_ERODE:
        morph_type = 'erode'
    elif morph_op == cv.MORPH_BLACKHAT:
        morph_type = 'blackhat'

    # io['morph_type'] = morph_type
    proc = io.initProcess(f"Morph-{morph_type}")

    def morph(_image, _ksize=ksize, _iterations=iterations) -> ImageIO:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (_ksize, _ksize))
        out = cv.morphologyEx(_image, morph_op, kernel, iterations=_iterations)
        return ImageIO(image=out)

    proc.addFunction(morph, f'morph_{morph_type}')

    return proc


# Edge Detection
def spatialGradient(_image: Image) -> ImageIO:
    def _grad(im):
        dx, dy = cv.spatialGradient(im, cv.CV_64F)
        dx, dy = dx.astype(np.float), dy.astype(np.float)
        mag = np.sqrt((dx * dx) + (dy * dy))
        return (mag / mag.max() * 255.).astype(np.uint8)

    if _image.numChannels == 3:
        r, g, b = _image.rgb
        out = np.zeros(_image.shape).astype(np.uint8)
        for i, channel in enumerate([r, g, b]):
            out[:, :, i] = _grad(channel)
    else:
        out = _grad(_image)
    return ImageIO(image=out)


def laplacianGradient(_image: Image) -> ImageIO:
    r, g, b = _image.rgb
    rout = cv.Laplacian(r, cv.CV_64F)
    gout = cv.Laplacian(g, cv.CV_64F)
    bout = cv.Laplacian(b, cv.CV_64F)
    out = rout + gout + bout
    out = out / np.max(out) * 255.0
    return ImageIO(image=out.astype('uint8'))


def edgeDetection(image: Optional[Image] = None, sigma=0.33, mode='canny', name=None) -> ImageProcess:
    n = "Edge Detection" if name is None else name
    proc = ImageIO(_func=edgeDetection, _setParams=locals()).initProcess(n)

    proc.addProcess(medianBlurProcess(ksize=9))

    if mode == 'gradient':

        proc.addFunction(laplacianGradient)

        proc.addProcess(otsuThresholdProcess())

    else:
        def autoCanny(_image: Image) -> ImageIO:

            def applyCanny(_im):
                # compute the median of the single channel pixel intensities
                intensity = np.median(_im)
                # apply automatic Canny edge detection using the computed median
                lower = int(max(0, (1.0 - sigma) * intensity))
                upper = int(min(255, (1.0 + sigma) * intensity))
                out = cv.Canny(_im, lower, upper)
                return out

            if len(_image.shape) > 2:
                r, g, b = _image.rgb
                h, s, v = _image.hsv
                rEdged = applyCanny(r)
                gEdged = applyCanny(g)
                bEdged = applyCanny(b)
                hEdged = applyCanny(h)
                sEdged = applyCanny(s)
                vEdged = applyCanny(v)
                edged = cv.bitwise_or(rEdged, gEdged)
                edged = cv.bitwise_or(edged, bEdged)
                hEdged = cv.bitwise_or(hEdged, sEdged)
                hEdged = cv.bitwise_or(hEdged, vEdged)
                edged = cv.bitwise_and(edged, hEdged)
            else:
                edged = applyCanny(_image)
            return ImageIO(image=edged)

        proc.addFunction(autoCanny)
        # proc.addProcess(morphologyExProcess(cv.MORPH_DILATE, ksize=3))

    # Flood Fill
    def floodFill(_image):
        # Copy the thresholded image.
        im_floodfill = _image.copy()

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = _image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv.floodFill(im_floodfill, mask, (0, 0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = _image | im_floodfill_inv

        # Display images.
        # cv.imshow("Thresholded Image", binaryImage)
        # cv.imshow("Floodfilled Image", im_floodfill)
        # cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        # cv.imshow("Foreground", im_out)
        # cv.waitKey(0)
        return ImageIO(image=im_out)

    proc.addFunction(floodFill)

    # cleanup
    proc.addProcess(morphologyExProcess(cv.MORPH_OPEN, ksize=3))

    return proc


def backgroundDistanceMask(ksize=315, name=None) -> ImageProcess:
    n = name if name is not None else "background distance masking"
    proc = ImageIO(_func=backgroundDistanceMask, _setParams=locals()).initProcess(n)

    def blurDiff(_image: Image):
        im1 = _image
        im2 = cv.medianBlur(im1, ksize=ksize)
        dIm = np.sum(np.square(np.subtract(im1.astype(np.float), im2.astype(np.float))), axis=2)
        return ImageIO(image=dIm.astype(np.uint8))

    proc.addFunction(blurDiff)

    return proc


def binarizeImageProcess(image: Optional[Image] = None):
    proc = ImageIO(_func=binarizeImageProcess, _setParams=locals()).initProcess("Binarize Image")

    def binarize(_image: Image):
        if len(_image.shape) > 2:
            r, g, b = _image.rgb
            dst = cv.bitwise_or(r, g)
            dst = cv.bitwise_or(dst, b)
        else:
            dst = _image
        return ImageIO(image=dst)

    proc.addFunction(binarize)

    return proc


# OTSU Thresholding

def otsu_threshold(_image: Image, _binaryOutput=False, invert=False) -> ImageIO:
    if _image.ndim > 2:  # Color image
        r_thresh_val, r = cv.threshold(_image[:, :, 0].astype(np.uint8), 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        g_thresh_val, g = cv.threshold(_image[:, :, 1].astype(np.uint8), 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        b_thresh_val, b = cv.threshold(_image[:, :, 2].astype(np.uint8), 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        dst = np.zeros(_image.shape).astype('uint8')
        dst[:, :, 0] = r.astype('uint8')
        dst[:, :, 1] = g.astype('uint8')
        dst[:, :, 2] = b.astype('uint8')
        grayThresh = np.mean(np.dstack([r_thresh_val, g_thresh_val, b_thresh_val]), 2).astype('uint8')
        ret = ImageIO(image=dst, r_thresh_val=r_thresh_val, g_thresh_val=g_thresh_val, b_thresh_val=b_thresh_val,
                      thresh_val=grayThresh)
    else:  # GrayScale
        thresh_val, dst = cv.threshold(_image.astype(np.uint8), 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY)
        ret = ImageIO(image=dst, thresh_val=thresh_val)

    if invert:
        dst = 255 - dst

    if _binaryOutput:
        ret['image'] = Image(dst).asGrayScale() > 127

    return ret


def otsuThresholdProcess(name: str = None, binaryOutput=False) -> ImageProcess:
    name = "Otsu's Thresholding" if name is None else name
    proc = ImageIO(_func=otsuThresholdProcess, _setParams=locals()).initProcess(name)
    proc.addFunction(otsu_threshold)
    return proc


# Adaptive Thresholding

def adaptiveThresholdProcess(
    blockSize: tuple = (15, 15), center: int = 0,
    smoothingKernelSize=5, name=None
    ) -> ImageProcess:
    n = "adaptive thresholding" if name is None else name
    proc = ImageIO(_func=adaptiveThresholdProcess, _setParams=locals()).initProcess(n)

    def adap_thresh(_image, _blockSize=blockSize, _center=center):
        if len(_image.shape) > 2:
            r = cv.adaptiveThreshold(_image[:, :, 0], 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                     *_blockSize, _center)
            g = cv.adaptiveThreshold(_image[:, :, 1], 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                     *_blockSize, _center)
            b = cv.adaptiveThreshold(_image[:, :, 2], 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                     *_blockSize, _center)
            out = np.zeros(_image.shape).astype('uint8')
            out[:, :, 0] = r.astype('uint8')
            out[:, :, 1] = g.astype('uint8')
            out[:, :, 2] = b.astype('uint8')
        else:
            out = cv.adaptiveThreshold(_image.asGrayScale(), 255.0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                       *_blockSize, _center)
        return ImageIO(image=out)

    proc.addFunction(adap_thresh)

    proc.addProcess(medianBlurProcess(ksize=smoothingKernelSize))

    return proc


def entropySegmentProcess(ksize=5, name=None) -> ImageProcess:
    n = name if name is not None else 'Entropy Segment'
    proc = ImageIO(_func=entropySegmentProcess, _setParams=locals()).initProcess(n)

    def _calcEntropy(_image, _ksize=ksize):
        img = _image
        disksize = _ksize
        if len(img.shape) > 2:
            r = entropy(img[:, :, 0], disk(disksize))
            g = entropy(img[:, :, 1], disk(disksize))
            b = entropy(img[:, :, 2], disk(disksize))
            out = np.array(img)
            out[:, :, 0] = r
            out[:, :, 1] = g
            out[:, :, 2] = b
        else:
            out = entropy(img, disk(disksize))
        return ImageIO(image=out)

    proc.addFunction(_calcEntropy)
    proc.addProcess(adaptiveThresholdProcess(blockSize=(7, 7)))
    return proc


# Mean shift filter
def meanShiftFilter(shiftSpatialWindow=41, shiftColorRadius=55, name: str = None):
    n = name if name is not None else "Mean Shift Filter"
    proc = ImageIO(_func=meanShiftFilter, _setParams=locals()).initProcess(n)

    def mean_shift_filter(_image: Image, _shiftSpatialWindow=shiftSpatialWindow, _shiftColorRadius=shiftColorRadius):
        _img = _image
        shifted = cv.pyrMeanShiftFiltering(_img, _shiftSpatialWindow, _shiftColorRadius)
        return ImageIO(image=shifted)

    proc.addFunction(mean_shift_filter)
    return proc


def calcEuclideanDistance(name: str = None):
    """
    compute the exact Euclidean distance from every binary
    pixel to the nearest zero pixel, then find peaks in this
    distance map
    """
    n = "Calculate Markers" if name is None else name
    proc = ImageIO(_func=calcEuclideanDistance, _setParams=locals()).initProcess(n)

    def _calcMarkersFromOtsu(_image: Image):
        thresh = _image
        D = ndimage.distance_transform_edt(thresh)
        localMax = skfeature.peak_local_max(D, indices=False, min_distance=20, labels=thresh)

        # perform a connected component analysis on the local peaks,
        # using 8-connectivity, then appy the Watershed algorithm
        _markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        heatmap = cv.applyColorMap(cv.normalize(D.astype(np.uint8), None, 255, 0, cv.NORM_MINMAX),
                                   cv.COLORMAP_JET)
        return ImageIO(image=D, heatmap=heatmap, localMax=localMax, markers=_markers, mask=thresh, display='heatmap')

    proc.addFunction(_calcMarkersFromOtsu)

    return proc


def watershed(mask: Optional[np.ndarray] = None, name: str = None):
    n = "Watershed" if name is None else name
    proc = ImageIO(_func=watershed, _setParams=locals()).initProcess(n)

    # Actual opencv watershed algorithm
    def _watershed(_image, _mask, _markers):
        if _mask is None:
            labels = skmorph.watershed(-_image, _markers)
        else:
            labels = skmorph.watershed(-_image, _markers, mask=_mask)
        out = np.uint8(labels)
        return ImageIO(rgbLabels=labels2rgb(labels), labels=labels, image=out, display='rgbLabels')

    proc.addFunction(_watershed)

    return proc


# Watershed
def watershedProcess(
    image: Optional[Image] = None,
    shiftSpatialWindow=41,
    shiftColorRadius=55,
    numFindColors: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    name=None
    ) -> ImageProcess:
    n = name if name is not None else "Watershed"
    proc = ImageIO(_func=watershedProcess, _setParams=locals()).initProcess(n)

    try:
        # Board color removal
        # h, s, v = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        # hist, edges = np.histogram(h.flatten(), bins=180)
        # mostFreqColor = np.argmax(hist)

        # Mean shift filter
        proc.addProcess(meanShiftFilter(shiftColorRadius=shiftColorRadius, shiftSpatialWindow=shiftSpatialWindow))
        # proc.addProcess(medianBlurProcess(ksize=ksize))

        # if numFindColors is not None:
        #     def _calcMarkersColor(_image, _numFindColors=numFindColors):
        #         uniqueColors, colorCounts = Image.uniqueColors(_image)
        #         maxCount = np.max(colorCounts)
        #         colors = uniqueColors[(colorCounts > _numFindColors) & (colorCounts != maxCount)]
        #         counts = colorCounts[(colorCounts > _numFindColors) & (colorCounts != maxCount)]
        #         mask = np.zeros(_image.shape[:-1]).astype(np.uint8)
        #         for c in colors:
        #             cImg = cv.inRange(_image, c - 1, c + 1)
        #             mask = cv.bitwise_or(mask, cImg)
        #         D = ndimage.distance_transform_edt(mask)
        #         localMax = skfeature.peak_local_max(D, indices=False, min_distance=20, labels=mask)
        #         _markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        #         heatmap = cv.applyColorMap(cv.normalize(D.astype(np.uint8), None, 255, 0, cv.NORM_MINMAX),
        #                                    cv.COLORMAP_JET)
        #         return ImageIO(image=heatmap, D=D, localMax=localMax, markers=_markers, threshImage=mask)
        #
        #     proc.addFunction(_calcMarkersColor)
        #
        # else:

        # Otsu Thresholding
        proc.addProcess(otsuThresholdProcess())

        # Binarize the 3 channels
        proc.addProcess(binarizeImageProcess())

        # Calculate the markers
        proc.addProcess(calcEuclideanDistance())

        # Calculate Watershed
        proc.addProcess(watershed())

    except ImageException as e:
        print(f"Error: {e}")

    return proc


def graphCutSegmentation(
    image: Optional[Image] = None,
    compactness=10.,
    numSegs=None,
    mode="similarity", name=None
    ) -> ImageProcess:
    numSegs = numSegs if numSegs is not None else int(np.median(image.shape) / 4.0)
    n = name if name is not None else "GraphCut"
    proc = ImageIO(_func=graphCutSegmentation, _setParams=locals()).initProcess(n)

    def normalized_cut_1(_image: Image, _compactness=compactness, _numSegs=numSegs):
        data = _image
        labels = segmentation.slic(data, compactness=_compactness, n_segments=_numSegs,
                                   start_label=1)
        out = color.label2rgb(labels, data, kind='avg', bg_label=0)
        return ImageIO(image=out, labels=labels)

    proc.addFunction(normalized_cut_1)

    def normalized_cut_2(_image, _labels, _mode=mode, sigma=255.0):
        data = _image
        g = graph.rag_mean_color(data, labels=_labels, mode=_mode, sigma=sigma)
        labels = graph.cut_normalized(_labels, g)
        out = color.label2rgb(labels, data, kind='avg', bg_label=0)
        return ImageIO(image=out, labels=labels)

    proc.addFunction(normalized_cut_2)

    return proc


def label2rgb(labels: np.ndarray = None) -> ImageProcess:
    proc = ImageIO(_func=label2rgb, _setParams=locals()).initProcess("Label 2 RGB")

    def label_2_rgb(_image, _labels):
        out = color.label2rgb(_labels, _image, kind='avg')
        return ImageIO(image=out, labels=_labels)

    proc.addFunction(label_2_rgb)

    return proc


def efficientGraphSegmentation(
    image: Optional[Image] = None,
    scale=200.,
    sigma=0.8,
    min_size=400,
    name=None
    ) -> ImageProcess:
    """ Based on 'Efficient Graph Based Image Segmentation'
        Graph segmentation help
        https://www.youtube.com/watch?v=iDKeR_swA8g
    """
    name = name if name is not None else "efficient graphcut"
    proc = ImageIO(_func=efficientGraphSegmentation, _setParams=locals()).initProcess(name)

    def felzenszwalb(_image, _sigma=sigma, _min_size=min_size, _scale=scale):
        labels = segmentation.felzenszwalb(_image, scale=_scale, sigma=_sigma, min_size=_min_size)
        return ImageIO(image=_image, labels=labels, display='labels')

    proc.addFunction(felzenszwalb)
    proc.addProcess(label2rgb())

    return proc


def histogramEqualization(_image: Image):
    if _image.ndim >= 3:
        ycrcb = cv.cvtColor(_image, cv.COLOR_RGB2YCrCb)
        channels = cv.split(ycrcb)
        channels[0] = cv.equalizeHist(channels[0])
        ycrcb = cv.merge(channels)
        return ImageIO(image=cv.cvtColor(ycrcb, cv.COLOR_YCrCb2RGB))
    return ImageIO(image=cv.equalizeHist(_image))


def find_text_east_detector(_image: Image, targetImageSize=320, min_confidence=0.05):
    if targetImageSize % 32 != 0:  # EAST detector requires image to be sized multiples of 32
        targetImageSize = 320

    im = Image(cv.resize(_image, (targetImageSize, targetImageSize)))

    rW = _image.width / float(im.width)
    rH = _image.height / float(im.height)

    layerNames = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    net = cv.dnn.readNet('./models/frozen_east_text_detection.pb')

    if im.numChannels < 3:
        im = Image(color.gray2rgb(im))

    m = im.channelMeans()

    blob = cv.dnn.blobFromImage(im, 1.0, (im.width, im.height), mean=m, swapRB=False, crop=False)

    net.setInput(blob)

    scores, geometry = net.forward(layerNames)

    (nRows, nCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(nRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(nCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # boxes = cv.dnn.NMSBoxes(np.array(rects), confidences, min_confidence, 0.5)
    boxes = imutils.object_detection.non_max_suppression(np.array(rects), confidences)

    boxIm = _image.copy()
    if len(boxIm.shape) < 3:
        boxIm = color.gray2rgb(boxIm).astype(np.uint8)

    for startX, startY, endX, endY in boxes:
        cv.rectangle(boxIm,
                     (int(startX * rW), int(startY * rH)),
                     (int(endX * rW), int(endY * rH)),
                     (0, 255, 0), 2)

    return ImageIO(image=_image, boxIm=boxIm, boxes=boxes, display='boxIm')


def cast_ray(gx, gy, edges, row, col, direction, max_angle_diff):
    """Casts a ray in an image given a starting point, an edge set, and the gradient
    Applies the SWT algorithm steps and outputs bounding boxes.

    Keyword Arguments:

    gx -- verticle component of the gradient
    gy -- horizontal component of the gradient
    edges -- the edge set of the image
    row -- the starting row location in the image
    col -- the starting column location in the image
    dir -- either 1 (light text) or -1 (dark text), the direction the ray should be cast
    max_angle_diff -- Controls how far from directly opposite the two edge gradeints should be
    """

    i = 1
    ray = [[row, col]]
    # Getting origin gradients
    g_row = gx[row, col] * direction
    g_col = gy[row, col] * direction

    # If we encounter an edge with no direction
    if g_row == 0 and g_col == 0:
        return None

    # Normalizing g_col and g_row to ensure we move ahead one pixel
    g_col_norm = g_col / magnitude(g_col, g_row)
    g_row_norm = g_row / magnitude(g_col, g_row)

    # TODO: Cap ray size based off of ratio?
    while True:
        # Calculating the next step ahead in the ray
        # Adding 0.5 to start in center of pixel
        col_step = int(np.floor(col + 0.5 + g_col_norm * i))
        row_step = int(np.floor(row + 0.5 + g_row_norm * i))
        i += 1
        try:
            # Checking if the next step is an edge
            if edges[row_step, col_step] > 0:
                # Checking that edge pixels gradient is approximately opposite the direction of travel
                g_opp_row = gx[row_step, col_step] * direction
                g_opp_col = gy[row_step, col_step] * direction
                theta = angle_between(g_row_norm, g_col_norm, -g_opp_row, -g_opp_col)

                if theta < max_angle_diff:
                    g_opp_row = g_opp_row / magnitude(g_opp_row, g_opp_col)
                    g_opp_col = g_opp_col / magnitude(g_opp_row, g_opp_col)
                    # print("Start Gradient: " + str(g_row_norm) + ", " + str(g_col_norm))
                    # print("End Gradient: " + str(-g_opp_row) + ", " + str(-g_opp_col))
                    return ray
                else:
                    return None
            else:
                ray.append([row_step, col_step])
        except IndexError:
            return None


def magnitude(x, y):
    return np.sqrt(x * x + y * y)


def dot(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


# assumes neither vector is zero
def angle_between(x1, y1, x2, y2):
    proportion = dot(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2))
    if abs(proportion) > 1:
        return np.pi / 2
    else:
        return np.arccos(dot(x1, y1, x2, y2) / (magnitude(x1, y1) * magnitude(x2, y2)))


def median_ray(ray, swt_img):
    # Accumulate pixel values and calculate median
    pixel_values = []
    for coordinate in ray:
        pixel_values.append(swt_img[coordinate[0], coordinate[1]])
    return np.median(pixel_values)


def stroke_width_transform(_image, _gradient_direction) -> ImageIO:
    edges = canny(_image).image
    dx, dy = cv.spatialGradient(_image.asGrayScale(), cv.CV_64F)
    # theta = np.arctan2(dy, dx)

    # Setting up SWT image
    swt_img = np.empty(edges.shape)
    swt_img[:] = np.Infinity  # Setting all values to infinite

    rays = []
    # Looping through each pixel, calculating rays
    for row in range(_image.shape[0]):
        for col in range(_image.shape[1]):
            edge = edges[row, col]
            if edge > 0:  # Checking if we're on an edge
                # Passing in single derivative values for rows and cols
                # Along with edges and ray origin
                ray = cast_ray(dx, dy, edges, row, col, _gradient_direction, np.pi / 2)
                if ray is not None:
                    # Adding ray to rays accumulator
                    rays.append(ray)
                    # Calculating the width of the ray
                    width = magnitude(ray[len(ray) - 1][0] - ray[0][0], ray[len(ray) - 1][1] - ray[0][1])
                    # Assigning width to each pixel in the ray
                    for point in ray:
                        if swt_img[point[0], point[1]] > width:
                            swt_img[point[0], point[1]] = width

    # Set values of infinity to zero so that only values that had ray > 0
    for row in range(swt_img.shape[0]):
        for col in range(swt_img.shape[1]):
            if swt_img[row, col] == np.Infinity:
                swt_img[row, col] = 0

    # Creating a copy of the SWT image
    swt_median = copy.deepcopy(swt_img)

    # Looping through rays and assigning the median value
    # to ray pixels that are above the median
    for ray in rays:
        # Getting median of each ray's values
        median = median_ray(ray, swt_img)

        # Loop through ray and change pixel values greater than median
        for coordinate in ray:
            if swt_img[coordinate[0], coordinate[1]] > median:
                swt_median[coordinate[0], coordinate[1]] = median

    return ImageIO(image=swt_median)


def strokeWidthTransform(image: Optional[Image] = None, name=None):
    name = name if name is not None else "stroke-width-transform"

    proc = ImageIO(_func=strokeWidthTransform, _setParams=locals()).initProcess(name)

    def concatSwt(_image: Image):
        # light text
        light = stroke_width_transform(_image, 1).image

        # dark text
        dark = stroke_width_transform(_image, -1).image

        return ImageIO(image=_image, light=light, dark=dark, display=['light', 'dark'])

    proc.addFunction(concatSwt)

    return proc

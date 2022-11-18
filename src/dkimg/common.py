from __future__ import annotations
from __future__ import unicode_literals
from __future__ import division
import cv2 as cv
from typing import Union, List, Optional, NamedTuple, Iterable
from enum import Enum
import numpy as np
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sb
import skimage.measure as skmeasure
from pathlib import Path
import pandas as pd
import ast
import itertools
import time
import threading
import sklearn.cluster as skcluster
import collections
import sys

jsonpickle_numpy.register_handlers()

Pathlike = Union[Path, str]


# def maximizePlot():
#
#     backend = matplotlib.get_backend()
#     manager = plt.get_current_fig_manager()
#
#     if 'Qt' in backend:
#         # Option 1
#         # QT backend
#         manager.window.showMaximized()
#     elif 'Tk' in backend:
#         # Option 2
#         # TkAgg backend
#         # manager.resize(*manager.window.maxsize())
#         manager.window.state('zoomed')
#     elif 'WX' in backend:
#         # Option 3
#         # WX backend
#         manager.frame.Maximize(True)
#     elif 'OSX' in backend:
#         manager.resize(width, height)


class Undefined:
    pass


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


class ImageException(Exception):
    pass


class ColorSpace(Enum):
    """
    ColorsSpace enumeration
    """
    rgb = "RGB"  #: Red Green Blue
    rgba = "RGBA"  #: Red Green Blue Alpha
    hsv = "HSV"  #: Hue Saturation Value / Intensity
    cym = "CYM"  #: Usually for print
    luv = "LUV"  #: CIE(L* u* v*)
    lab = "LAB"  #: CIE(L* a* b*)
    bgr = "BGR"  #: Blue Green Red (OpenCV Default)
    bgra = "BGRA"  #: Blue Green Red Alpha
    hls = "HLS"  #: Hue Lightness Saturation
    gray = "GRAY"  #:
    yuv = "YUV"  #: TV color representation (europe)
    ycr = "YCR_CB"  #:
    xyz = "XYZ"  #: CIE(XYZ)

    # See https://paperpile.com/shared/VCkEnN for more
    # Cheng, H. D., Jiang, X. H., Sun, Y., & Wang, J. (2001). Color image segmentation: advances and prospects. Pattern Recognition, 34(12), 2259–2281.

    # TODO: Non-OpenCV conversions
    #   RGB <--> YIQ


# -------------------------------------------------
# Utility Classes
# -------------------------------------------------

class Point(np.ndarray):

    def __new__(cls, x, y, **kwargs):
        vals = np.array([x, y])
        arr = np.asarray([x, y], **kwargs).view(cls)
        return arr

    def __array_finalize__(self, obj):
        if self.shape != (2,):
            ImageException(f"Invalid Point shape : {self.shape}")

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def col(self):
        return self.x

    @property
    def row(self):
        return self.y


class BoundingBox(NamedTuple):
    minCol: int
    minRow: int
    maxCol: int
    maxRow: int

    def __repr__(self):
        return str(f"{self.minCol},{self.minRow},{self.maxCol},{self.maxRow}")

    def __str__(self):
        return self.__repr__()

    @property
    def asTuple(self):
        return tuple(self)

    @staticmethod
    def totalBoundingBox(box1: BoundingBox, box2: BoundingBox):
        totalBounding = BoundingBox(
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]))
        return totalBounding


class Rect:
    def __init__(self, box: BoundingBox = None, origin: Union[Point, tuple] = None, width=None, height=None):
        """

        :param box: tuple of the form (min_col, min_row, max_col, max_row)
        :param origin: Point or tuple (x,y) (col, row)
        :param width: width of box
        :param height: height of box
        """
        if (box is None and origin is None) and (width is None and height is None):
            raise Exception("Need at least one parameter to be non-null")

        if box is not None:
            self.origin = Point(box.minCol, box.minRow)
            self.width = box.maxCol - box.minCol
            self.height = box.maxRow - box.minRow

        if origin is not None:
            if isinstance(origin, Point):
                self.origin = origin
            else:
                self.origin = Point(origin[0], origin[1])

        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

    @property
    def center(self) -> Point:
        return Point(x=self.origin.x + self.width / 2.0, y=self.origin.y + self.height / 2.0)

    @property
    def limits(self) -> tuple:
        return self.origin.x, self.origin.y, self.origin.x + self.width, self.origin.y + self.height

    @property
    def vertices(self) -> np.ndarray:
        x = self.origin.x
        y = self.origin.y
        width = self.width
        height = self.height
        return np.array([[x, y],
                         [x + width, y],
                         [x + width, y + height],
                         [x, y + height]])

    @property
    def area(self):
        return self.width * self.height

    def contains(self, rect: Rect):
        return (self.origin.x < rect.origin.x and self.origin.y < rect.origin.y) \
               and \
               (self.width > rect.width and self.height > rect.height)


class VGGDict(dict):
    pass


class Vertices(np.ndarray):
    """
    Holds a polygon, with vertices

    :ivar xy: Nx2 np.ndarray of points, or Rect,  VGGDict

    """

    connected = True

    def __new__(cls, xy: Union[list, np.ndarray, Rect, VGGDict] = None, connected=True, **kwargs):
        # See numpy docs on subclassing ndarray
        if xy is None:
            xy = np.zeros((0, 2))

        # Default to integer type if not specified, since this is how pixel coordinates will be represented anyway
        if 'dtype' not in kwargs:
            kwargs['dtype'] = int

        if isinstance(xy, Rect):
            vertices = xy.vertices
        elif isinstance(xy, dict):  # VGG shape dictionaries
            if xy['name'] == 'rect':
                vertices = Rect(origin=(xy['x'], xy['y']), width=xy['width'], height=xy['height']).vertices
            elif xy['name'] == 'polygon':
                x = xy['all_points_x']
                y = xy['all_points_y']
                vertices = np.ones((len(x), 2))
                vertices[:, 0] = x
                vertices[:, 1] = y
            elif xy['name'] == 'circle':
                cx, cy = xy['cx'], xy['cy']
                r = xy['r']
                vertices = Rect(origin=(cx - int(r / 2.0), cy - int(r / 2.0)), width=int(r * 2.0),
                                height=int(r * 2.0)).vertices
            else:
                raise ImageException('shape dict parsing not implemented')
        else:
            vertices = xy

        if vertices is None:
            raise Exception("Points not initialized")

        if len(vertices.shape) > 2:
            vertices = vertices[0, :, :]

        if len(vertices.shape) != 2 or (vertices.shape[1] != 2 and vertices.shape[0] == 0):
            raise Exception(f'Badly formed xy argument: {xy}')

        if vertices.shape[0] < 3:
            raise Exception(f"Number of vertices must be >= 3, found {vertices.shape[0]}")

        arr = np.asarray(vertices, **kwargs).view(cls)
        arr.connected = connected
        return arr

    def __array_finalize__(self, obj):
        shape = self.shape
        shapeLen = len(shape)
        # indicates point, so the one dimension must have only 2 elements
        if 1 < shapeLen < 2 and shape[0] != 2:
            raise ImageException(f'A one-dimensional vertex array must be shape (2,).'
                                 f' Receieved array of shape {shape}')
        elif shapeLen > 2 or shapeLen > 1 and shape[1] != 2:
            raise ImageException(f'Vertex list must be Nx2. Received shape {shape}.')
        if obj is None:
            return

        self.connected = getattr(obj, 'connected', True)

    @property
    def empty(self):
        return len(self) == 0

    def asPoint(self):
        if self.size == 2:
            return self.reshape(-1)
        # Reaching here means the user requested vertices as point when
        # more than one point is in the list
        raise ImageException(f'asPoint() can only be called when one vertex is in'
                             f' the vertex list. Currently has shape {self.shape}')

    def asRowCol(self):
        return np.fliplr(self)

    @property
    def x(self):
        # Copy to array first so dimensionality checks are no longer required
        return np.array(self).reshape(-1, 2)[:, [0]]

    @x.setter
    def x(self, newX):
        self.reshape(-1, 2)[:, 0] = newX

    @property
    def y(self):
        return np.array(self).reshape(-1, 2)[:, [1]]

    @y.setter
    def y(self, newY):
        self.reshape(-1, 2)[:, 1] = newY

    @property
    def rows(self):
        return self.y

    @rows.setter
    def rows(self, newRows):
        self.y = newRows

    @property
    def cols(self):
        return self.x

    @cols.setter
    def cols(self, newCols):
        self.x = newCols

    @property
    def width(self):
        return np.max(self.x) - np.min(self.x)

    @property
    def height(self):
        return np.max(self.y) - np.min(self.y)

    @property
    def area(self):
        return cv.contourArea(self)

    @property
    def boundingBox(self) -> BoundingBox:
        """

        :return: returns Bounding Box Tuple
        """
        return BoundingBox(np.min(self.x), np.min(self.y), np.max(self.x),
                           np.max(self.y))

    @property
    def origin(self) -> Point:
        """
        Convenience method to return a tuple of x and y origin

        :return: tuple of x, y coordinates
        """
        bbox = self.boundingBox
        return Point(bbox[0], bbox[1])

    @property
    def center(self) -> Point:
        """
        Convenience method to return center of the rect

        :return: Point of x,y coordinates at center of rect
        """
        return Point(np.mean(self.x), np.mean(self.y))

    @property
    def centeredVertices(self):
        verts = np.array(self)
        verts[:, 0] = verts[:, 0] - np.min(verts[:, 0])
        verts[:, 1] = verts[:, 1] - np.min(verts[:, 1])
        return verts

    def containsPoint(self, point: Union[Point, tuple]):
        if isinstance(point, Point):
            pts = np.array([[point.x, point.y]])
        else:
            pts = np.array([np.array(point)])

        return skmeasure.points_in_poly(pts, self)[0]

    def verticesToString(self) -> str:
        return np.array2string(self, separator=",")

    def mask(self, **kwargs):
        centered = kwargs.get('centered', False)
        size = kwargs.get('size', None)

        if centered:
            centeredVertices = self.centeredVertices
            size = (centeredVertices[:, 1].max(), centeredVertices[:, 0].max())
            mask = np.zeros(size)
            return cv.drawContours(mask, [centeredVertices], 0, 1, -1)
        elif size is not None:
            mask = np.zeros(size)
            return cv.drawContours(mask, [self], 0, 1, -1)

        size = self[:, 1].max() + 1, self[:, 0].max() + 1
        mask = np.zeros(size)
        return cv.drawContours(mask, [self], 0, 1, -1)

    def __str__(self):
        return jsonpickle.dumps(self, unpicklable=True)

    @staticmethod
    def fromBoundingBoxString(string: str) -> Vertices:
        tup = tuple(map(int, string.split(',')))
        return Vertices(Rect(box=BoundingBox(*tup)))

    @staticmethod
    def fromVerticesString(string: str) -> Vertices:
        try:
            return Vertices(np.array(jsonpickle.loads(string)))
        except:
            if '[' not in string and ']' not in string:
                string = f"[{string}]"
                return Vertices(Rect(box=BoundingBox(*ast.literal_eval(string))))
            else:
                return Vertices(np.array(list(ast.literal_eval(string))))

    @staticmethod
    def fromJSONPickle(jsonString: str):
        return jsonpickle.loads(jsonString)


class Color(np.ndarray):
    colorSpace: ColorSpace
    label: int
    count: int
    freq: float

    @property
    def hsv(self):
        if self.colorSpace == ColorSpace.hsv:
            return self
        return Color(cv.cvtColor(self.reshape((-1, 1, 3)).astype(np.uint8), cv.COLOR_RGB2HSV).reshape((3,)),
                     colorSpace=ColorSpace.hsv, label=self.label, count=self.count, freq=self.freq)

    @property
    def rgb(self):
        if self.colorSpace == ColorSpace.rgb:
            return self
        return Color(cv.cvtColor(self.reshape((-1, 1, 3)).astype(np.uint8), cv.COLOR_HSV2RGB).reshape((3,)),
                     colorSpace=ColorSpace.rgb, label=self.label, count=self.count, freq=self.freq)

    def __new__(
        cls, value: Iterable,
        colorSpace: ColorSpace,
        label: int = None,
        count: int = None,
        freq: float = None,
        **kwargs
    ):

        arr: Color = np.asarray(value, **kwargs).view(cls)
        arr.colorSpace = colorSpace
        arr.label = label
        arr.count = count
        arr.freq = freq
        cls.__array_finalize__(arr)
        return arr

    def __array_finalize__(self, obj: np.ndarray = None):
        pass


# -------------------------------------------------
# Image Processing Classes
# -------------------------------------------------

class Image(np.ndarray):
    """
    Encapsulates image data, the original image path, and the 'name' of the image

    :ivar name: str, optional
    :ivar data: numpy.ndarrays
    :ivar colorSpace: ColorSpace, defaults to RGB if not supplied
    """

    name: str
    colorSpace: ColorSpace

    def __new__(
        cls,
        src: Optional[Union[Pathlike, np.ndarray, List[List[Image]]]] = None,
        colorSpace: ColorSpace = ColorSpace.rgb, name=None, **kwargs
    ):
        """
        Initializer for Image

        :param src: src can either be a Path, or a tuple of a string name, and raw numpy.ndarray data
        :param colorSpace: defaults to ColorSpace.rgb.  Note that OpenCV defaults to BGR when reading images from a file path
        """

        if isinstance(src, (str, Path)):
            src = Path(src)
            name: Optional[str] = src.name
            data: np.ndarray = cv.imread(src.resolve().as_posix())
            data = cv.cvtColor(data, cv.COLOR_BGR2RGB)
            colorSpace = ColorSpace.rgb
        elif isinstance(src, List) and isinstance(src[0], List) and isinstance(src[0][0], Image):
            im = Image.stitch(src)
            name = im.name
            data = im.data
        else:
            name = "Unnamed Image" if name is None else name
            data: np.ndarray = src

        shape = kwargs.pop('shape', None)
        if shape is not None:
            data = np.empty(shape=shape)

        arr: Image = np.asarray(data, **kwargs).view(cls)
        arr.name = name
        arr.colorSpace = colorSpace

        if arr.name is None:
            raise Exception(f"Image name can not be none with source: {src}.")
        if arr.data is None:
            raise Exception(f"Image data can not be none for source: {src}.")
        if arr.colorSpace is None:
            raise Exception(f"Image color model not set with source: {src}.")

        cls.__array_finalize__(arr)
        return arr

    def __array_finalize__(self, obj: np.ndarray = None):
        if obj is None:
            return
        for trait, default in self.getDefaultAttrs.items():
            setattr(self, trait, getattr(obj, trait, default))
        if self.numChannels == 1:
            self.colorSpace = ColorSpace.gray

    @property
    def getDefaultAttrs(self):
        return dict(name=None, colorSpace=ColorSpace.rgb)

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    @property
    def shape(self):
        return self.data.shape

    @property
    def red(self):
        return self.channel(0)

    @property
    def green(self):
        return self.channel(1)

    @property
    def blue(self):
        return self.channel(2)

    @property
    def numChannels(self):
        return self.shape[2] if self.ndim > 2 else 1

    @property
    def rgb(self):
        return self.red, self.green, self.blue

    def channelMeans(self):
        if self.numChannels == 3:
            return int(self.red.mean()), int(self.green.mean()), int(self.blue.mean())
        return self.mean()

    def split(self, num: tuple = (4, 4)) -> List[List[Image]]:
        return Image.splitImage(self, num)

    @staticmethod
    def uniqueColors(image: np.ndarray):
        return np.unique(image.reshape(-1, image.shape[-1]), axis=0, return_counts=True)

    @staticmethod
    def splitImage(image: Image, num: tuple = (4, 4)) -> List[List[Image]]:
        rSplit, cSplit = num
        # while image.shape[0] % rSplit != 0:
        #     rSplit += 1
        #
        # while image.shape[1] % cSplit != 0:
        #     cSplit += 1

        outImages = []
        rowImages = np.array_split(image, indices_or_sections=rSplit, axis=0)

        for im in rowImages:
            colImages = np.array_split(im, indices_or_sections=cSplit, axis=1)
            outImages.append([Image(im, name=image.name) for im in colImages])
        return outImages

    @staticmethod
    def stitch(images: Union[List[List[Image]], List[List[np.ndarray]]]) -> Image:
        outIm = None
        name = images[0][0].name if isinstance(images[0][0], Image) else None
        for row in images:
            rowIm = np.concatenate([im.data if isinstance(im, Image) else im for im in row], axis=1)
            if outIm is None:
                outIm = rowIm
            else:
                outIm = np.concatenate((outIm, rowIm), axis=0)
        return Image(outIm, name=name)

    def channel(self, num):
        if self.numChannels > 1:
            return self[:, :, num]
        return self

    @staticmethod
    def _getColorSpaceConstant(fromSpace: ColorSpace, toSpace: ColorSpace) -> Optional[str]:
        """
        Returns the OpenCV constant for to color space conversion pass into cvtColor
        :param fromSpace: Source `ColorSpace` enum value
        :param toSpace: Target `ColorSpace` enum value
        :return: OpenCV constant string
        """
        cvtConst = getattr(cv, f"COLOR_{str(fromSpace.value).upper()}2{str(toSpace.value).upper()}",
                           None)
        if cvtConst is None:
            print(f"Unable to find OpenCV constant for ColorSpaces: {fromSpace} - {toSpace}")
        return cvtConst

    def cropped(self, bbox: BoundingBox):
        return self[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    @staticmethod
    def convertColorSpace(image: Image, toSpace: ColorSpace) -> Optional[Image]:
        """
        Returns a new Image with the new colorspace, :code:`None` if :code:`inPlace` is False

        :param image:
        :param toSpace: Target `ColorSpace` value
        :return: Image or None
        """
        cvtConst = Image._getColorSpaceConstant(fromSpace=image.colorSpace, toSpace=toSpace)
        if cvtConst is None:
            return None
        return cv.cvtColor(image, cvtConst)

    def asPoints(self):
        return self.reshape((np.shape(self)[0] * np.shape(self)[1], 3))

    @property
    def hsv(self):
        hsvIm = self.asHSV()
        return hsvIm[:, :, 0], hsvIm[:, :, 1], hsvIm[:, :, 2]

    def asHSV(self) -> Image:

        if self.colorSpace == ColorSpace.hsv:
            return self

        if self.colorSpace == ColorSpace.rgb:
            im = Image(cv.cvtColor(self, cv.COLOR_RGB2HSV))
            im.colorSpace = ColorSpace.hsv
            return im

    def asGrayScale(self) -> Image:
        if self.colorSpace != ColorSpace.gray:
            return Image.convertColorSpace(Image.toUint8(self), ColorSpace.gray).view(Image)
        else:
            return self

    @staticmethod
    def paletteFromColors(colors: List[Color], patchSize):
        numColors = len(colors)
        patchImage = np.zeros((patchSize, patchSize * numColors, 3)).astype(np.uint8)

        for i, color in enumerate(colors):
            c = (int(color[0]), int(color[1]), int(color[2]))
            colStart = patchSize * i
            colEnd = patchSize * i + patchSize - 1
            cv.rectangle(patchImage, (colStart, 0), (colEnd, patchSize - 1), c, -1)
            center = (int(colStart) + int(patchSize * .10), int(patchSize / 2.0))
            cv.putText(patchImage, f"{color.label}:{color.freq:.2f}", center, cv.FONT_HERSHEY_PLAIN, 1, 2)
        return patchImage

    @property
    def area(self):
        return self.shape[0] * self.shape[1]

    def plotHueHistogram(self, convert=True):
        if convert:
            h, s, v = cv.split(cv.cvtColor(self, cv.COLOR_RGB2HSV))
        else:
            h, s, v = cv.split(self)

        plt.figure()
        sb.distplot(h.flatten(), bins=256)
        plt.show()

        hist, edges = np.histogram(h.flatten(), bins=180)

        return np.argmax(hist)

    def show(self: Image, cmap='afmhot', savePath: Pathlike = None):
        plt.figure()
        if self.numChannels == 1:
            plt.imshow(self, cmap=cmap)
        else:
            plt.imshow(self)
        plt.axis('off')
        plt.title(self.name)
        if savePath is None:
            plt.show()
        else:
            plt.savefig(Path(savePath).resolve().as_posix())

    def plotImageColors3D(self, resize=None):
        if resize is not None:
            small_img = cv.resize(self,
                                  (0, 0),  # set fx and fy, not the final size
                                  fx=resize[0],
                                  fy=resize[1],
                                  interpolation=cv.INTER_CUBIC)
        else:
            small_img = self

        c1, c2, c3 = cv.split(small_img)

        pixelColors = small_img.reshape((np.shape(small_img)[0] * np.shape(small_img)[1], 3))
        norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
        norm.autoscale(pixelColors)
        pixelColors = norm(pixelColors).tolist()

        fig = plt.figure()
        axis = fig.add_subplot(1, 2, 1, projection="3d")
        axis.scatter(c1.flatten(), c2.flatten(), c3.flatten(), facecolors=pixelColors, marker=".")
        axis.set_xlabel('Red')
        axis.set_ylabel('Green')
        axis.set_zlabel('Blue')
        plt.show()

    def vectorized(self):
        return self.reshape(self.shape[0] * self.shape[1], 3)

    def extractColors(
        self, numColors: int = 5,
        resize: bool = True,
        colorSpace=ColorSpace.hsv,
        targetWidth: int = 640,
        patchSize: int = 100,
        method='opencv',
        mask=None
    ):
        """
        Uses K-Means to extract the top `numColors` colors in the image.

        :param method: 'opencv' or 'sklearn'
        :param colorSpace: the colorspace to perform the clustering in
        :param patchSize:
        :param mask:
        :param targetWidth:
        :param numColors:
        :param resize: Resizes image to 512x512
        :return: tuple of ((tuple color, frequency), patchImage)
        """
        if method == 'opencv':
            im = self
            if colorSpace == ColorSpace.hsv:
                im = im.asHSV()

            stopCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret, labels, centers = cv.kmeans(np.float32(im.vectorized()),
                                             K=numColors,
                                             bestLabels=None,
                                             criteria=stopCriteria,
                                             attempts=10,
                                             flags=cv.KMEANS_PP_CENTERS)
            centers = np.uint8(centers)
            labels = labels.flatten()
            res = centers[labels]
        else:
            if resize and self.width > targetWidth:
                h = self.width / self.height * targetWidth
                im = Image(cv.resize(self, (int(h), targetWidth)))
            else:
                im = self

            if colorSpace == ColorSpace.hsv:
                im = im.asHSV()

            model = skcluster.KMeans(n_clusters=numColors)
            labels = model.fit_predict(im.vectorized())
            centers = np.array(model.cluster_centers_).astype(np.int)
            res = centers[labels]

        resultImage = np.uint8(res.reshape(im.shape))

        if colorSpace == ColorSpace.hsv:
            resultImage = cv.cvtColor(resultImage, cv.COLOR_HSV2RGB)

        labelCounts = collections.Counter(labels)
        totalCount = sum(labelCounts.values())

        _colors = []
        for key, count in labelCounts.items():
            _colors.append(Color(centers[key],
                                 colorSpace=colorSpace,
                                 label=key,
                                 count=count,
                                 freq=count / totalCount).rgb)

        _colors.sort(key=lambda x: x.count, reverse=True)

        labelImage = labels.reshape(im.shape[:2])

        patchImage = Image.paletteFromColors(_colors, patchSize=patchSize)

        return _colors, resultImage, labelImage, patchImage

    @classmethod
    def toUint8(cls, image: Image):
        if image.dtype == float:
            # Assume already normed to 1 as max, scale to 255
            image *= 255
        return image.astype('uint8').view(cls)


class Component:

    def __init__(
        self, shape: Union[Vertices, Rect, BoundingBox, str, np.ndarray], regionprops=None,
        data: pd.Series = None
    ):
        """

        :param shape: string shape is assumed to be a string of vertices
        :param data:
        """

        if shape is None:
            self.vertices = Vertices.fromVerticesString(data['Vertices'])
        else:
            if isinstance(shape, Vertices):
                self.vertices = shape
            elif isinstance(shape, Rect) or isinstance(shape, np.ndarray):
                self.vertices = Vertices(shape)
            elif isinstance(shape, BoundingBox):
                self.vertices = Vertices(Rect(box=shape))
            elif isinstance(shape, str):
                self.vertices = Vertices.fromVerticesString(shape)
            else:
                raise ImageException('Invalid Input to Component')

        self.regionprops = self.calcRegionProps() if regionprops is None else regionprops
        self.data: pd.Series = data

    @property
    def deviceClass(self):
        try:
            return self.data['Class']
        except TypeError:
            return None
        except KeyError:
            return None

    @property
    def allCoords(self):
        return self.regionprops.coords

    def calcRegionProps(self):
        mask = self.vertices.mask()
        rp = skmeasure.regionprops(mask.astype(np.uint8))
        # plt.imshow(mask), plt.show()
        return rp[0]


class ProgressBar:
    waiting_done = False
    waiting_bar = [
        " [=     ]",
        " [ =    ]",
        " [  =   ]",
        " [   =  ]",
        " [    = ]",
        " [     =]",
        " [    = ]",
        " [   =  ]",
        " [  =   ]",
        " [ =    ]",
    ]
    waiting_text = ""

    def animate(self, text):
        self.waiting_text = text
        for c in itertools.cycle(self.waiting_bar):
            if self.waiting_done:
                self.waiting_done = False
                break
            sys.stdout.write(f'\r{self.waiting_text} ' + c)
            sys.stdout.flush()
            time.sleep(0.35)

    def show(self, text):
        t = threading.Thread(target=self.animate, args=(text,))
        t.start()

    def done(self):
        self.waiting_done = True

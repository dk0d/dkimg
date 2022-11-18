from matplotlib import patches
import matplotlib.gridspec as gridspec
from .constants import MatchColumns
from .common import Component, Image
from typing import Union, Tuple
# from pyqtgraph.Qt import QtGui, QtCore
# import pyqtgraph as pg
import pandas as pd
from dataclasses import dataclass
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sb
from typing import List
from pathlib import Path
import numpy as np
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class ColorPair:
    def __init__(self, color1, color2):
        self.color1 = color1
        self.color2 = color2


def pairPlot(data,
             plotSets,
             hue=None,
             title=None,
             colPrefix="",
             savePath: Path = None):
    for setTitle, pset in plotSets.items():
        plotcols = [f'{colPrefix}{p}' for p in pset]
        g = sb.pairplot(data=data, vars=plotcols, hue=hue)
        if title is not None:
            g.fig.suptitle({title}, y=1.08, fontweight="bold", fontsize=14)
        if savePath is not None:
            plt.savefig(savePath.as_posix())


def plotComponents(components: List[Component],
                   image: Image = None,
                   title=None,
                   polyColors: Union[str, Tuple[float, float, float]] = 'r',
                   label=None,
                   ax: plt.Axes = None):
    if ax is None:
        fig, ax = plt.subplots(1)

    if image is not None:
        ax.imshow(image.asGrayScale(), cmap='gray')

    ax.set_axis_off()

    if title is not None:
        ax.title(title)

    rect_handle = None
    for i, box in enumerate(components):
        minc, minr, maxc, maxr = box.vertices.boundingBox
        # rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr, linewidth=1, edgecolor=polyColors,
        #                          facecolor='none', label=label)
        poly = patches.Polygon(box.vertices, edgecolor=polyColors, linewidth=2,
                               facecolor='none', label=label)
        ax.add_patch(poly)
        if i == 0:
            rect_handle = poly

    return rect_handle


def drawComponents(components: List[Component], image: Image, color=(0, 255, 0), thickness=3):
    """
    Draws components onto a color image in-place

    :param components: List of Component
    :param image: Image
    :param color: color needs to be an np.ndarray or tuple and UINT8
    :param thickness:
    :return:
    """
    contours = [comp.vertices for comp in components]
    cv.drawContours(image, contours=contours, contourIdx=-1, color=color, thickness=thickness)


def plotDetectedDeviceClassSummary(self, prop: str = 'area', savedir: Path = None):
    plotTypes = [sb.swarmplot, sb.boxenplot, sb.barplot]
    plotData = self.allAnalytics.loc[
        (self.allAnalytics['detectionType'] == 'FP') | (self.allAnalytics['detectionType'] == 'TP')]

    classes = pd.unique(plotData['class'])
    if len(classes) > 1:
        print(f"Not enough classes detected to plot -- Classes: {classes}")
        return

    fig = sb.catplot(x=prop, y="detectionType", row='class',
                     height=1.5, aspect=5, orient='h',
                     data=plotData, figsize=(10, 10))

    imgName = self.process.image.name.replace(".", "_")
    savedir = savedir.joinpath(
        f"{imgName}_{self.process.name}_detected-class.png") if savedir is not None else None
    saveShowFigure(fig, savedir)


def plotDeviceClassProperty(self, prop: str = "area", savedir: Path = None):
    classProps = self.allAnalytics.groupby('class')
    # import hvplot.pandas
    # plotData = self.allAnalytics.loc[pd.notna(self.allAnalytics['class'])]

    plotTypes = [sb.swarmplot, sb.boxenplot, sb.barplot]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))

    for ptype, ax in zip(plotTypes, axes.ravel()):
        ptype(x="class", y=prop, hue="detectionType", data=self.allAnalytics, ax=ax)
    plt.suptitle(f'{self.name} Detection Statistics')
    imgName = self.process.image.name.replace(".", "_")
    savedir = savedir.joinpath(
        f"{imgName}_{self.process.name}_{property}-class.png") if savedir is not None else None
    saveShowFigure(fig, savedir)


def saveShowFigure(fig, savePath: Path = None, overwrite=False):
    if savePath is not None:
        if savePath.exists() and not overwrite:
            print(f"...{savePath} exists, skipping save..")
            plt.close()
            return
        fig.savefig(savePath.as_posix())
        plt.close()
    else:
        plt.show()


# def pyqtPlotstages(process: ImageProcess, saveDir: Path = None, name: str = None, maxCols=4):
#     stages: List[ProcessStage] = process.stagesFlattened()
#
#     if len(stages) == 0:
#         print("Steps list is empty.....")
#         return
#
#     numStages = len(stages) + 1
#     ncols = min(numStages, maxCols)
#     nrows = int(np.ceil(numStages / ncols))
#
#     win = pg.GraphicsLayoutWidget(show=True)
#     origItem = None
#     indices = [tuple(i) for i in list(np.argwhere(np.ones((nrows, ncols))))]
#     for stage, idx in zip(stages, indices):
#         title = stage.name
#         result = ImageIO(**stage.result) if stage.result else ImageIO()
#         imageItem = win.addItem(title=title, item=pg.ImageItem(result.display))
#
#         if origItem is None:
#             origItem = imageItem
#         else:
#             imageItem.setXLink(origItem)
#             imageItem.setYLink(origItem)


def colorMap(colorMaps: List[ColorPair], steps=600) -> np.ndarray:
    gradWidth = 20
    cmapImage = np.zeros((steps, gradWidth * len(colorMaps), 3))
    for c, cmap in enumerate(colorMaps):
        gradient = np.zeros((steps, 3))
        gradient[:, 0] = np.round(np.linspace(cmap.color1[0], cmap.color2[0], steps))
        gradient[:, 1] = np.round(np.linspace(0, 255, steps))
        gradient[:, 2] = np.round(np.linspace(0, 255, steps))
        for s in range(steps):
            for col in range(c * gradWidth, c * gradWidth + gradWidth):
                cmapImage[s, col, :] = gradient[s, :]
    return cv.cvtColor(cmapImage.astype('uint8'), cv.COLOR_HSV2RGB)


def plotSphere(c=None, r=None, w=0, subdev=20, ax=None, sigma_multiplier=1):
    """
        plot a sphere surface
        Input:
            c: 3 elements list, sphere center
            r: 3 element list, sphere original scale in each axis ( allowing to draw elipsoids)
            subdiv: scalar, number of subdivisions (subdivision^2 points sampled on the surface)
            ax: optional pyplot axis object to plot the sphere in.
            sigma_multiplier: sphere additional scale (choosing an std value when plotting gaussians)
        Output:
            ax: pyplot axis object
    """
    if r is None:
        r = [1, 1, 1]
    if c is None:
        c = [0, 0, 0]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:complex(0, subdev), 0.0:2.0 * pi:complex(0, subdev)]
    x = sigma_multiplier * r[0] * sin(phi) * cos(theta) + c[0]
    y = sigma_multiplier * r[1] * sin(phi) * sin(theta) + c[1]
    z = sigma_multiplier * r[2] * cos(phi) + c[2]
    cmap = cmx.ScalarMappable()
    cmap.set_cmap('jet')
    c = cmap.to_rgba(w)
    ax.plot_surface(x, y, z, color=c, alpha=0.1, linewidth=1)
    return ax

from .common import Image, Rect, lazy_property, Component, BoundingBox, Vertices
from .constants import MatchColumns, IntersectionType
from .processing import ImageProcess, ImageException, ImageIO, ProcessIO
from . import algorithms as alg
from . import plotting
import cv2 as cv
from pathlib import Path
import jsonpickle
from typing import List, Union, Optional, Tuple
import numpy as np
from sklearn import preprocessing
import pandas as pd
import skimage.feature as skfeature
import cloudpickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sb
import tqdm
from skimage.measure import regionprops, regionprops_table, shannon_entropy
import copy


class AnnotationComparisonResult:

    def __init__(self, matched: pd.DataFrame, falsePos, falseNeg, name=None):
        """

        :param matched:
        :param falsePos:
        :param falseNeg:
        :param name:
        """
        self.matched: pd.DataFrame = matched
        self.falsePos: List[Component] = falsePos
        self.falseNeg: List[Component] = falseNeg
        self.name = str(name) if name is not None else "Annotation Result"

    @lazy_property
    def FP(self):
        return len(self.falsePos)

    @lazy_property
    def TP(self):
        return self.matched.shape[0]

    @lazy_property
    def FN(self):
        return len(self.falseNeg)

    @lazy_property
    def precision(self):  # TP / (TP + FP)
        try:
            return self.TP / (self.TP + self.FP)
        except ZeroDivisionError:
            return np.nan

    @lazy_property
    def truePosRate(self):  # TP / (TP + FN)
        try:
            return self.TP / (self.TP + self.FN)
        except ZeroDivisionError:
            return np.nan

    @lazy_property
    def missRate(self):  # FN / (FN + TP) = 1 - TPR
        try:
            return self.FN / (self.FN + self.TP)
        except ZeroDivisionError:
            return np.nan

    @property
    def groundTruthComponents(self) -> List[Component]:
        # return [Component(shape=r[MatchColumns.groundTruth.name], data=r) for _, r in self.matched.iterrows()]
        return list(self.matched[MatchColumns.groundTruth.name].values)

    @property
    def truePositiveComponents(self) -> List[Component]:
        # return [Component(shape=r[MatchColumns.detected.name], data=r) for _, r in self.matched.iterrows()]
        return list(self.matched[MatchColumns.truePositive.name])

    @property
    def matchedTotalBoundingComponents(self) -> List[Component]:
        return [Component(shape=r[MatchColumns.totalBBox.name], data=r) for _, r in self.matched.iterrows()]

    @lazy_property
    def analytics(self) -> pd.Series:
        analytics = pd.Series()
        analytics.loc['labeled'] = self.matched.shape[0] + self.FN
        analytics.loc['detected'] = self.matched.shape[0] + self.FP
        analytics.loc['falsePositives'] = self.FP
        analytics.loc['falseNegatives'] = self.FN
        analytics.loc['truePosRate'] = self.truePosRate
        analytics.loc['precision'] = self.precision
        analytics.loc['missRate'] = self.missRate
        analytics.loc['iouMean'] = np.mean(self.matched['iouScore'].values)
        analytics.loc['iouStdDev'] = np.std(self.matched['iouScore'].values)
        analytics = analytics.append(self._interSectTypeAnalytics(IntersectionType.equalOffset))
        analytics = analytics.append(self._interSectTypeAnalytics(IntersectionType.overOffset))
        analytics = analytics.append(self._interSectTypeAnalytics(IntersectionType.underOffset))
        analytics = analytics.append(self._interSectTypeAnalytics(IntersectionType.innerMatch))
        analytics = analytics.append(self._interSectTypeAnalytics(IntersectionType.outerMatch))
        analytics.name = self.name
        return analytics

    def _interSectTypeAnalytics(self, t: IntersectionType) -> pd.Series:
        fil = self.matched['intersectType'] == t.name
        ser = pd.Series()
        ser.loc[f'{t.name}Num'] = self.matched.loc[fil].shape[0]
        ser.loc[f'{t.name}Mean'] = np.mean(self.matched.loc[fil, 'iouScore'])
        ser.loc[f'{t.name}Std'] = np.std(self.matched.loc[fil, 'iouScore'])
        return ser

    def pairPlot(self):
        cols = {
            "MatchColumns": [MatchColumns.groundTruthArea.name,
                             MatchColumns.truePositiveArea.name,
                             MatchColumns.totalBBoxArea.name,
                             MatchColumns.iouScore.name]
        }
        hueVal = MatchColumns.intersectType.name
        normed = self.matched[cols]
        normed = pd.DataFrame(preprocessing.scale(normed), columns=cols)
        normed[hueVal] = self.matched[hueVal]
        plotting.pairPlot(data=normed, plotSets=cols, hue=hueVal)

    def plotAnalysisResult(self, savePath: Path = None, overwrite=False):
        try:
            # sb.set(style='whitegrid')
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            sb.barplot(x='Device Type', y='iouScore', hue="intersectType", data=self.matched, palette='muted',
                       ax=ax[0])
            sb.despine(ax=ax[0])
            ax[0].set_ylabel('IoU Score')
            ax[0].set_xlabel('Intersection Type')
            ax[0].set_title('IoU Scores')

            sb.countplot(x='Device Type', data=self.matched, ax=ax[1])
            sb.despine(ax=ax[1])
            ax[1].set_ylabel('Count')
            ax[1].set_xlabel('Matched Device Type')
            ax[1].set_title('Device Type Distribution')

            plt.tight_layout()
            plotting.saveShowFigure(fig, savePath, overwrite)
        except ValueError:
            print(f'Unable to plot stats, no matches found')

    def resultsForType(self, intersectType: IntersectionType):
        typeFilter = self.matched['intersectType'] == intersectType.value
        return self.matched.loc[typeFilter, :]

    def printTable(self):
        max_rows = pd.get_option('display.max_rows')
        max_cols = pd.get_option('display.max_columns')
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(self.analytics)
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_cols)


class DetectionMetric:

    @staticmethod
    def iouBoundingBox(box1: Union[tuple, list], box2: Union[tuple, list]):
        """
        Calculates the ratio of the area of intersection over the area of the union of the two bounding boxes
        :param box1: tuple or list of (min_col, min_row, max_col, max_row)
        :param box2: tuple or list of (min_col, min_row, max_col, max_row)
        :return:
        """
        # TODO: Implement generalized intersection over union (?)
        #   https://www.groundai.com/project/generalized-intersection-over-union-a-metric-and-a-loss-for-bounding-box-regression/1
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        if interArea == 0:
            return 0

        # compute the area of both the prediction and ground-truth
        # rectangles
        box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(box1Area + box2Area - interArea)

        # return the intersection over union value
        return iou

    @staticmethod
    def iouCoords(mask1: np.ndarray, mask2: np.ndarray):
        """

        :param mask1: Nx2 np.ndarray of [col, row] vertices
        :param mask2: Nx2 np.ndarray of [col, row] vertices
        :return:
        """
        #
        # s1 = set([tuple(coords1[i, :]) for i in range(coords1.shape[0])])
        # s2 = set([tuple(coords2[i, :]) for i in range(coords2.shape[0])])
        imSize = max(mask1.shape[0], mask2.shape[0]), max(mask1.shape[1], mask2.shape[1])
        im1 = np.zeros(imSize)
        im2 = np.zeros(imSize)
        im1[:mask1.shape[0], :mask1.shape[1]] = mask1
        im2[:mask2.shape[0], :mask2.shape[1]] = mask2
        intersect = np.sum(cv.bitwise_and(im1, im2))
        if intersect == 0.0:
            return intersect
        _union = np.sum(cv.bitwise_or(im1, im2))
        iou = intersect / _union
        return iou


# -------------------------------------------------
# Image Annotation Helpers
# -------------------------------------------------
class ImageAnnotation:
    _filenameColumn: str = "Source Image Filename"

    @property
    def filenames(self):
        return self._filenames(self._filenameColumn, self.annotations)

    @staticmethod
    def _filenames(filenameColumn: str, df: pd.DataFrame):
        return list(np.unique(df[filenameColumn]))

    @property
    def imagePaths(self):
        return self._imagePaths(self.imagePath, self.annotations)

    @staticmethod
    def _imagePaths(rootPath: Path, df: pd.DataFrame):
        return [rootPath.joinpath(f) for f in ImageAnnotation._filenames(ImageAnnotation._filenameColumn, df)]

    def __init__(self, annotationPath: Path, imageDirPath: Path):
        self.path: Path = annotationPath
        self.imagePath = imageDirPath
        self.annotations: Optional[pd.DataFrame] = None
        self._loadAnnotations(path=self.path)

    def _loadTableAtPath(self, path: Path):
        if path.is_dir():
            for file in path.glob('*.csv'):
                if self.annotations is None:
                    self.annotations = pd.read_csv(file.absolute().as_posix())
                else:
                    self.annotations = pd.concat([self.annotations, pd.read_csv(file.absolute().as_posix())])
            self.annotations.reset_index(drop=True, inplace=True)
        else:
            self.annotations = pd.read_csv(path.absolute().as_posix())

    def _loadAnnotations(self, path: Path):
        if not path.exists():
            raise Exception(f'No file found for path \'{self.path}\'')

        self._loadTableAtPath(path=path)

        if 'region_shape_attributes' in self.annotations.columns:
            self.annotations = ImageAnnotation._legacyConvert(self.annotations)

        rn = None
        if 'filename' in self.annotations.columns:
            rn = {'filename': self._filenameColumn}
        elif 'Filename' in self.annotations.columns:
            rn = {'Filename': self._filenameColumn}
        if rn is not None:
            self.annotations.rename(columns={'filename': self._filenameColumn})
        self.annotations.sort_values(by=self._filenameColumn, inplace=True)

    @staticmethod
    def _regionShapeToVertices(shape: Union[dict, str]) -> str:
        if isinstance(shape, str):
            shape = jsonpickle.loads(shape)
        return Vertices(shape).verticesToString()

    @staticmethod
    def _legacyConvert(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts a DataFrame table from VGG annotator and formats it to an ImgAnnotator table
        :param df:
        :return:
        """

        targetColumns = ['Instance ID',
                         'Vertices',
                         'Validated',
                         'Source Image Filename',
                         'Author',
                         'Timestamp',
                         'Device Text',
                         'Class',
                         'Board Text',
                         'Logo',
                         'Notes']

        columnMap = {
            'filename': 'Source Image Filename',
            'file_size': None,
            'region_count': None,
            'region_id': 'Instance ID',
            'region_shape_attributes': ('Vertices', lambda x: ImageAnnotation._regionShapeToVertices(x)),
            'region_attributes': {
                'type': 'Class',
                'name': 'Device Name',
                'text on devICse': 'Device Text',
                'logo': 'Logo',
                'notes': 'Notes'
            }
        }

        newDF = pd.DataFrame(columns=targetColumns)

        for col, value in columnMap.items():
            if isinstance(value, str):
                newDF[value] = df[col]
            elif isinstance(value, tuple):
                newDF[value[0]] = df[col].apply(value[1])
            elif isinstance(value, dict):
                for subCol, subVal in value.items():
                    newDF[subVal] = df[col].apply(lambda x: jsonpickle.loads(x)[subCol] if subCol in x else '')
        return newDF

    @staticmethod
    def _verticesToComponents(frame: pd.DataFrame):
        # TODO: read in image and calc region props based on label image
        return [Component(row['Vertices'], regionprops=None, data=row) for _, row in frame.iterrows()]

    @staticmethod
    def _polygonsToVertices(polygons: List[Vertices]) -> List[str]:
        return [poly.verticesToString() for poly in polygons]

    ########################################################################

    def components(self, filterValue=None, column: str = None) -> List[Component]:
        if column is None:
            column = self._filenameColumn
        if filterValue is not None:
            try:
                filtered = self.annotations.loc[self.annotations[column] == filterValue]
                filtered.reset_index(drop=True, inplace=True)
            except:
                filtered = self.annotations
        else:
            filtered = self.annotations
        comps = ImageAnnotation._verticesToComponents(filtered)
        return comps

    def plotOn(self, image: Image, title=None, ax=None):
        plotting.plotComponents(components=self.components(filterValue=image.name),
                                image=image, title=title,
                                ax=ax)

    @staticmethod
    def intersectionType(labelRect: Union[BoundingBox, list],
                         detectedRect: Union[BoundingBox, list]) -> IntersectionType:
        labeled = Rect(labelRect)
        detected = Rect(detectedRect)
        lArea = labeled.area
        dArea = detected.area

        if dArea > lArea:  # can be outer, over
            if detected.contains(labeled):
                return IntersectionType.outerMatch
            return IntersectionType.overOffset
        elif dArea < lArea:  # can be inner, under
            if labeled.contains(detected):
                return IntersectionType.innerMatch
            return IntersectionType.underOffset

        # equal offset
        return IntersectionType.equalOffset

    @staticmethod
    def detectionFromAnnotation(annotationPath: Path):
        if not annotationPath.is_file():
            raise ImageException(
                "ImageAnnotation can only create a process for a single annotation file, not a directory of files")

        ann = ImageAnnotation(annotationPath)
        if len(ann.imagePaths) > 1:
            raise ImageException(
                "ImageAnnotation can only create a process for an annotation file that contains a single image")

        imPath = ann.imagePaths[0]
        if not imPath.exists():
            raise ImageException(f"ImageAnnotation points to image file that does not exist: {imPath}")

        image = Image(ann.imagePaths[0])

        def _componentLabelImage(_image: Image, components: List[Component]):
            labels, metadata = ComponentAnalysis.labelImageFromComponents(components, imShape=_image.shape)
            return ImageIO(image=labels, labels=labels, labelMetadata=metadata)

        proc = ImageProcess.fromFunction(_componentLabelImage, name="Annotation Process")
        proc.run(ImageIO(image=image, components=ann.components()))

        result = ComponentDetection.findComponents(proc)

        return result


class ComponentDetectionResult:

    def __init__(self, process: ImageProcess, components: Optional[List[Component]]):
        self.process: ImageProcess = process
        self.components: Optional[List[Component]] = components
        self.comparisonResult: Optional[AnnotationComparisonResult] = None

    @property
    def name(self):
        return f"{self.process.image.name}|{self.process.name}"

    @property
    def filename(self):
        return f"{self.name.replace('.', '_')}.res"

    @property
    def analytics(self):
        return self.comparisonResult.analytics

    def pairplot(self, saveDir: Path = None):
        plotSets = {
            "Intensities": ['mean', 'stDev', 'gradMean', 'gradStdDev', 'laplace_mean', 'laplace_stDev'],
            "GLCM": ['ASM', 'contrast', 'energy', 'correlation', 'homogeneity', 'dissimilarity']
        }
        for c in range(len(self.process.image.shape)):
            title = f'Channel {c}'
            fullPath = saveDir if saveDir is None else saveDir.joinpath(
                f'{self.name.replace(".", "_")}_c{c}_{title}.png')
            plotting.pairPlot(data=self.comparisonResult,
                              plotSets=plotSets,
                              title=title,
                              hue='detectionType',
                              savePath=fullPath)

    def compare(self, toAnnotation: ImageAnnotation, force=False):
        """
        Compares the loaded annotations to a list of test polygons.

        :param force:
        :param toAnnotation: ImageAnnotation
        :return AnnotationResult
        """
        if not self.process.dirty:
            raise ImageException(f"Process must be run before comparison: {self.process.name}")

        if self.components is None:
            result = ComponentDetection.findComponents(self.process)
            self.components = copy.copy(result.components)

        if not force and self.comparisonResult is not None:
            return

        detectedPolygons = self.components
        trueComponents = toAnnotation.components(filterValue=self.process.image.name)
        if len(trueComponents) == 0:
            raise ImageException(f'No compare annotations found... ({self.process.image.name})')

        falsePositives: List[Component] = detectedPolygons
        falseNegatives: List[Component] = []
        matchCols = MatchColumns.allValues()
        annCols = list(toAnnotation.annotations.columns)

        for c in ['Instance ID', 'Vertices', 'Validated']:
            annCols.remove(c)

        matchCols.extend(annCols)
        matchTable = pd.DataFrame(columns=matchCols)

        for c, comp in enumerate(tqdm.tqdm(trueComponents, desc=f'{self.name} - Comparing components')):
            candidateIdxs = []
            for i, fp_box in enumerate(falsePositives):
                score = DetectionMetric.iouBoundingBox(comp.vertices.boundingBox, fp_box.vertices.boundingBox)
                if score > 0:
                    candidateIdxs.append([i, score])
            matchedIdx = None
            matchedScore = None
            if len(candidateIdxs) > 0:
                scores = np.array(candidateIdxs)
                maxScoreIdx = int(np.argmax(scores[:, 1]))
                matchedIdx, matchedScore = scores[maxScoreIdx, :]
                matchedIdx = int(matchedIdx)
            if matchedIdx is not None:
                fp_box = falsePositives.pop(matchedIdx)
                matchedScore = DetectionMetric.iouCoords(comp.vertices.mask(), fp_box.vertices.mask())
                iType = ImageAnnotation.intersectionType(comp.vertices.boundingBox, fp_box.vertices.boundingBox).value
                row = {MatchColumns.groundTruth.name: comp,
                       MatchColumns.truePositive.name: fp_box,
                       MatchColumns.iouScore.name: matchedScore,
                       MatchColumns.intersectType.name: iType}
                for index, val in comp.data.items():
                    if index in annCols:
                        row[index] = val
                matchTable = matchTable.append(row, ignore_index=True)
            else:
                falseNegatives.append(comp)
        self.comparisonResult = AnnotationComparisonResult(matched=matchTable,
                                                           falsePos=falsePositives,
                                                           falseNeg=falseNegatives,
                                                           name=self.name)

    @lazy_property
    def falsePosAnalytics(self) -> pd.DataFrame:
        return self.allAnalytics.loc[self.allAnalytics['detectionType'] == 'FP', :]

    @lazy_property
    def falseNegAnalytics(self) -> pd.DataFrame:
        return self.allAnalytics.loc[self.allAnalytics['detectionType'] == 'FN', :]

    @lazy_property
    def groundTruthAnalytics(self):
        return self.allAnalytics.loc[self.allAnalytics['detectionType'] == 'GT', :]

    @lazy_property
    def detectedAnalytics(self) -> pd.DataFrame:
        return self.allAnalytics.loc[self.allAnalytics['detectionType'] == 'TP', :]

    # @lazy_property
    # def matchTotalBoxAnalytics(self) -> pd.DataFrame:
    #     return self.allAnalytics.loc[self.allAnalytics['detectionType'] == 'TB', :]

    @lazy_property
    def allAnalytics(self) -> pd.DataFrame:
        allComps = [(c, 'GT') for c in self.comparisonResult.groundTruthComponents]
        allComps.extend([(c, 'FP') for c in self.comparisonResult.falsePos])
        allComps.extend([(c, 'FN') for c in self.comparisonResult.falseNeg])
        allComps.extend([(c, 'TP') for c in self.comparisonResult.truePositiveComponents])
        # allComps.extend([(c, 'TB') for c in self.analysisResult.matchedTotalBoundingComponents])
        return ComponentAnalysis.analyticsFor(self.name, self.process.image, allComps)

    def runAnalytics(self):
        _ = self.allAnalytics

    def getAnalytics(self, stats: List[str]):
        outs = []
        for st in stats:
            if st.lower() == 'gt':
                outs.append(self.groundTruthAnalytics)
            elif st.lower() == 'de':
                outs.append(self.detectedAnalytics)
            # elif st.lower() == 'tb':
            #     outs.append(self.matchTotalBoxAnalytics)
            elif st.lower() == 'fn':
                outs.append(self.falseNegAnalytics)
            elif st.lower() == 'fp':
                outs.append(self.falsePosAnalytics)
        return pd.concat(outs)

    def plot(self, imgAsGray=True, saveDir: Path = None, opencv=False):
        if self.comparisonResult is None:
            print('Analysis not yet set')
            return

        tp = self.comparisonResult.truePosRate
        precision = self.comparisonResult.precision
        scores = list(self.comparisonResult.matched[MatchColumns.iouScore.name])

        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3)

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(self.process.image)
        ax.set_axis_off()
        ax.set_title(f'Original Image - {self.process.image.name}')

        ax = fig.add_subplot(gs[0, 1], sharex=ax, sharey=ax)
        if self.process.resultImage.numChannels > 1:
            ax.imshow(self.process.resultImage)
        else:
            ax.imshow(self.process.resultImage, cmap='afmhot')
        ax.set_axis_off()
        ax.set_title(f"{self.process.name} Result")

        ax = fig.add_subplot(gs[0, 2], sharex=ax, sharey=ax)
        ax.set_title('Resulting Components')
        ax.set_axis_off()
        gtColor = (0.0, 0.3, 1.0)
        tpColor = (0.0, 1.0, 0.3)
        fnColor = (1.0, 0.3, 0.0)
        handles = []
        if opencv:
            if imgAsGray:
                showImg = cv.cvtColor(self.process.image.asGrayScale(), cv.COLOR_GRAY2RGB)
            else:
                showImg = np.array(self.process.image)

            if len(self.comparisonResult.groundTruthComponents) > 0:
                plotting.drawComponents(self.comparisonResult.groundTruthComponents, image=showImg,
                                        color=np.array(gtColor) * 255.0)
                handles.append(patches.Patch(facecolor=gtColor, edgecolor='w', label='GT'))
            if len(self.comparisonResult.truePositiveComponents) > 0:
                plotting.drawComponents(self.comparisonResult.truePositiveComponents, image=showImg,
                                        color=np.array(tpColor) * 255.0)
                handles.append(patches.Patch(facecolor=tpColor, edgecolor='w', label='TP'))
            if len(self.comparisonResult.falseNeg) > 0:
                plotting.drawComponents(self.comparisonResult.falseNeg, image=showImg,
                                        color=np.array(fnColor) * 255.0)
                handles.append(patches.Patch(facecolor=fnColor, edgecolor='w', label='FN'))

            ax.imshow(showImg)
        else:
            if imgAsGray:
                ax.imshow(self.process.image.asGrayScale(), cmap='gray')
            else:
                ax.imshow(self.process.image)
            if len(self.comparisonResult.groundTruthComponents) > 0:
                h = plotting.plotComponents(self.comparisonResult.groundTruthComponents, polyColors=gtColor,
                                            label='GT', ax=ax)
                handles.append(h)
            if len(self.comparisonResult.truePositiveComponents) > 0:
                h = plotting.plotComponents(self.comparisonResult.truePositiveComponents, polyColors=tpColor,
                                            label='TP', ax=ax)
                handles.append(h)
            if len(self.comparisonResult.falseNeg) > 0:
                h = plotting.plotComponents(self.comparisonResult.falseNeg, polyColors=fnColor, label='FN',
                                            ax=ax)
                handles.append(h)

        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)

        if len(self.comparisonResult.falsePos) > 0:
            ax = fig.add_subplot(gs[1, 0], sharex=ax, sharey=ax)
            if opencv:
                if imgAsGray:
                    showImg = cv.cvtColor(self.process.image.asGrayScale(), cv.COLOR_GRAY2RGB)
                else:
                    showImg = np.array(self.process.image)
                plotting.drawComponents(self.comparisonResult.groundTruthComponents, image=showImg,
                                        color=np.array(fnColor) * 255.0)
                ax.imshow(showImg)
            else:
                if imgAsGray:
                    ax.imshow(self.process.image.asGrayScale(), cmap='gray')
                else:
                    ax.imshow(self.process.image)

                _ = plotting.plotComponents(self.comparisonResult.falsePos, polyColors=(1.0, 0.3, 0.0),
                                            label="false pos.", ax=ax)
            ax.set_title('False Positives')
            ax.set_axis_off()
            ax = fig.add_subplot(gs[1, 1:])
        else:
            ax = fig.add_subplot(gs[1, :])

        if len(scores) > 0:
            sb.distplot(scores, norm_hist=True, ax=ax, bins=100)

        iou_mean = np.mean(scores) if len(scores) > 0 else np.nan
        iou_std = np.std(scores) if len(scores) > 0 else np.nan
        ax.set_title('IOU Score Distribution')
        ax.set_xlabel('IoU Score')

        summaryString = f'{"True Positive Rate":20} {tp:>5.2f}'
        summaryString += f'\n{"Precision":20} {precision:>5.2f}'
        summaryString += f'\n{"Matched":20} {len(self.comparisonResult.truePositiveComponents):>5}'
        summaryString += f'\n{"False Neg":20} {len(self.comparisonResult.falseNeg):>5}'
        summaryString += f'\n{"False Pos":20} {len(self.comparisonResult.falsePos):>5}'

        ax.annotate(f'$\mu$', (iou_mean + .01, 0.7 * ax.get_ylim()[1]))
        ax.axvline(iou_mean, linestyle=':', c='gray')
        ax.axvline(iou_mean - iou_std, linestyle=':', c='r')
        ax.axvline(iou_mean + iou_std, linestyle=':', c='r')
        props = ax.annotate(summaryString, (.005, .8), xycoords='axes fraction', fontfamily='monospace')
        props.set_bbox(dict(facecolor='gray', alpha=0.5, edgecolor='darkgray'))

        # plt.suptitle(self.name, fontsize=14, fontweight='bold')
        # maximizePlot()

        imgName = self.process.image.name.replace(".", "_")
        savePath = saveDir.joinpath(
            f"{imgName}_{self.process.name}-components.png") if saveDir is not None else None
        plotting.saveShowFigure(fig, savePath)

    def summary(self, groupby=None, full=False):
        if groupby is None:
            groupby = ['class', 'detectionType']

        if full:
            summary = self.allAnalytics.groupby(groupby).describe()
        else:
            summary = self.allAnalytics.groupby(groupby).agg(
                count=('area', 'count'),
                max_area=('area', max),
                min_area=('area', min),
                mean_area=('area', 'mean'),
                comp_density=('area', sum)
            )
            summary['comp_density'] = summary['comp_density'] / self.process.image.area
        return summary

    def save(self, path: Path, overwrite=False):

        assert path.is_dir(), f"Save path ({path}) must be a directory"

        savepath = path.joinpath(self.filename)

        if overwrite or not savepath.exists():
            print(f"Saving to {savepath.as_posix()}...", end='')
            # pickled = pickle.dumps(self)
            pickled = cloudpickle.dumps(self)
            savepath.write_bytes(pickled)
            print('Done')
        else:
            print(f'File Exists -- {savepath}')

    @staticmethod
    def load(path: Path):
        print(f"Loading {path.as_posix()}...", end='')
        obj = cloudpickle.load(path.open('rb'))
        # obj = pickle.loads(path.read_bytes())
        print('Done')
        return obj


# https://paperpile.com/shared/tOM3UT
# Zhang, H., Fritts, J. E., & Goldman, S. A. (2008). Image segmentation evaluation: A survey of unsupervised methods. Computer Vision and Image Understanding: CVIU, 110(2), 260–280.
# Analyzing image segmentation methods
class ComponentAnalysis:
    """

    """

    # TODO: Implement segmentation metrics
    #   Cardoso, J. S., & Corte-Real, L. (2005). Toward a generic evaluation of image segmentation. IEEE Transactions on Image Processing: A Publication of the IEEE Signal Processing Society, 14(11), 1773–1782.
    #   https://paperpile.com/shared/U9UNmk  - may be useful in the future

    def __init__(self):
        pass

    # TODO: Wrap up image analytics
    @staticmethod
    def analyticsFor(processname: str, image: Image, components: List[Tuple[Component, str]]) -> pd.DataFrame:
        summaryTable = pd.DataFrame()
        for compTuple in tqdm.tqdm(components, desc=f"{processname} - Analytics"):
            comp, detectionType = compTuple
            cropped = image.cropped(comp.vertices.boundingBox)
            # imgStats = ComponentDetectionAnalysis.colorImageStatsFor(cropped)
            imgStats = pd.Series([comp.vertices.area,
                                  comp.vertices.height,
                                  comp.vertices.width,
                                  comp.deviceClass.lower() if comp.deviceClass is not None else None
                                  ],
                                 index=['area', 'height', 'width', 'class'])
            # imgStats = imgStats.append(geometry)
            if comp.regionprops is not None:
                regionSeries = pd.Series(
                    {k: comp.regionprops[k] for k in comp.regionprops if
                     pd.api.types.is_scalar(comp.regionprops[k])})
                regionSeries.rename(index={'area': 'pixel_area'}, inplace=True)
                regionSeries.drop(index='bbox_area', inplace=True)
                imgStats = imgStats.append(regionSeries)
            imgStats = imgStats.append(pd.Series({"detectionType": detectionType}))
            try:
                summaryTable = summaryTable.append(imgStats, ignore_index=True)
            except ValueError:
                newTable = pd.DataFrame()
                newTable = newTable.append(imgStats, ignore_index=True)
                summaryTable = newTable.append(summaryTable, ignore_index=True)
        return summaryTable


    @staticmethod
    def colorImageStatsFor(image: Image, mask=None):
        means, stds = cv.meanStdDev(image, mask=mask)
        lmeans, lstds = cv.meanStdDev(cv.Laplacian(image, cv.CV_8U), mask=mask)

        channels = ['c0', 'c1', 'c2']
        data = pd.Series()
        for ch, m, s, lm, ls in zip(channels, means, stds, lmeans, lstds):
            data[f"{ch}_mean"] = m[0]
            data[f"{ch}_stDev"] = s[0]
            data[f"{ch}_laplace_mean"] = lm[0]
            data[f"{ch}_laplace_stDev"] = ls[0]

        for chan, cname in enumerate(channels):
            # Histogram
            # data[f"{cname}_hist"] = ComponentDetectionAnalysis.histogramAnalysis(image, [chan], mask=mask)

            # Entropy
            data[f"{cname}_entropy"] = shannon_entropy(image[:, :, chan])

            # Gradient
            gmean, gStdDev = ComponentAnalysis.gradientAnalysis(image[:, :, chan], mask)
            data[f'{cname}_gradMean'] = gmean
            data[f'{cname}_gradStdDev'] = gStdDev

            # Texture (Grey Level Co-occurence Matrix)
            #   includes a measure of contrast
            glcm = ComponentAnalysis.glcmAnalysis(image[:, :, chan])
            for name, val in glcm.iteritems():
                data[f"{cname}_{name}"] = val

                # MAYBE: measure of texture
                #   Local Binary Pattern

        # TODO: measure of focus

        return data

    @staticmethod
    def maskFromComponents(comps: List[Component], imShape: tuple):
        labels, _ = ComponentAnalysis.labelImageFromComponents(comps, imShape)
        mask = labels > 0
        return mask

    @staticmethod
    def labelImageFromComponents(comps: List[Component], imShape: tuple):
        m = np.zeros(imShape[:2], np.uint8)
        metadata = pd.DataFrame()
        for c, comp in enumerate(comps):
            cv.drawContours(m, [comp.vertices], 0, c + 1, -1)
            metadata = metadata.append(comp.data, ignore_index=True)
        return m, metadata

    @staticmethod
    def glcmAnalysis(image: Image):

        assert len(image.shape) == 2, f"Image must be 2D ndarray"

        glcm = skfeature.greycomatrix(image, distances=[2, 2, 2, 2, 2],
                                      angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4,
                                              3 * np.pi / 2, 7 * np.pi / 4],
                                      normed=True)
        stats = pd.Series()
        for stat in ['dissimilarity', 'correlation', 'contrast', 'homogeneity', 'ASM', 'energy']:
            stats[stat] = skfeature.greycoprops(glcm, stat)[0, 0]

        return stats

    @staticmethod
    def histogramAnalysis(image: Image, chan: [int] = None, mask=None, normalize=False):

        assert image.dtype == np.uint8, "Image must be uint8"

        hist = cv.calcHist([image], chan, mask, [256], [0, 256])

        if normalize:
            hist = cv.normalize(hist, hist)

        return hist.T[0]

    @staticmethod
    def gradientAnalysis(image: Image, mask=None):
        assert len(image.shape) == 2, f"Image must be 2D ndarray"
        dx, dy = cv.spatialGradient(image, cv.CV_64F)
        dx, dy = dx.astype(np.float), dy.astype(np.float)
        mag = np.sqrt((dx * dx) + (dy * dy))
        dmeans, dstds = cv.meanStdDev(mag, mask=mask)

        return dmeans[0, 0], dstds[0, 0]


class ComponentDetection:
    """
    Component detection methods
    """

    defaultMinComponentArea = 6  # If integer, absolute pixel area, if float, then is percent of image area
    defaultMaxComponentArea = .5  # If integer, absolute pixel area, if float, then is percent of image area

    def __init__(self):
        pass

    @staticmethod
    def baselineCompare(image: Image,
                        annotations: ImageAnnotation,
                        savePath: Path = None,
                        minAreaThresh=defaultMinComponentArea, maxAreaThresh=defaultMaxComponentArea, plotStages=False):
        results: List[ComponentDetectionResult] = []

        def addResult(_res: ComponentDetectionResult):
            if savePath is not None:
                p = savePath.joinpath(_res.filename)
                if p.exists():
                    loaded = ComponentDetectionResult.load(p)
                    results.append(loaded)
                    return
            results.append(_res)

        # Otsu
        addResult(ComponentDetection.findComponentsOtsu(minAreaThresh=minAreaThresh,
                                                        maxAreaThresh=maxAreaThresh))

        # Edge Detection
        addResult(ComponentDetection.findComponentsEdgeDetection(minAreaThresh=minAreaThresh,
                                                                 maxAreaThresh=maxAreaThresh))
        # Adaptive thresholding
        addResult(ComponentDetection.findComponentsAdaptiveThresholding(minAreaThresh=minAreaThresh,
                                                                        maxAreaThresh=maxAreaThresh))

        # Watershed
        addResult(ComponentDetection.findComponentsWatershed(minAreaThresh=minAreaThresh,
                                                             maxAreaThresh=maxAreaThresh))

        # Graphcut
        addResult(ComponentDetection.findComponentsEfficientGraphCut(minAreaThresh=minAreaThresh,
                                                                     maxAreaThresh=maxAreaThresh))

        for res in results:
            res.process.run(ImageIO(image=image))
            res.compare(annotations)

        if plotStages:
            for res in results:
                res.plot()

        return results

    @staticmethod
    def findComponents(process: ImageProcess) -> ComponentDetectionResult:
        """

        :param process:
        :param minAreaThresh: to set must be included in process input ProcessIO keys
        :param maxAreaThresh: to set must be included in process input ProcessIO keys
        :return:
        """

        if not process.dirty:
            return ComponentDetectionResult(process, None)

        bwImage: Image = process.resultImage
        if bwImage.dtype == np.uint8:
            image_data = bwImage
        else:
            image_data = bwImage.astype(np.uint8)

        labels: Optional[np.ndarray] = None
        labelMetaData: Optional[pd.DataFrame] = None

        try:
            minAreaThresh = process.input['minAreaThresh']
        except KeyError:
            minAreaThresh = ComponentDetection.defaultMinComponentArea

        try:
            maxAreaThresh = process.input['maxAreaThresh']
        except KeyError:
            maxAreaThresh = ComponentDetection.defaultMaxComponentArea

        try:
            labels = process.result['labels']
        except KeyError:
            pass

        try:
            labelMetaData = process.result['labelMetadata']
        except KeyError:
            pass

        boxes = []

        try:

            if labels is None:
                _, labels, _, _ = cv.connectedComponentsWithStats(image_data, connectivity=8)
                image_area = bwImage.area
            else:
                image_area = labels.shape[0] * labels.shape[1]

            if labelMetaData is None:
                labelNums = np.unique(labels)
                labelNums.sort()
                labelMetaData = pd.DataFrame(data=[None] * len(labelNums), index=labelNums)

            regions = regionprops(labels)
            max_area = image_area * np.abs(maxAreaThresh) if np.abs(maxAreaThresh) <= 1.0 else maxAreaThresh
            min_area = image_area * np.abs(minAreaThresh) if np.abs(minAreaThresh) <= 1.0 else minAreaThresh
            for region in regions:
                # bbox = region.bbox
                if min_area < region.bbox_area <= max_area:
                    # region coords is (row, col), we need (col, row)
                    coords = np.roll(region.coords, 1, axis=1)
                    cnts, _ = cv.findContours((region.convex_image > 0).astype(np.uint8),
                                              cv.RETR_CCOMP,
                                              cv.CHAIN_APPROX_SIMPLE)

                    # Translate contour coordinates (region coordinates) back to image coordinates
                    coords = cnts[0][:, 0] + np.min(coords, axis=0)

                    # create component
                    c = Component(shape=coords, data=labelMetaData.loc[region.label - 1, :], regionprops=region)
                    boxes.append(c)
        except Exception as e:
            print(f"Error in finding components: {e}")
        return ComponentDetectionResult(process, boxes)

    @staticmethod
    def findComponentsOtsu(minAreaThresh=defaultMinComponentArea, maxAreaThresh=defaultMaxComponentArea,
                           smoothingSize=7) -> ComponentDetectionResult:
        io = ImageIO(_func=ComponentDetection.findComponentsOtsu, _setParams=locals())
        process = io.initProcess("Otsu's Thresholding")
        process.addProcess(alg.medianBlurProcess(ksize=smoothingSize))
        process.addProcess(alg.otsuThresholdProcess())
        process.addProcess(alg.binarizeImageProcess())
        process.addProcess(alg.morphologyExProcess(cv.MORPH_OPEN, ksize=9))
        # process.addProcess(alg.morphologyExProcess(cv.MORPH_CLOSE, ksize=11))
        return ComponentDetection.findComponents(process)

    # TODO: Adaptive Histogram Thresholding
    @staticmethod
    def findComponentsAdaptiveThresholding(minAreaThresh=defaultMinComponentArea,
                                           maxAreaThresh=defaultMaxComponentArea, ) -> ComponentDetectionResult:
        io = ImageIO(_func=ComponentDetection.findComponentsAdaptiveThresholding, _setParams=locals())
        process = io.initProcess("Adaptive Thresholding")
        process.addProcess(alg.adaptiveThresholdProcess())
        return ComponentDetection.findComponents(process)

    # TODO: Background subtraction
    @staticmethod
    def findComponentsBackgroundGaussianSubtraction(minAreaThresh=defaultMinComponentArea,
                                                    maxAreaThresh=defaultMaxComponentArea, ) -> ComponentDetectionResult:
        io = ImageIO(_func=ComponentDetection.findComponentsBackgroundGaussianSubtraction, _setParams=locals())
        process = io.initProcess("Gaussian Background Sub")
        process.addProcess(alg.gmmBackroundMask())
        return ComponentDetection.findComponents(process)

    @staticmethod
    def findComponentsGraphCut(image: Image, minAreaThresh=defaultMinComponentArea,
                               maxAreaThresh=defaultMaxComponentArea, ) -> ComponentDetectionResult:
        process = ImageIO(_func=ComponentDetection.findComponentsGraphCut, _setParams=locals()).initProcess(
            "GraphCut")
        process.addProcess(alg.graphCutSegmentation())
        return ComponentDetection.findComponents(process)

    @staticmethod
    def findComponentsEfficientGraphCut(minAreaThresh=defaultMinComponentArea,
                                        maxAreaThresh=defaultMaxComponentArea, ) -> ComponentDetectionResult:
        process = ImageIO(_func=ComponentDetection.findComponentsEfficientGraphCut,
                          _setParams=locals()).initProcess("Efficient GraphCut")
        process.addProcess(alg.efficientGraphSegmentation(scale=300,
                                                          sigma=.7,
                                                          min_size=600))
        return ComponentDetection.findComponents(process)

    @staticmethod
    def findComponentsWatershed(minAreaThresh=defaultMinComponentArea,
                                maxAreaThresh=defaultMaxComponentArea, ) -> ComponentDetectionResult:
        process = ImageIO(_func=ComponentDetection.findComponentsWatershed, _setParams=locals()).initProcess(
            "Watershed")
        process.addProcess(alg.watershedProcess())
        return ComponentDetection.findComponents(process)

    @staticmethod
    def findComponentsEdgeDetection(minAreaThresh=defaultMinComponentArea, maxAreaThresh=defaultMaxComponentArea, ):
        process = ImageIO(_func=ComponentDetection.findComponentsEdgeDetection,
                          _setParams=locals()).initProcess(
            "Edge Detection")
        process.addProcess(alg.edgeDetection())
        return ComponentDetection.findComponents(process)

    # TODO: Region-Based Methods
    #   Growing
    #   Splitting/Merging
    #   Multiobjective Data Clustering
    #   https://paperpile.com/shared/qZnIdD
    #   Law, M. H. C., Topchy, A. P., & Jain, A. K. (2004). Multiobjective data clustering. Proceedings of the 2004 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2004. CVPR 2004., 2, II – II.

    # TODO: Moderate Methods
    #   Characteristic Feature Clustering
    #   Fuzzy Techniques
    #   Color to Reflection Techniques

    # TODO: Advanced Methods
    #   Neural Networks (not yet)
    #   Physics-Based Methods
    #       dichromatic reflection model -- Shafer, S. A. (1985). Using color to separate reflection components. Color Research and Application, 10(4), 210–218.
    #           https://paperpile.com/shared/8h18Db
    #       approximate color-reflectance model -- Healey, G. (1989). Using color for geometry-insensitive segmentation. JOSA A, 6(6), 920–937.
    #           https://paperpile.com/shared/5z1yeE
    #   Generative Adversarial Networks (?)

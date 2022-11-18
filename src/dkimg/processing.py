from __future__ import annotations
from .common import Image, ImageException, ProgressBar, Undefined
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Dict, Callable, Any, Union, Tuple
from typing_extensions import Protocol, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import datetime as dt
import inspect
from abc import abstractmethod
import copy

@dataclass
class ProcessIO(dict):
    UNDEFINED = Undefined()

    def __repr__(self):
        return dict.__repr__(self)

    def __str__(self):
        return dict.__str__(self)

    def __init__(self, **kwargs):
        requiredInputKeys = kwargs.pop('requiredInputKeys', None)
        super().__init__(kwargs)

        if requiredInputKeys is None:
            requiredInputKeys = set()
        diff = requiredInputKeys - set(self.keys())
        if len(diff) > 0:
            # raise ImageException(f"Missing required keys: {diff}")
            for k in list(diff):
                self[k] = None
        self.requiredInputKeys = requiredInputKeys
        self.parsedKeys = {}

        _function = kwargs.get("_func", None)
        if _function is not None and isinstance(_function, Callable):
            _defvals = kwargs.get("_setParams", None)
            self.parseFrom(_function, _defvals)
            if "_setParams" in self.keys():
                self.pop("_setParams")
            self.pop('_func')
        else:
            pass

        # Output make should have form key, value where
        # key = required input key of next stage
        # value = key in self that maps to the value of interest
        self._outputMap = {}

    def parseFrom(self, func: Callable, setParams: Dict[str, Any] = None):
        spec = inspect.signature(func).parameters
        if setParams is None:
            setParams = {}
        for k, v in spec.items():
            nk = k.lstrip("_")
            self.parsedKeys[nk] = k
            try:
                self[nk] = setParams[nk]
            except KeyError:
                self[nk] = ProcessIO.UNDEFINED if v.default is v.empty else v.default

    @property
    def metadata(self):
        return self['metadata']

    @metadata.setter
    def metadata(self, v):
        self['metadata'] = v

    @property
    def outputMap(self):
        return self._outputMap

    @property
    def parsedParams(self):
        return {self.parsedKeys[k]: v for k, v in self.items()}

    @property
    def inputKeys(self):
        keys = set([k for k, v in self.items() if v is ProcessIO.UNDEFINED])
        keys.update(self.requiredInputKeys)
        return keys

    @property
    def hyperParamKeys(self):
        return set(self.keys()) - set(self.inputKeys) - self.requiredInputKeys

    @property
    def hyperParams(self):
        return {k: self[k] for k in self.hyperParamKeys}

    @property
    def display(self) -> Optional[Union[List[Image], Image]]:

        try:
            imname = self['image'].name
        except KeyError:
            imname = 'process'

        if 'display' in self.keys():
            displayKey = self['display']
            if displayKey is None:
                return None
            if not isinstance(displayKey, List):
                displayKey = [displayKey]
            imdata = []
            for i, k in enumerate(displayKey):
                try:
                    data = self[k]
                    if hasattr(data, 'name'):
                        name = data.name
                    else:
                        name = k
                    imdata.append(Image(src=data, name=name))
                except Exception:
                    pass
        elif 'image' in self.keys():
            imdata = self['image']
        else:
            return None

        if isinstance(imdata, np.ndarray):
            return Image(src=imdata, name=imname)
        elif isinstance(imdata, Image) or isinstance(imdata, List):
            return imdata
        return None

    def mapOutput(self, resultKey, nextInputKey):
        if resultKey not in self.keys():
            print(f"Attempting to map result key that doesn't exist in IO {resultKey}")
        else:
            self._outputMap[nextInputKey] = resultKey

    def resetOutputMap(self):
        self._outputMap = {}

    def initProcess(self, name):
        return Process(name=name, ioSpec=self)


@dataclass
class ImageIO(ProcessIO):

    @property
    def image(self) -> Image:
        return self["image"]

    @image.setter
    def image(self, i):
        self["image"] = i

    def __init__(self, **kwargs):

        # raise ImageException(f"Invalid initializer input {type(image)}")
        requiredInputKeys = kwargs.get('requiredInputKeys', None)
        if requiredInputKeys is None:
            requiredInputKeys = set()
        requiredInputKeys.add('image')
        kwargs['requiredInputKeys'] = requiredInputKeys

        super().__init__(**kwargs)

        if self['image'] is None:
            self.image = np.array([[]])

        if not isinstance(self['image'], Image):
            self['image'] = Image(self['image'])

    def initProcess(self, name):
        return ImageProcess(name=name, ioSpec=self)

    @staticmethod
    def fromPath(path: Path) -> ImageIO:
        if path.exists():
            return ImageIO(image=Image(path))
        else:
            raise ImageException(f"Path does not exist: {path.as_posix()}.")


ProcessFunction = Callable[[], ProcessIO]


@runtime_checkable
class ProcessStage(Protocol):
    name: str
    _input: ProcessIO = None
    inputSpec: ProcessIO = None
    stages: List[ProcessStage] = []
    allowDisable = False
    disabled = False

    def __repr__(self) -> str:
        selfCls = type(self)
        oldName: str = super().__repr__()
        # Remove module name for brevity
        oldName = oldName.replace(f'{selfCls.__module__}.{selfCls.__name__}',
                                  f'{selfCls.__name__} \'{self.name}\'')
        return oldName

    def __str__(self) -> str:
        return repr(self)

    @property
    def requiredInputs(self):
        return self.inputSpec.inputKeys

    @property
    @abstractmethod
    def runningTime(self) -> dt.timedelta:
        raise NotImplementedError

    @property
    @abstractmethod
    def result(self) -> Optional[ProcessIO]:
        raise NotImplementedError

    @property
    @abstractmethod
    def timeTable(self):
        raise NotImplementedError

    @property
    def input(self):
        if self._input is None:
            self._input = copy.copy(self.inputSpec)
        return self._input

    @property
    def dirty(self) -> bool:  # TODO: setters for input to flag somethig has changed
        return self.result is not None

    def __call__(self, io: ProcessIO, **kwargs):
        force = kwargs.get('force', False)
        disable = kwargs.get('disable', False)
        verbose = kwargs.get('verbose', True)
        self.run(io, force=force, disable=disable, verbose=verbose)

    def updateParams(self, **io):
        for k, v in io.items():
            if k in self.input.hyperParamKeys:
                self.input[k] = v

    def updateInput(self, io: ProcessIO):
        if self.input is None:
            self._input = self.inputSpec

        for k in self.requiredInputs:
            if k in io.keys():
                self._input[k] = io[k]
            elif k in io.outputMap.keys():
                self._input = io[io.outputMap[k]]
            else:
                ks = self.requiredInputs - set(io.keys())
                raise ImageException(
                    f"Missing input key {k} in result keys: ({ks}) and output map {io.outputMap.keys()}")

    def run(self, io: ProcessIO = None, force=False, disable=False, verbose=True) -> ProcessIO:
        raise NotImplementedError

    def iterInput(self):
        return iter(self.input)

    def printStageParameters(self):
        raise NotImplementedError

    def hasChildren(self):
        return len(self.stages) > 0

    def updateParamsForStage(self, name: str, **kwargs):
        """

        :param name:
        :return:
        """
        if self.name == name:
            self.updateParams(**kwargs)
            return

        for stage in self.stages:
            if stage.name == name:
                stage.updateParams(**kwargs)
                return
            else:
                stage.updateParamsForStage(name, **kwargs)


class AtomicProcess(ProcessStage):
    """
    Holds the function and the parameters to processs an image
    """

    @property
    def result(self) -> Optional[ProcessIO]:
        return self._result

    @property
    def runningTime(self) -> dt.timedelta:
        return self._processing_time

    @property
    def timeTable(self):
        return pd.DataFrame(data=[[self.name, str(self.runningTime)]], columns=['stage', 'time'])

    def __init__(self, function: ProcessFunction, name: str = None, **kwargs):
        self.function: ProcessFunction = function
        self.name = self.function.__name__ if name is None else name
        self.inputSpec = ImageIO(_func=function, _setParams=kwargs)
        self._result: Optional[ProcessIO] = None
        self._processing_time: dt.timedelta = dt.timedelta(0)

    def run(self, io: ProcessIO = None, force=False, disable=False, verbose=True) -> ProcessIO:
        if not force and self.dirty:
            return self.result

        self.updateInput(io)
        if disable:
            def runFunc(**_ins) -> ProcessIO:
                return io

            text = f'--- DISABLED [{self.function.__name__}] ---'
        else:
            runFunc = self.function
            text = f'Running {self.function.__name__}'

        assert self.input is not None, f"io cannot be none for an Atomic Function"
        # print(f'Running {self.function.__name__}', end=' -- ')
        waitingBar = None
        if verbose:
            waitingBar = ProgressBar()
            waitingBar.show(text)

        start = dt.datetime.now()
        self._result: ProcessIO = runFunc(**self.input.parsedParams)
        self._processing_time = dt.datetime.now() - start

        if waitingBar is not None:
            waitingBar.done()

        if verbose:
            print(f'\r{self.function.__name__} -- DONE ({self.runningTime})...')
        return self.result

    def printStageParameters(self):
        print(f"--{self.name}")
        print(f"spec  \t{self.inputSpec.hyperParams}")
        print(f"input \t{self.input.hyperParams}")

    # TODO: Pickling Protocol (currently reliant on cloudpickle)
    # def __getstate__(self):
    #     attr = inspect.getmembers(AtomicFunction, lambda a: a)
    #     attr = [a for a in attr if not ((a[0].startswith('__') and a[0].endswith('__')) or (a[0].startswith('_')))]
    #     attributes = {k: self.__getattribute__(k) for k, _ in attr}
    #     code = {k: v for k, v in inspect.getmembers(self.function, lambda x: x)}
    #     attributes['module'] = code['__module__']
    #     attributes['caller'] = code['__qualname__']
    #     attributes['function_name'] = code['__name__']
    #     attributes['function_source'] = inspect.cleandoc(inspect.getsource(self.function))
    #     return attributes
    #
    # def __setstate__(self, state):
    #     pass


class Process(ProcessStage):

    @classmethod
    def fromFunction(cls, function: ProcessFunction, name: str, **kwargs):
        proc = ProcessIO().initProcess(name)
        proc.addFunction(function, **kwargs)
        return proc

    @property
    def result(self) -> Optional[ProcessIO]:
        if len(self.stages) > 0:
            return self.stages[-1].result
        return None

    @property
    def runningTime(self):
        t = np.sum(np.array([s.runningTime for s in self.stages]))
        return t

    @property
    def timeTable(self) -> pd.DataFrame:
        table = None
        for stage in self.stages:
            table = stage.timeTable if table is None else pd.concat([table, stage.timeTable])
        table['a'] = self.name
        cs = [c for c in table.columns if 'stage' in c]
        colname = f"stage_{len(cs)}"
        table[colname] = self.name
        return table

    def __init__(self, name: str, ioSpec: Optional[ProcessIO] = None):
        self.name = name
        self.inputSpec = ioSpec if ioSpec is not None else ProcessIO()
        self.stages: List[ProcessStage] = []
        self.disabled = False
        self.allowDisable = True

    def printProcessingTimeTable(self):
        print(self.timeTable)

    def getResultForStepName(self, stepName: str) -> Optional[ProcessIO]:
        """
        Return result for result with stepName
        :param stepName:
        :return: Optional[StepResult]
        """
        for proc in self.stages:
            if proc.dirty and proc.name == stepName:
                return proc.result
        return None

    def getResultAtIdx(self, idx: int) -> Optional[ProcessIO]:
        """
        Return result for step at the passed index
        :param idx:
        :return:
        """
        if idx >= len(self.stages):
            return None
        return self.stages[idx].result

    def getResultNames(self) -> [str]:
        """
        Returns a string array of the result IDs in the process
        :return:
        """
        names = [step.name for step in self.stages]
        return names

    def addProcess(self, process: Process):
        if self.name is None:
            self.name = process.name
        self.stages.append(process)

    def addFunction(self, function: ProcessFunction, name: str = None, **kwargs):
        """
        Function to add a process step with parameters to the process

        :param name:
        :param function: a function that returns an StepResult taking `params` as input
        """

        step = AtomicProcess(function=function, name=name, **kwargs)

        count = len(list(filter(lambda n: n == step.name, [step.name.split("#")[0] for step in self.stages])))

        if count > 0:
            count += 1
            step.name = f"{step.name}#{count}"

        self.stages.append(step)
        return step

    def run(self, io: ProcessIO = None, force=False, disable=False, verbose=True) -> ProcessIO:
        if io is None:
            _activeIO = self.input
        else:
            _activeIO = copy.copy(io)
            self.updateInput(io)

        for i, stage in enumerate(self.stages):
            newIO = stage.run(_activeIO, force=force, disable=self.disabled or disable, verbose=verbose)
            _activeIO.update(newIO)

        return self.result

    def printStageParameters(self):
        print(f"--\nProcess: {self.name}\n--")
        print(f"spec  \t{self.inputSpec.hyperParams}")
        print(f"input \t{self.input.hyperParams}")
        for stage in self.stages:
            stage.printStageParameters()

    def stagesFlattened(self):
        # Traverse stage tree
        def getProcessStages(_stages: List[ProcessStage]) -> List[ProcessStage]:
            _results: List[ProcessStage] = []
            for step in _stages:
                if isinstance(step, Process):
                    _results.extend(getProcessStages(step.stages))
                else:
                    _results.append(step)
            return _results

        return getProcessStages(self.stages)

    # TODO: update/refactor saving of results with serialized params for each step etc.


class ImageProcess(Process):
    """
    Encapsulates a series of processing steps with methods to save and plot the results
    """

    input: ImageIO

    @staticmethod
    def fromFunction(function: ProcessFunction, name: str, **kwargs):
        proc = ImageIO().initProcess(name)
        proc.addFunction(function, **kwargs)
        return proc

    @property
    def image(self):
        return self.input.image

    @property
    def resultImage(self) -> Optional[Image]:
        return self.image if self.result is None else ImageIO(**self.result).image

    def __init__(self, name: str, ioSpec: Optional[ImageIO] = None):
        io = ImageIO() if ioSpec is None else ioSpec
        super().__init__(name=name, ioSpec=io)

    def _plotStagesMatplotlib(self, stages: List[Tuple[str, Image]], figsize=(20,15), saveDir: Path = None, name: str = None, maxCols=4):
        numStages = len(stages)
        ncols = min(numStages, maxCols)
        nrows = int(np.ceil(numStages / ncols))
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=4./72, h_pad=4./72, hspace=0, vspace=0)

        gs = plt.GridSpec(nrows, ncols, figure=fig)

        recentAxis = None

        stageIdx = 0

        ax0 = None
        for r in range(nrows):
            for c in range(ncols):

                if stageIdx >= numStages:
                    continue

                title, display = stages[stageIdx]

                if stageIdx == 0:
                    ax = fig.add_subplot(gs[r, c])
                    ax0 = ax
                elif display.shape[:2] == self.input.image.shape[:2]:
                    ax = fig.add_subplot(gs[r, c], sharex=ax0, sharey=ax0)
                else:
                    if recentAxis is not None and recentAxis[1][:2] == display.shape[:2]:
                        ax = fig.add_subplot(gs[r, c], sharex=recentAxis[0], sharey=recentAxis[0])
                    else:
                        ax = fig.add_subplot(gs[r, c])
                        recentAxis = (ax, display.shape)

                if display.numChannels == 1:
                    ax.imshow(display, cmap='afmhot')
                else:
                    ax.imshow(display)
                ax.set_title(title)
                ax.set_axis_off()
                stageIdx += 1

        if name is not None:
            plt.suptitle(name)

        # maximizePlot()

        if saveDir is None:
            # plt.tight_layout()
            plt.show()
        else:
            # plt.tight_layout()
            imgName = self.image.name.replace(".", "_")
            plt.savefig(saveDir.joinpath(f"{imgName}_{self.name}_-stages.png").as_posix())
            plt.close()

    # def _plotStagesBokeh(self, stages: List[Tuple[str, Image]], saveDir: Path = None, name: str = None, maxCols=4):
    #     hv.renderer('bokeh')
    #
    #     p = bplotting.figure()


    def plotStages(self: ImageProcess,
                   saveDir: Path = None,
                   name: str = None,
                   figsize=(20,10),
                   maxCols=4,
                   ignoreDuplicateResults=False):

        stages: List[Tuple[str, Image]] = [(f'Original Image ({self.input.image.name})', self.input.image)]

        for stage in self.stagesFlattened():
            if stage.result.display is None:
                continue

            if isinstance(stage.result.display, List):
                for i, d in enumerate(stage.result.display):
                    stages.append((f"{stage.name} ({d.name})", d))
            else:
                stages.append((stage.name, stage.result.display))

        removeIdxs = []
        if ignoreDuplicateResults:
            lastNewResult = stages[0][1]
            # Take out stages whose results are the same as the previous result
            for ii, stage in enumerate(stages[1:], 1):
                curResult = stage[1]
                if np.array_equal(lastNewResult, curResult):
                    removeIdxs.append(ii)
                else:
                    lastNewResult = curResult
            for remIdx in reversed(removeIdxs):
                del stages[remIdx]

        if len(stages) == 0:
            print(f"Display stages for {self.name} is empty.....")
            return

        self._plotStagesMatplotlib(stages, figsize=figsize, saveDir=saveDir, name=name, maxCols=maxCols)


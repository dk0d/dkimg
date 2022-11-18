from __future__ import annotations
from enum import Enum


class IntersectionType(Enum):
    equalOffset = 'equalOffset'
    overOffset = 'overOffset'
    underOffset = 'underOffset'
    innerMatch = 'innerMatch'
    outerMatch = 'outerMatch'


class MatchColumns(Enum):
    groundTruth = 'groundTruth'
    truePositive = 'truePositive'
    iouScore = 'iouScore'
    intersectType = 'intersectType'

    @staticmethod
    def allValues():
        return list(map(lambda x: str(x).split('.')[-1], MatchColumns))

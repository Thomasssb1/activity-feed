from model import Model
import numpy as np
import math
from datetime import datetime


class CandidateModel:
    def __init__(self, classificationModel: Model, trainingData: np.ndarray):
        self.classificationModel = classificationModel

    @staticmethod
    def reduce(inputData: np.ndarray, language: str) -> np.ndarray:
        """Reduces the input dataset to 50,000 using a simple reduction approach

        Args:
            inputData: input test data
            language: target language
        Returns:
            ndarray: filtered input data
        """
        MINIMUM_RELEVANCE = 10  # days
        INITIAL_RELEVANCE = 100  # days

        def timeDecayFunc(timestamp: float, epoch: datetime) -> float:
            days = (epoch - datetime.fromtimestamp(timestamp)).days
            return INITIAL_RELEVANCE / 1 + math.log(1 + days) + MINIMUM_RELEVANCE

        now = datetime.now()
        decayIndices = np.argsort(
            np.array([timeDecayFunc(data[0], epoch=now) for data in inputData])
        )[:50_000]
        return inputData[decayIndices]

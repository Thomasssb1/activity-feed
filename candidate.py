from model import Model
import numpy as np
import math
from datetime import datetime


class CandidateModel:
    def __init__(self, classificationModel: Model, trainingData: np.ndarray):
        self.classificationModel = classificationModel

    @staticmethod
    def getTimeDecay(inputData: np.ndarray) -> np.ndarray:
        """Applies a time decay function to the input data, and returns each data
        points decay values in the shape (n, 1).

        Args:
            inputData: input data to apply time decay to
        Returns:
            ndarray: time decayed data
        """
        MINIMUM_RELEVANCE = 10  # days
        INITIAL_RELEVANCE = 100  # days

        def timeDecayFunc(timestamp: float, epoch: datetime) -> float:
            days = (epoch - datetime.fromtimestamp(timestamp)).days
            return (INITIAL_RELEVANCE / (1 + math.log(1 + days))) + MINIMUM_RELEVANCE

        now = datetime.now()
        return np.array(
            [timeDecayFunc(data[0], epoch=now) for data in inputData]
        ).reshape(-1, 1)

    @staticmethod
    def getLocationMissing(inputData: np.ndarray) -> np.ndarray:
        """Returns a binary array indicating whether the location is missing for
        each data point

        Args:
            inputData: input data
        Returns:
            ndarray: binary array indicating whether the location is missing
        """
        return np.where((inputData[:, 1] == 0) & (inputData[:, 2] == 0), 0, 1).reshape(
            -1, 1
        )

    @staticmethod
    def getDistance(
        inputData: np.ndarray, pos: tuple[float, float], epsilon: float = 1e-10
    ) -> np.ndarray:
        """Returns the normalised distance of each data point to the target location

        Args:
            inputData: input data
            pos: (lat, lon) tuple containing the target location, i.e. user location
        Returns:
            ndarray: distance array
        """
        points = inputData[:, 1:3]
        difference = np.array(points - np.array([pos[0], pos[1]]), dtype=np.float64)

        distances = np.linalg.norm(difference, axis=1).reshape(-1, 1)
        return np.maximum(distances, epsilon)

    @staticmethod
    def reduce(inputData: np.ndarray) -> np.ndarray:
        """Reduces the input dataset to 50,000 using a simple reduction approach.
        Also appends a time decay factor feature to each data point

        Args:
            inputData: input test data
        Returns:
            ndarray: filtered input data
        """
        decayArray = CandidateModel.getTimeDecay(inputData)
        decayIndices = np.argsort(decayArray, axis=0).flatten()[:50_000]

        return np.hstack([inputData, decayArray])[decayIndices]

    @staticmethod
    def normalise(inputData: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
        # calculate the per column maximum and minimum values
        maximum = np.max(inputData, axis=0)
        minimum = np.min(inputData, axis=0)
        difference = maximum - minimum
        difference[difference == 0] = epsilon
        return (inputData - minimum) / difference

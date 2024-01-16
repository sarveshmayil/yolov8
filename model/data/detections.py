# Based on roboflow/supervision Detection class

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np

from .utils import non_max_suppression, validate_detections_fields


@dataclass
class Detections:
    """
    A dataclass representing detection results.

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape
            `(n, H, W)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
    """

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None

    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            mask=self.mask,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id
        )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int]
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None
            )

    def __eq__(self, other: Detections) -> bool:
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.mask, other.mask),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id)
            ]
        )

    @classmethod
    def from_yolo(cls, yolo_results) -> Detections:
        """
        Creates a Detections instance from a YOLO inference result.

        Args:
            yolo_results:
                The output predictions from YOLO model

        Returns:
            Detections: A new Detections object.
        """
        yolo_detections_predictions = yolo_results.detach().cpu().numpy()

        return cls(
            xyxy=yolo_detections_predictions[:, :4],
            confidence=yolo_detections_predictions[:, 4],
            class_id=yolo_detections_predictions[:, 5].astype(int),
        )

    @classmethod
    def empty(cls) -> Detections:
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object. If all elements in a field are not
        `None`, the corresponding field will be stacked.
        Otherwise, the field will be set to `None`.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.
        """
        if len(detections_list) == 0:
            return Detections.empty()

        for detections in detections_list:
            validate_detections_fields(
                xyxy=detections.xyxy,
                mask=detections.mask,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=detections.tracker_id
            )

        xyxy = np.vstack([d.xyxy for d in detections_list])

        def stack_or_none(name: str):
            if all(d.__getattribute__(name) is None for d in detections_list):
                return None
            if any(d.__getattribute__(name) is None for d in detections_list):
                raise ValueError(f"All or none of the '{name}' fields must be None")
            return (
                np.vstack([d.__getattribute__(name) for d in detections_list])
                if name == "mask"
                else np.hstack([d.__getattribute__(name) for d in detections_list])
            )

        mask = stack_or_none("mask")
        confidence = stack_or_none("confidence")
        class_id = stack_or_none("class_id")
        tracker_id = stack_or_none("tracker_id")

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id
        )

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union[Detections, List, np.ndarray, None]:
        """
        Get a subset of the Detections object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the Detections or an item from the data.

        Returns:
            Union[Detections, Any]: A subset of the Detections object or an item from
                the data field.
        """
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None
        )

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each detection in the set of object detections.
        If masks field is defined property returns are of each mask.
        If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection
            in the format of `(area_1, area_2, ..., area_n)`,
            where n is the number of detections.
        """
        if self.mask is not None:
            return np.array([np.sum(mask) for mask in self.mask])
        else:
            return self.box_area

    @property
    def box_area(self) -> np.ndarray:
        """
        Calculate the area of each bounding box in the set of object detections.

        Returns:
            np.ndarray: An array of floats containing the area of each bounding
                box in the format of `(area_1, area_2, ..., area_n)`,
                where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])

    def with_nms(
        self, threshold: float = 0.5, class_agnostic: bool = False
    ) -> Detections:
        """
        Perform non-maximum suppression on the current set of object detections.

        Args:
            threshold (float, optional): The intersection-over-union threshold
                to use for non-maximum suppression. Defaults to 0.5.
            class_agnostic (bool, optional): Whether to perform class-agnostic
                non-maximum suppression. If True, the class_id of each detection
                will be ignored. Defaults to False.

        Returns:
            Detections: A new Detections object containing the subset of detections
                after non-maximum suppression.

        Raises:
            AssertionError: If `confidence` is None and class_agnostic is False.
                If `class_id` is None and class_agnostic is False.
        """
        if len(self) == 0:
            return self

        assert (
            self.confidence is not None
        ), "Detections confidence must be given for NMS to be executed."

        if class_agnostic:
            predictions = np.hstack((self.xyxy, self.confidence.reshape(-1, 1)))
            indices = non_max_suppression(
                predictions=predictions, iou_threshold=threshold
            )
            return self[indices]

        assert self.class_id is not None, (
            "Detections class_id must be given for NMS to be executed. If you intended"
            " to perform class agnostic NMS set class_agnostic=True."
        )

        predictions = np.hstack(
            (self.xyxy, self.confidence.reshape(-1, 1), self.class_id.reshape(-1, 1))
        )
        indices = non_max_suppression(predictions=predictions, iou_threshold=threshold)
        return self[indices]
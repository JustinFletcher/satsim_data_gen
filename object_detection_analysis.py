"""
author: Justin Fletcher
date: 7 Oct 2018
"""

from __future__ import absolute_import, division, print_function

import os
import json
import argparse
import itertools
import numpy as np
import pandas as pd
from numpy import linalg as la
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class ObjectDetectionAnalysis(ABC):
    def __init__(self,
                 truth_boxes,
                 inferred_boxes,
                 confidence_thresholds=None):
        """
        Abstract base class constructor, as well as executor of core analysis.
        Relies on polymorphic analyse_detections() function which should be
        implemented by the child class. Additionally expects
        compute_statistics() to be implemented by the child class.

        :param truth_boxes: list of truth boxes
        :param inferred_boxes: list of predicted boxes
        :param confidence_thresholds: list of desired confidence thresholds to
               use in the analysis
        """
        super().__init__()
        self.truth_boxes = truth_boxes
        self.inferred_boxes = inferred_boxes

        # If no confidence thresholds are provided, sample some logistically.
        if confidence_thresholds is None:
            confidence_thresholds = sigmoid(np.linspace(-100, 100, 100))
            confidence_thresholds = np.concatenate([[0.0],
                                                    confidence_thresholds
                                                    [1.0]],
                                                   axis=0)
        self.confidence_thresholds = np.unique(confidence_thresholds)

        # Create a dict to map the analyses to images.
        confidence_analysis = list()

        # Iterate over each image in the dataset, and evaluate performance.
        for image_number, image_name in enumerate(self.truth_boxes.keys()):
            # Run the analysis on this image.
            analyses = self.analyse_detections(image_name)

            # Concatenate the results
            confidence_analysis += analyses
        self.confidence_analysis = confidence_analysis

    @abstractmethod
    def analyse_detections(self, image_name):
        """
        Should be implemented by child class to create the analyzed results.

        :param image_name: the image currently being analyzed
        :return: a list of analysis results for this image
        """
        pass

    @abstractmethod
    def compute_statistics(self, statistics_dict=None):
        """
        Convert analysis results to DataFrame and comnpute statistics.

        :param statistics_dict: desired statistic (key-value=name-function)
        :return: pandas DataFrame containing results
        """
        pass

    @staticmethod
    def _precision(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives" and "false_postives"
        and returns the precision value.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of precision values for each TP/FN value in the input
        """
        tp = detection_counts_dict["true_positives"]
        fp = detection_counts_dict["false_positives"]

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 1.0
        return precision

    @staticmethod
    def _recall(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives" and "false_negatives"
        and returns the recall value.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of recall values for each TP/FN value in the input
        """
        tp = detection_counts_dict["true_positives"]
        fn = detection_counts_dict["false_negatives"]

        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0
        return recall

    @staticmethod
    def _f1(detection_counts_dict):
        """
        Accepts a dict containing keys "true_positives", "false_positives", and
        "false_negatives" and returns the F1 score.

        :param detection_counts_dict: dictionary containing TP and FN counts
        :return: list of F1 values for each TP/FN value in the input
        """
        precision = ObjectDetectionAnalysis._precision(detection_counts_dict)
        recall = ObjectDetectionAnalysis._recall(detection_counts_dict)

        if (precision + recall) != 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        return f1

    @staticmethod
    def _filter_boxes_by_confidence(inferred_boxes,
                                    confidence_threshold):
        """
        Helper function to filter out any boxes that don't achieve a desired confidence score.

        :param inferred_boxes: list of inferred boxes
        :param confidence_threshold: confidence threshold below which we should filter out
        :return: the filtered version of inferred_boxes
        """
        filtered_boxes = list()

        # Iterate over each inferred box...
        for inferred_box in inferred_boxes:

            # ...check if the confidence of this box exceeds the threshold...
            if inferred_box["confidence"] > confidence_threshold:

                # If it does, add it to the filtered boxes, which are returned.
                if confidence_threshold == 1.0:
                    print("Somehow we got a match with confidence = 0")
                    print("score = " + str(inferred_box["confidence"]))
                filtered_boxes.append(inferred_box["box"])

        return filtered_boxes


class ObjectDetectionAnalysisIOU(ObjectDetectionAnalysis):
    def __init__(self,
                 truth_boxes,
                 inferred_boxes,
                 confidence_thresholds=None,
                 iou_thresholds=None):
        """
        Constructor for the IOU-based Object Detection Analysis. This is actually
        the original version of this code, though now seperated from the base class
        to permit different forms of detection analysis to leverage the same
        construct.

        Here the intersect-over-union is used along with confidence as the main
        matching criteria.

        :param truth_boxes: list of ground truth boxes
        :param inferred_boxes: list of predicted boxes
        :param confidence_thresholds: list of desired confidence thresholds
        :param iou_thresholds: list of desired IOU thresholds
        """
        # If no IOU thresholds are provided, sample some linearly.
        if iou_thresholds is None:
            self.iou_thresholds = np.linspace(0.01, 0.99, 5)
        else:
            self.iou_thresholds = iou_thresholds

        # Call the base class to execute the analysis
        super().__init__(truth_boxes=truth_boxes,
                         inferred_boxes=inferred_boxes,
                         confidence_thresholds=confidence_thresholds)

    def analyse_detections(self, image_name):
        """
        One of the two abstract methods implemented by this child class. This is called
        by the base class when it is constructed to perform the bulk of the analysis
        computation. In this case that amounts to counting TPs, FPs, and FNs for the
        boxes predicted at different IOU thresholds and confidence thresholds.

        :param image_name: the name of the image currently under consideration
        :return: a list of lists intended to be transformed into a pandas DataFrame
        """
        # First get all of our inputs from the class's members
        truth_boxes = self.truth_boxes[image_name]
        inferred_boxes = self.inferred_boxes[image_name]
        confidence_thresholds = self.confidence_thresholds
        iou_thresholds = self.iou_thresholds

        # Instantiate a list to hold design-performance points.
        design_points = list()

        # Iterate over each combination of confidence and IoU threshold.
        for (iou_threshold,
             confidence_threshold) in itertools.product(iou_thresholds,
                                                        confidence_thresholds):

            # Compute the foundational detection counts at this design point.
            counts_dict = self._compute_detection_counts(truth_boxes,
                                                         inferred_boxes,
                                                         iou_threshold,
                                                         confidence_threshold)

            # Add this IoU threshold and confidence threshold to counts_dict.
            counts_dict["iou_threshold"] = iou_threshold
            counts_dict["confidence_threshold"] = confidence_threshold

            # Make a list image name and five always-present data values
            data_line = [image_name,
                         counts_dict["iou_threshold"],
                         counts_dict["confidence_threshold"],
                         counts_dict["true_positives"],
                         counts_dict["false_positives"],
                         counts_dict["false_negatives"]]

            # Add this design point to the list.
            design_points.append(data_line)
        return design_points

    def _compute_detection_counts(self,
                                  truth_boxes,
                                  inferred_boxes,
                                  iou_threshold,
                                  confidence_threshold):
        """
        Helper function to the analyze routine. Really this is most of the heavy lifting of
        the analysis: comparing boxes and determining which ones are matched.

        :param truth_boxes: list of ground truth boxes
        :param inferred_boxes: list of predicted boxes
        :param iou_threshold: the IOU threshold currently being used
        :param confidence_threshold: the confidence threshold currently being used
        :return: dictionary of TPs, FPs, and FNs
        """
        # First, remove from the inferred boxes all boxes below conf threshold.
        inferred_boxes = ObjectDetectionAnalysisIOU._filter_boxes_by_confidence(inferred_boxes,
                                                                                confidence_threshold)

        # If there are no inferred boxes, all truth boxes are false negatives.
        if not inferred_boxes:
            true_postives = 0
            false_positives = 0
            false_negatives = len(truth_boxes)
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # If there are no truth boxes, all inferred boxes are false positives.
        if not truth_boxes:
            true_postives = 0
            false_positives = len(inferred_boxes)
            false_negatives = 0
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # Declare a list to hold inferred-truth-IoU triples.
        overlaps = list()

        # For each combination of inferred and truth boxes...
        for inferred_box, truth_box in itertools.product(inferred_boxes,
                                                         truth_boxes):
                # Compute the IoU.
                iou = self._iou(inferred_box, truth_box)

                # If the IoU exceeds the required threshold, append overlap.
                if iou > iou_threshold:
                    overlaps.append([inferred_box, truth_box, iou])

        # If no confidence-filtered boxes have enough IoU with the truth boxes.
        if not overlaps:

            # Serious mistake in original code! Reproduced here.
            true_postives = 0
            false_positives = 0
            false_negatives = len(truth_boxes)

            # Alternative, correct, definition:
            # true_postives = 0
            # false_positives = len(pred_boxes)
            # false_negatives = len(truth_boxes)

        # Otherwise, if at least one box meet IoU with the truth boxes.
        else:

            matched_inferred_boxes = list()
            matched_truth_boxes = list()
            overlaps = sorted(overlaps, key=lambda x: x[2], reverse=True)

            # Iterate over overlaps...
            for i, [inferred_box, truth_box, _] in enumerate(overlaps):

                # ...and if neither box in this overlapping pair is matched...
                if list(inferred_box) not in matched_inferred_boxes:

                    if list(truth_box) not in matched_truth_boxes:

                        # ...match both by adding them to their matched lists.
                        matched_inferred_boxes.append(list(inferred_box))
                        matched_truth_boxes.append(list(truth_box))

            # The number of true positives is the number of matched boxes.
            true_postives = len(matched_truth_boxes)

            # The number of false positives is the excess of inferrered boxes.
            false_positives = len(inferred_boxes) - len(matched_inferred_boxes)

            # The number of false negatives is the excess of truth boxes.
            false_negatives = len(truth_boxes) - len(matched_truth_boxes)

        return {'true_positives': true_postives,
                'false_positives': false_positives,
                'false_negatives': false_negatives}

    def compute_statistics(self, statistics_dict=None):
        """
        Function to convert the results of analyse_detections into a meaningful
        pandas DataFrame. This includes the literal conversion, as well as the
        application of desired statistics functions to that DataFrame.

        :param statistics_dict: if the user desires non-standard statistics, they should provide them here
        :return: the resulting DataFrame, grouped by IOU and confidence thresholds.
        """
        df_header = [
            "image_name",
            "iou_threshold",
            "confidence_threshold",
            "true_positives",
            "false_positives",
            "false_negatives"]

        # Build the confidence analysis into a dataframe.
        analysis_df = pd.DataFrame(self.confidence_analysis,
                                   columns=df_header)

        # First, if no statistic function dict is provided, use the defualt.
        if statistics_dict is None:
            statistics_dict = {"precision": ObjectDetectionAnalysisIOU._precision,
                               "recall": ObjectDetectionAnalysisIOU._recall,
                               "f1": ObjectDetectionAnalysisIOU._f1}

        data = analysis_df[["true_positives",
                            "false_positives",
                            "false_negatives",
                            "confidence_threshold",
                            "iou_threshold"]]

        # Sum the data over images by confidence and IoU.
        grouped = data.groupby(["iou_threshold", "confidence_threshold"]).sum()

        # Iterate over each statistic function.
        for statisitic_name, statistic_fn in statistics_dict.items():
            # Apply this statistic function across the dataframe.
            grouped[statisitic_name] = grouped.apply(statistic_fn, axis=1)

        return grouped

    def plot_pr_curve(self, iou_thresholds):
        """
        Override me with better plotting.

        :param iou_thresholds:
        :return:
        """
        df = self.compute_statistics()

        # Start a new plot.
        ax = plt.gca()

        # Iterate over each unique IoU threshold.
        for iou_threshold in iou_thresholds.unique():

            # Get only the rows where the IoU threshold is this IoU threshold.
            ax.scatter(df.xs(iou_threshold, level=0)["recall"],
                       df.xs(iou_threshold, level=0)["precision"],
                       label=str(iou_threshold))

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.set_xlim([0.0, 1.2])
        ax.set_ylim([0.0, 1.2])
        plt.show()

    @staticmethod
    def _iou(pred_box, gt_box):
        """
        Calculate IoU of single predicted and ground truth box

        :param pred_box: location of predicted object as [xmin, ymin, xmax, ymax]
        :param gt_box: location of ground truth object as [xmin, ymin, xmax, ymax]
        :return: value of the IoU for the two boxes.
        """
        x1_t, y1_t, x2_t, y2_t = gt_box
        x1_p, y1_p, x2_p, y2_p = pred_box

        if (x1_p > x2_p) or (y1_p > y2_p):
            raise AssertionError(
                "Prediction box is malformed? pred box: {}".format(pred_box))
        if (x1_t > x2_t) or (y1_t > y2_t):
            raise AssertionError(
                "Ground Truth box is malformed? true box: {}".format(gt_box))

        if (x2_t < x1_p) or (x2_p < x1_t) or (y2_t < y1_p) or (y2_p < y1_t):
            return 0.0

        far_x = np.min([x2_t, x2_p])
        near_x = np.max([x1_t, x1_p])
        far_y = np.min([y2_t, y2_p])
        near_y = np.max([y1_t, y1_p])

        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
        pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
        return iou


class ObjectDetectionAnalysisL2N(ObjectDetectionAnalysis):
    def __init__(self,
                 truth_boxes,
                 inferred_boxes,
                 confidence_thresholds,
                 l2n_thresholds):
        # We need to couple with the SatNet sensor parameters somehow
        # self.path_to_satnet = "C:\\Users\\imcquaid\\Documents\\Datasets\\satnet\\SatNet.v.1.0.0.0\\SatNet\\data"
        self.path_to_satnet = "/opt/tfrecords/SatNet.v.1.0.0.0/SatNet/data"

        # If no L2N thresholds are provided, sample some linearly
        if l2n_thresholds is None:
            self.l2n_thresholds = np.arange(1, 10)
        else:
            self.l2n_thresholds = l2n_thresholds

        # Call the base class to execute the analysis
        super().__init__(truth_boxes=truth_boxes,
                         inferred_boxes=inferred_boxes,
                         confidence_thresholds=confidence_thresholds)

    def analyse_detections(self, image_name):
        # Some nonsense to get our annotations loaded
        if type(image_name) is str:
            # On my (Windows) machine this seems to be the control-flow
            converted_image_name = image_name
        else:
            # When used in the model validation loop, this seems to happen
            converted_image_name = image_name.decode()
        folder_name = converted_image_name.split("_")[0]
        file_name = "_".join(converted_image_name.split("_")[1:]) + ".json"
        annotation_path = os.path.join(self.path_to_satnet, folder_name, "Annotations", file_name)

        # Read the file
        with open(annotation_path, 'rb') as handle:
            annotation_dict = json.load(handle)["data"]
            sensor_props = annotation_dict["sensor"]
            handle.close()

        # Now extract sensor parameters
        sensor_width = sensor_props["width"]
        sensor_height = sensor_props["height"]
        # sensor_ifov_x = sensor_props["iFOVy"]
        # sensor_ifov_y = sensor_props["iFOVx"]
        # TODO: temporary fix to convert full FOV in deg to iFOV in arcseconds
        sensor_ifov_x = sensor_props["iFOVy"] * 3600.0 / sensor_height
        sensor_ifov_y = sensor_props["iFOVx"] * 3600.0 / sensor_width

        # First get all of our inputs from the class's members
        truth_boxes = self.truth_boxes[image_name]
        inferred_boxes = self.inferred_boxes[image_name]
        truth_centroids = [ObjectDetectionAnalysisL2N._convert_box_to_centroid(box,
                                                                               image_size=[sensor_height, sensor_width],
                                                                               ifov=[sensor_ifov_y, sensor_ifov_x],
                                                                               is_prediction=False) for box in truth_boxes]
        inferred_centroids = [ObjectDetectionAnalysisL2N._convert_box_to_centroid(box,
                                                                                  image_size=[sensor_height, sensor_width],
                                                                                  ifov=[sensor_ifov_y, sensor_ifov_x],
                                                                                  is_prediction=True) for box in inferred_boxes]
        confidence_thresholds = self.confidence_thresholds
        l2n_thresholds = self.l2n_thresholds

        # Instantiate a list to hold design-performance points
        design_points = list()

        # Iterate over each combination of confidence and IoU threshold
        for (l2n_threshold,
             confidence_threshold) in itertools.product(l2n_thresholds,
                                                        confidence_thresholds):

            # Compute the foundational detection counts at this design point
            counts_dict = self._compute_detection_counts(truth_centroids,
                                                         inferred_centroids,
                                                         l2n_threshold,
                                                         confidence_threshold)

            # Add this L2N threshold and confidence threshold to counts_dict
            counts_dict["l2n_threshold"] = l2n_threshold
            counts_dict["confidence_threshold"] = confidence_threshold

            # Make a list image name and five always-present data values
            data_line = [image_name,
                         counts_dict["l2n_threshold"],
                         counts_dict["confidence_threshold"],
                         counts_dict["true_positives"],
                         counts_dict["false_positives"],
                         counts_dict["false_negatives"]]

            # Add this design point to the list
            design_points.append(data_line)

        return design_points

    def _compute_detection_counts(self,
                                  truth_centroids,
                                  inferred_centroids,
                                  l2n_threshold,
                                  confidence_threshold):

        # First, remove from the inferred boxes all boxes below conf threshold.
        inferred_centroids = ObjectDetectionAnalysisL2N._filter_boxes_by_confidence(inferred_centroids,
                                                                                    confidence_threshold)

        # If there are no inferred boxes, all truth boxes are false negatives.
        if not inferred_centroids:
            true_postives = 0
            false_positives = 0
            false_negatives = len(truth_centroids)
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # If there are no truth boxes, all inferred boxes are false positives.
        if not truth_centroids:
            true_postives = 0
            false_positives = len(inferred_centroids)
            false_negatives = 0
            return {'true_positives': true_postives,
                    'false_positives': false_positives,
                    'false_negatives': false_negatives}

        # Declare a list to hold inferred-truth-IoU triples.
        overlaps = list()

        # For each combination of inferred and truth boxes...
        for inferred_centroid, truth_centroid in itertools.product(inferred_centroids,
                                                                   truth_centroids):
                # Compute the L2N.
                l2n = self._l2n(inferred_centroid, truth_centroid)

                # If the L2N is less than or equals the required threshold, append overlap.
                if l2n < l2n_threshold:
                    overlaps.append([inferred_centroid, truth_centroid, l2n])

        # If no confidence-filtered boxes have small enough overlap with the truth boxes.
        if not overlaps:
            # Serious mistake in original code! Reproduced here.
            # true_postives = 0
            # false_positives = 0
            # false_negatives = len(truth_boxes)

            # Alternative, correct, definition:
            true_postives = 0
            false_positives = len(inferred_centroids)
            false_negatives = len(truth_centroids)

        # Otherwise, if at least one predicted centroid meets the L2N threshold with the truth centroids.
        else:

            matched_inferred_centroids= list()
            matched_truth_centroids = list()

            overlaps = sorted(overlaps, key=lambda x: x[2], reverse=False)

            # Iterate over overlaps...
            for i, [inferred_centroid, truth_centroid, l2n] in enumerate(overlaps):
                # ...and if neither box in this overlapping pair is matched...
                if list(inferred_centroid) not in matched_inferred_centroids:

                    if list(truth_centroid) not in matched_truth_centroids:

                        # ...match both by adding them to their matched lists.
                        matched_inferred_centroids.append(list(inferred_centroid))
                        matched_truth_centroids.append(list(truth_centroid))

            # The number of true positives is the number of matched boxes.
            true_postives = len(matched_truth_centroids)

            # The number of false positives is the excess of inferred centroids.
            false_positives = len(inferred_centroids) - len(matched_inferred_centroids)

            # The number of false negatives is the excess of truth centroids.
            false_negatives = len(truth_centroids) - len(matched_truth_centroids)

        return {'true_positives': true_postives,
                'false_positives': false_positives,
                'false_negatives': false_negatives}

    @staticmethod
    def _convert_box_to_centroid(orig_box, image_size, ifov, is_prediction):
        """
        Helper function to convert the standard 4-coordinate boxes to only
        centroid coordinates. Also handles coordinate conversion from relative
        image position to arcseconds of offset.

        :param orig_box: the original box dictionary, where "box" maps to a len 4 list
            and "confidence" to the confidence scalar
        :return: the box converted to use 2-coordinates corresponding to the y and x centers
        """
        if is_prediction:
            # First figure out our centroids
            x1, y1, x2, y2 = orig_box["box"]
            y_cent = (y2 + y1) / 2.0
            x_cent = (x2 + x1) / 2.0

            # Convert coordinate frame from relative to arcseconds offset
            y_cent = y_cent * image_size[0] * ifov[0]
            x_cent = x_cent * image_size[1] * ifov[1]

            # Now build the correct dictionary structure with centroids instead
            new_box = {
                "box": [y_cent, x_cent],
                "confidence": orig_box["confidence"]
            }

            # Done, so return
            return new_box
        else:
            # First figure out our centroids
            x1, y1, x2, y2 = orig_box
            y_cent = (y2 + y1) / 2.0
            x_cent = (x2 + x1) / 2.0

            # Convert coordinate frame from relative to arcseconds offset
            y_cent = y_cent * image_size[0] * ifov[0]
            x_cent = x_cent * image_size[1] * ifov[1]

            # Done, so return
            return [y_cent, x_cent]

    def compute_statistics(self, statistics_dict=None):
        """
        Function to convert the results of analyse_detections into a meaningful
        pandas DataFrame. This includes the literal conversion, as well as the
        application of desired statistics functions to that DataFrame.

        :param statistics_dict: if the user desires non-standard statistics, they should provide them here
        :return: the resulting DataFrame, grouped by L2N and confidence thresholds.
        """
        df_header = [
            "image_name",
            "l2n_threshold",
            "confidence_threshold",
            "true_positives",
            "false_positives",
            "false_negatives"]

        # Build the confidence analysis into a dataframe.
        analysis_df = pd.DataFrame(self.confidence_analysis,
                                   columns=df_header)

        # First, if no statistic function dict is provided, use the defualt.
        if statistics_dict is None:
            statistics_dict = {"precision": ObjectDetectionAnalysisL2N._precision,
                               "recall": ObjectDetectionAnalysisL2N._recall,
                               "f1": ObjectDetectionAnalysisL2N._f1}

        data = analysis_df[["true_positives",
                            "false_positives",
                            "false_negatives",
                            "confidence_threshold",
                            "l2n_threshold"]]

        # Sum the data over images by confidence and IoU.
        grouped = data.groupby(["l2n_threshold", "confidence_threshold"]).sum()

        # Iterate over each statistic function.
        for statisitic_name, statistic_fn in statistics_dict.items():
            # Apply this statistic function across the dataframe.
            grouped[statisitic_name] = grouped.apply(statistic_fn, axis=1)

        return grouped

    def _l2n(self, pred_ctr, gt_ctr):
        """
        Calculate L2 norm (i.e. Euclidian distance) between two centroids

        :param pred_ctr: location of predicted object as [yctr, xctr] in arcseconds
        :param gt_ctr: location of ground truth object as [yctr, xctr] in arcseconds
        :return: value of the L2 Norm for the two centroids in arcseconds
        """
        pred_ctr_array = np.array(pred_ctr)
        gt_ctr_array = np.array(gt_ctr)

        # Now compute the Euclidian distance between the two centers
        l2n = la.norm(pred_ctr_array - gt_ctr_array)
        return l2n


def load_detections_from_json(json_file):
    """
    Pull a detection dictionary into memory from a JSON file.

    :param json_file: path to the JSON file
    :return: dictionary of detections
    """
    print('Loading json file...')
    with open(json_file, 'rb') as handle:
        unserialized_data = json.load(handle)
        handle.close()
        return unserialized_data


def extract_boxes_json(detections_dict, score_limit=0.0):
    """
    Extracts boxes from a given detection dict. Rewrite this for different
    detection dictionary storing architectures.

    :param detections_dict:
    :param score_limit:
    :return:
    """
    inferred_boxes = dict()

    # Zip over the inferred dectections dict values.
    for image_name, boxes, scores in zip(detections_dict['image_name'],
                                         detections_dict['predicted_boxes'],
                                         detections_dict['predicted_scores']):
        scored_boxes = list()

        # Iterate over the list of inferred boxes and scores...
        for box, score in zip(boxes, scores):
            # This works only for a 2-class (1 positive class) problem

            score = score[1]

            # ...and if the score exceeds the limit...
            if score >= score_limit:

                # ....create a mapping dict, and append it to the list.
                scored_box_dict = {"box": box,
                                   "confidence": score
                                   }

                scored_boxes.append(scored_box_dict)

        # Finally, map the scored boxes list to the image name.
        inferred_boxes[image_name] = scored_boxes

    truth_boxes = dict()

    # Iterate over the truth box dict values.
    truth_img_count = 0
    truth_box_count = 0
    for image_name, boxes in zip(detections_dict['image_name'],
                                 detections_dict['ground_truth_boxes']):
        truth_img_count += 1
        if image_name in truth_boxes.keys():
            print("Error, this name has occured before.")
            print("boxes before = " + str(truth_boxes[image_name]))
            print("boxes now = " + str(boxes))
            x = 5 / 0
        # Map each list of boxes to the cooresponding image name.
        truth_box_count += len(boxes)
        truth_boxes[image_name] = boxes

    print("Truth image count = " + str(truth_img_count))
    print("Truth box count = " + str(truth_box_count))
    return inferred_boxes, truth_boxes


def main(unused_argv):
    if FLAGS.input_type is "json":
        detections_dict = load_detections_from_json(FLAGS.input_file)
        (inferred_boxes,
         truth_boxes) = extract_boxes_json(detections_dict,
                                           score_limit=FLAGS.score_limit)
    else:
        print(FLAGS.input_type + " is not a recognized input type!")
        return 1

    # Run the analysis
    # detection_analysis = ObjectDetectionAnalysisIOU(truth_boxes,
    #                                                 inferred_boxes,
    #                                                 confidence_thresholds=None,
    #                                                 iou_thresholds=[0.85])
    detection_analysis = ObjectDetectionAnalysisL2N(truth_boxes,
                                                    inferred_boxes,
                                                    confidence_thresholds=None,
                                                    l2n_thresholds=[2, 4, 6, 8])

    # Compute the statistics
    stat_df = detection_analysis.compute_statistics()

    if FLAGS.get_recall_at_99precision:
        iou_df = stat_df.loc[0.85, :]
        print("iou_df = " + str(iou_df))
        precision = iou_df["precision"]
        recall = iou_df["recall"]

        # Restrict to unique precision values
        _, idxs = np.unique(precision, return_index=True)
        precision = precision.iloc[idxs]
        recall = recall.iloc[idxs]

        pred_recall = np.interp([0.99], precision, recall, left=-1, right=-2)
        print("Predicted Recall = " + str(pred_recall[0]))
        print("(for IOU=0.85, Precision=0.99)")

    if FLAGS.print_dataframe:

        # Display the dataframe.
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None):
            print(stat_df)

    if FLAGS.plot_pr_curve:

        # Plot the PR curve.
        detection_analysis.plot_pr_curve()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file",
                        default='.\\yolo3_eval.json',
                        help="The input file to use.")

    parser.add_argument("--input_type",
                        default="json",
                        help="One of [pickle_frcnn, pickle_yolo3, json]. Indicates pickle format.")

    parser.add_argument("--score_limit",
                        default=0.0,
                        help="All inferred boxes w/ lower scores are removed.")

    parser.add_argument("--get_recall_at_99precision", action='store_true',
                        default=False,
                        help="If True, gets the recall at IOU=0.85 and Precision=0.99")

    parser.add_argument("--print_dataframe", action='store_true',
                        default=False,
                        help="If True, prints a pandas dataframe of analysis.")

    parser.add_argument("--plot_pr_curve", action='store_true',
                        default=False,
                        help="If True, plots the PR curve.")

    FLAGS, unparsed = parser.parse_known_args()

    main(unparsed)

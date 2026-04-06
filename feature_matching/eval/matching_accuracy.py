from pathlib import Path
import numpy as np
import torch
from .retrieval_metrics import distance_from_coordinates

def calculate_coordinate_from_image_position(
    position: torch.Tensor | np.ndarray,
    camera_angle: float,
):
    # Receive a list of position -> a list of coordinates, because of projective geometry, this is much harder than I thought
    pass

def error_per_query(
    paired_coordinates: torch.Tensor | np.ndarray,
):
    #Compute estimation error for all matches
    #Given a list of coordinate, compute the
    errors = list()
    for pair in paired_coordinates:
        error = distance_from_coordinates(pair[0], pair[1])
        errors.append(error)

    return np.array(errors)


def accuracy_per_query(errors: np.ndarray, tolerance: float):
    return np.sum(errors <= tolerance) / len(errors)

def accuracy_across_queries(errors_list: list[np.ndarray], tolerance: float):
    errors = np.concatenate(errors_list)
    return np.sum(errors <= tolerance) / len(errors)


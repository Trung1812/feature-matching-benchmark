import numpy as np
import pytest
from geopy.distance import geodesic
from pyproj import Transformer

from feature_matching.eval.utils import (
	distance_from_coordinates,
	transform_image_to_utm,
	transform_wgs84_to_utm,
)


def test_transform_image_to_utm_requires_at_least_4_pairs():
	correspondences = [
		((0.0, 0.0), (1000.0, 2000.0)),
		((1.0, 0.0), (1010.0, 2000.0)),
		((1.0, 1.0), (1010.0, 2010.0)),
	]

	with pytest.raises(AssertionError, match="requires at least 4 pairs"):
		transform_image_to_utm(correspondences, np.array([[0.5, 0.5]], dtype=np.float32))


def test_transform_image_to_utm_maps_points_with_homography():
	correspondences = [
		((0.0, 0.0), (1000.0, 2000.0)),
		((1.0, 0.0), (1010.0, 2000.0)),
		((1.0, 1.0), (1010.0, 2010.0)),
		((0.0, 1.0), (1000.0, 2010.0)),
	]
	query_coords = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float32)

	transformed = transform_image_to_utm(correspondences, query_coords)
	expected = np.array([[1005.0, 2005.0], [1002.0, 2008.0]], dtype=np.float64)

	assert transformed.shape == (2, 2)
	np.testing.assert_allclose(transformed, expected, rtol=1e-5, atol=1e-4)


def test_transform_wgs84_to_utm_matches_pyproj_reference():
	coords = np.array(
		[
			[105.83416, 21.027764],
			[105.804817, 21.028511],
		],
		dtype=np.float64,
	)
	transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
	expected = np.array([transformer.transform(c[0], c[1]) for c in coords], dtype=np.float64)

	transformed = transform_wgs84_to_utm(coords)

	assert transformed.shape == (2, 2)
	np.testing.assert_allclose(transformed, expected, rtol=1e-7, atol=1e-6)


def test_distance_from_coordinates_utm_uses_euclidean_distance():
	coord1 = (0.0, 0.0)
	coord2 = (3.0, 4.0)

	distance = distance_from_coordinates(coord1, coord2, crs="UTM")

	assert distance == pytest.approx(5.0)


def test_distance_from_coordinates_wgs84_uses_geodesic_distance():
	coord1 = (21.027764, 105.83416)
	coord2 = (21.028511, 105.804817)
	expected = geodesic(coord1, coord2).meters

	distance = distance_from_coordinates(coord1, coord2, crs="WGS84")

	assert distance == pytest.approx(expected, rel=1e-9)


def test_distance_from_coordinates_rejects_unsupported_crs():
	with pytest.raises(AssertionError, match="Does not support distance computation"):
		distance_from_coordinates((0.0, 0.0), (1.0, 1.0), crs="EPSG:3857")

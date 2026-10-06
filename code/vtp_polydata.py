from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass(frozen=True)
class SurfaceField:
    face_centers: np.ndarray
    face_areas: np.ndarray
    cell_data: dict[str, np.ndarray]
    point_count: int
    face_count: int


def read_ascii_vtp_surface(path: Path) -> SurfaceField:
    """Read an ASCII VTK PolyData surface and compute polygon geometry."""
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    if piece is None:
        raise ValueError(f"missing PolyData Piece in {path}")

    points_parent = piece.find("Points")
    polys_parent = piece.find("Polys")
    cell_parent = piece.find("CellData")
    if points_parent is None or polys_parent is None or cell_parent is None:
        raise ValueError(f"incomplete PolyData structure in {path}")

    points_node = points_parent.find("DataArray")
    if points_node is None:
        raise ValueError(f"missing point coordinates in {path}")
    points = _numbers(points_node, np.float64).reshape(-1, 3)
    connectivity = _named_numbers(polys_parent, "connectivity", np.int64)
    offsets = _named_numbers(polys_parent, "offsets", np.int64)
    expected_faces = int(piece.attrib["NumberOfPolys"])
    if len(offsets) != expected_faces:
        raise ValueError(f"polygon count mismatch in {path}")

    starts = np.concatenate([np.zeros(1, dtype=np.int64), offsets[:-1]])
    counts = offsets - starts
    centers = np.empty((expected_faces, 3), dtype=np.float64)
    areas = np.empty(expected_faces, dtype=np.float64)
    for vertex_count in np.unique(counts):
        if vertex_count < 3:
            raise ValueError(f"polygon with fewer than three vertices in {path}")
        rows = np.flatnonzero(counts == vertex_count)
        for first in range(0, len(rows), 100_000):
            current = rows[first : first + 100_000]
            positions = starts[current, None] + np.arange(vertex_count)[None, :]
            polygon = points[connectivity[positions]]
            centers[current] = polygon.mean(axis=1)
            anchor = polygon[:, 0]
            area = np.zeros(len(current), dtype=np.float64)
            for corner in range(1, int(vertex_count) - 1):
                cross = np.cross(polygon[:, corner] - anchor, polygon[:, corner + 1] - anchor)
                area += 0.5 * np.linalg.norm(cross, axis=1)
            areas[current] = area

    data: dict[str, np.ndarray] = {}
    for node in cell_parent.findall("DataArray"):
        name = node.attrib.get("Name")
        if name:
            values = _numbers(node, np.float64)
            components = int(node.attrib.get("NumberOfComponents", "1"))
            if components > 1:
                values = values.reshape(-1, components)
            if values.shape[0] != expected_faces:
                raise ValueError(f"cell field {name} has an invalid length in {path}")
            data[name] = values

    return SurfaceField(
        face_centers=centers,
        face_areas=areas,
        cell_data=data,
        point_count=len(points),
        face_count=expected_faces,
    )


def _named_numbers(parent: ET.Element, name: str, dtype: np.dtype) -> np.ndarray:
    for node in parent.findall("DataArray"):
        if node.attrib.get("Name") == name:
            return _numbers(node, dtype)
    raise ValueError(f"missing DataArray {name}")


def _numbers(node: ET.Element, dtype: np.dtype) -> np.ndarray:
    if node.attrib.get("format", "ascii") != "ascii":
        raise ValueError("only ASCII VTP arrays are supported")
    return np.fromstring(node.text or "", sep=" ", dtype=dtype)

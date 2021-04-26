from typing import Tuple, List, Set, Dict, Optional, Union
import numpy as np  # no opencv used

Rectangle = Tuple[int, int, int, int]
Contour = Set[Tuple[int, int]]
BGR = np.ndarray  # shape=(h, w, 3), dtype=np.uint8
Grayscale = np.ndarray  # shape=(h, w), dtype=np.uint8
Binary = np.ndarray  # shape=(h, w), dtype=bool


def circle_mask(shape: Tuple[int, int], center: Tuple[int, int], radius: int) -> Binary:
    """
    Generate a filled circle mask.
    """
    I, J = np.indices(shape)
    return (I - np.full(shape, center[0])) ** 2 + (J - np.full(shape, center[1])) ** 2 <= radius ** 2


def sequential_label(binary_img: Binary) -> Grayscale:
    """
    Label connected objects in a binary image sequentially.
    """

    def find_root(v: int) -> int:
        while True:
            parent = temp_map[v]
            if parent == v:
                return v
            v = parent

    h, w = binary_img.shape
    temp_img = np.zeros((h + 1, w + 2), dtype=np.uint16)  # enlarge width and height; assume no more than 65535 hits
    temp_map = {}  # map a label to the known minimal label in its group
    current = 0
    # first round: label and group
    for y in range(h):
        for x in range(w):
            if binary_img[y][x] == 0:
                continue
            left, up_left, up, up_right = temp_img[y + 1][x], temp_img[y][x], temp_img[y][x + 1], temp_img[y][x + 2]
            nonzero = set(label for label in (left, up_left, up, up_right) if label > 0)
            if not nonzero:  # all checked neighbors are empty
                current += 1
                temp_map[current] = current
                temp_img[y + 1][x + 1] = current
            else:  # unify labels
                roots = set(find_root(label) for label in nonzero)
                m = min(roots)
                for root in roots:
                    temp_map[root] = m
                temp_img[y + 1][x + 1] = m
    # second round: relabel as all labels and groups are known,
    for label in temp_map:  # map a label to the representative in its group
        temp_map[label] = find_root(label)
    label_map: Dict[int, int] = {v: i + 1 for i, v in enumerate(set(temp_map.values()))}  # relabel from 1 to n
    label_img = np.zeros_like(binary_img, dtype=np.uint8)  # assume no more than 255 labels
    for y in range(h):
        for x in range(w):
            label = temp_img[y + 1][x + 1]
            if label > 0:
                label_img[y][x] = label_map[temp_map[label]]
    return label_img


def extract_objs(label_img: Grayscale, minimal_area: int) -> List[Binary]:
    """
    Extract objects from a label image and filter out small ones.
    """
    objs = []
    for label in np.unique(label_img)[1:]:
        obj: Binary = label_img == label
        if obj.sum() >= minimal_area:
            objs.append(obj)
    return objs


def bounding_box(obj: Binary) -> Rectangle:
    """
    Get bounding box of an object.
    """
    nonzero_row, nonzero_col = np.nonzero(obj)
    y = nonzero_row.min()
    x = nonzero_col.min()
    return x, y, nonzero_col.max() + 1 - x, nonzero_row.max() + 1 - y


def find_contour(obj: Binary) -> Contour:
    """
    Moore-Neighbor tracing algorithm.
    """
    neighbors = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    contour: Contour = set()
    h, w = obj.shape
    y, x = start = np.unravel_index(np.argmax(obj), obj.shape)
    visited_neighbors: Set[Tuple[int, int]] = set()
    direction = 0
    while True:
        dy, dx = neighbors[direction]
        ny, nx = y + dy, x + dx
        if not (0 <= ny < h and 0 <= nx < w) or obj[ny][nx] == 0:
            direction = (direction + 1) % 8
            if (y, x) == start:
                if (ny, nx) in visited_neighbors:  # stopping criterion: from start point visit a neighbor of it twice
                    break
                visited_neighbors.add((ny, nx))
        else:
            direction = (direction // 2 + 3) % 4 * 2  # reverse the direction
            y = ny
            x = nx
            contour.add((y, x))
    return contour


def fill_contour(obj: Binary) -> Binary:
    """
    Fill the holes inside a contour by flood-fill the region outside.
    """
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    x0, y0, w, h = bounding_box(obj)
    x0 -= 1
    y0 -= 1
    w += 2
    h += 2
    subimg = np.full((h, w), 1, dtype=bool)
    stack: List[List[int]] = [[0, 0, 0]]
    while stack:
        y, x, d = stack[-1]
        if d == 4:
            stack.pop()
            continue
        subimg[y][x] = 0
        stack[-1][2] += 1
        y += directions[d][0]
        x += directions[d][1]
        if 0 <= y < h and 0 <= x < w and not obj[y + y0][x + x0] and subimg[y][x]:
            stack.append([y, x, 0])
    ret = np.zeros_like(obj)
    ret[y0:(y0 + h), x0:(x0 + w)] = subimg
    return ret


def L1_contour_distance(contour0: Contour, contour1: Contour) -> int:
    """
    L1 distance between 2 contours.
    """
    return min(abs(y0 - y1) + abs(x0 - x1) for (y0, x0) in contour0 for (y1, x1) in contour1)


def box_contain(box0: Rectangle, box1: Rectangle) -> int:
    """
    Whether a bounding box contains another.
    """
    x0, y0, w0, h0 = box0
    x1, y1, w1, h1 = box1
    if x0 <= x1 and x0 + w0 >= x1 + w1 and y0 <= y1 and y0 + h0 >= y1 + h1:
        return 1
    if x1 <= x0 and x1 + w1 >= x0 + w0 and y1 <= y0 and y1 + h1 >= y0 + h0:
        return -1
    return 0


def L1_box_distance(box0: Rectangle, box1: Rectangle) -> int:
    """
    L1 distance between 2 bounding boxes.
    """
    x0, y0, w0, h0 = box0
    x1, y1, w1, h1 = box1
    return max(x1 - (x0 + w0), x0 - (x1 + w1), 0) + max(y1 - (y0 + h0), y0 - (y1 + h1), 0)


def average_color(img: BGR, obj: Binary) -> np.ndarray:
    """
    Average color of an object.
    """
    return np.average(img[obj], axis=0)


def color_distance(img0: BGR, obj0, img1: BGR, obj1) -> float:
    """
    Color distance (the maximum difference of 3 color channels) between 2 objects.
    """
    return np.abs(average_color(img0, obj0) - average_color(img1, obj1)).max()


class MergeCandidate:
    """
    Helper class of merge_objs.
    """
    ID = 0

    def __init__(self, obj: Binary, contour: Optional[Contour] = None):
        self.id = MergeCandidate.ID
        MergeCandidate.ID += 1
        self.obj = obj
        self.contour = contour if contour else find_contour(obj)
        self.box = bounding_box(obj)


def merge_objs(img: BGR, objs: List[Binary], maximal_distance: int, maximal_color_distance: int):
    """
    Merge objects that are too close and has similar colors.
    """
    candidates = [MergeCandidate(obj) for obj in objs]
    unmergable_pairs = set()  # cache
    merged_candidate: Optional[MergeCandidate] = candidates[0]  # dummy initial value
    while merged_candidate:
        merged_candidate = None
        n = len(candidates)
        for i in range(n - 1):
            candidate_i = candidates[i]
            id_i = candidate_i.id
            for j in range(i + 1, n):
                candidate_j = candidates[j]
                id_j = candidate_j.id
                # if checked or bounding boxes are far, no need to proceed
                if (id_i, id_j) in unmergable_pairs or L1_box_distance(candidate_i.box,
                                                                       candidate_j.box) > maximal_distance:
                    continue
                contain = box_contain(candidate_i.box, candidate_j.box)
                if contain:
                    # merge the contained object to the containing object
                    merged_candidate = candidate_i if contain > 0 else candidate_j
                    merged_candidate.obj = np.logical_or(candidate_i.obj, candidate_j.obj)
                elif L1_contour_distance(candidate_i.contour, candidate_j.contour) <= maximal_distance:
                    color_dist = color_distance(img, candidate_i.obj, img, candidate_j.obj)
                    if color_dist <= maximal_color_distance:
                        merged_contour = candidate_i.contour.union(candidate_j.contour)
                        merged_obj = np.logical_or(candidate_i.obj, candidate_j.obj)
                        merged_candidate = MergeCandidate(merged_obj, merged_contour)
                if merged_candidate:
                    # remove the merged objects and add the new object
                    candidates = [candidate for k, candidate in enumerate(candidates) if k not in (i, j)]
                    candidates.append(merged_candidate)
                    break
                unmergable_pairs.add((id_i, id_j))
                unmergable_pairs.add((id_j, id_i))
            if merged_candidate:
                break
    return [candidate.obj for candidate in candidates]


def moments(obj: Binary) -> Dict[str, Union[int, float]]:
    """
    Raw moments, central moments and scale invariants.
    """
    h, w = obj.shape
    moments_map = {}
    I, J = np.indices((h, w), sparse=True)
    for i in range(4):
        for j in range(4 - i):
            moments_map[f'm{i}{j}'] = (I ** i * J ** j * obj).sum()
    x_bar, y_bar = moments_map['m10'] / moments_map['m00'], moments_map['m01'] / moments_map['m00']
    moments_map['mu20'] = moments_map['m20'] - x_bar * moments_map['m10']
    moments_map['mu02'] = moments_map['m02'] - y_bar * moments_map['m01']
    moments_map['mu11'] = moments_map['m11'] - x_bar * moments_map['m01']
    moments_map['mu21'] = moments_map['m21'] - 2 * x_bar * moments_map['m11'] \
                          - y_bar * moments_map['m20'] + 2 * x_bar ** 2 * moments_map['m01']
    moments_map['mu12'] = moments_map['m12'] - 2 * y_bar * moments_map['m11'] \
                          - x_bar * moments_map['m02'] + 2 * y_bar ** 2 * moments_map['m10']
    moments_map['mu30'] = moments_map['m30'] - 3 * x_bar * moments_map['m20'] + 2 * x_bar ** 2 * moments_map['m10']
    moments_map['mu03'] = moments_map['m03'] - 3 * y_bar * moments_map['m02'] + 2 * y_bar ** 2 * moments_map['m01']
    for i in range(4):
        for j in range(max(2 - i, 0), 4 - i):
            moments_map[f'nu{i}{j}'] = moments_map[f'mu{i}{j}'] / moments_map['m00'] ** (1 + (i + j) / 2)
    return moments_map


def HuMoments(obj: Binary) -> np.ndarray:
    """
    Normalized first 6 Hu Moments.
    """
    moments_map = moments(np.uint8(obj))
    huMoments = [moments_map['nu20'] + moments_map['nu02'],
                 (moments_map['nu20'] - moments_map['nu02']) ** 2 + 4 * moments_map['nu11'] ** 2,
                 (moments_map['nu30'] - 3 * moments_map['nu12']) ** 2
                 + (3 * moments_map['nu21'] - moments_map['nu03']) ** 2,
                 (moments_map['nu30'] + moments_map['nu12']) ** 2 + (moments_map['nu21'] + moments_map['nu03']) ** 2,
                 (moments_map['nu30'] - 3 * moments_map['nu12']) * (moments_map['nu30'] + moments_map['nu12'])
                 * ((moments_map['nu30'] + moments_map['nu12']) ** 2
                    - 3 * (moments_map['nu21'] + moments_map['nu03']) ** 2)
                 + (3 * moments_map['nu21'] - moments_map['nu03'])
                 * (moments_map['nu21'] + moments_map['nu03'])
                 * (3 * (moments_map['nu30'] + moments_map['nu12']) ** 2
                    - (moments_map['nu21'] + moments_map['nu03']) ** 2),
                 (moments_map['nu20'] - moments_map['nu02'])
                 * ((moments_map['nu30'] + moments_map['nu12']) ** 2 - (moments_map['nu21'] + moments_map['nu03']) ** 2)
                 + 4 * moments_map['nu11'] * (moments_map['nu30'] + moments_map['nu12'])
                 * (moments_map['nu21'] + moments_map['nu03'])]
    return np.log10(np.abs(huMoments))


def match(img0: BGR, objs0: List[Binary], img1: BGR, objs1: List[Binary], maximal_color_distance=255) \
        -> Tuple[int, int]:
    """
    Find matched pair from 2 images and their extracted objects.
    """
    huMoments0 = [HuMoments(obj) for obj in objs0]
    huMoments1 = [HuMoments(obj) for obj in objs1]
    distance_matrix = np.array([[np.linalg.norm(x - y) for x in huMoments1] for y in huMoments0])
    sorted_indices = np.argsort(distance_matrix, axis=None)
    pair = (0, 0)
    for index in sorted_indices:
        pair = np.unravel_index(index, distance_matrix.shape)
        color_dist = color_distance(img0, objs0[pair[0]], img1, objs1[pair[1]])
        # after sorting by the difference of normalized Hu Moments, the first pair that has similar color is the target
        if color_dist <= maximal_color_distance:
            break
    return pair

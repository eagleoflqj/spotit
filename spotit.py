import sys
from cv2 import imread, imwrite  # only I/O functions used
from util import *


class MagicNumbers:
    CIRCLE_CENTER_X = 200
    CIRCLE_CENTER_Y = 200
    CIRCLE_RADIUS = 178
    CARDS_THRESHOLD = 200
    IMAGES_THRESHOLD = 220
    MINIMAL_AREA = 20
    MAXIMAL_DISTANCE = 10
    MAXIMAL_MERGE_COLOR_DISTANCE = 70
    MAXIMAL_MATCH_COLOR_DISTANCE = 50


CARDS = "cards"
IMAGES = "images"


def check_args():
    try:
        assert len(sys.argv) == 5
        category = sys.argv[1]
        assert category in (CARDS, IMAGES)
        file0, file1, output = sys.argv[2:5]
        return category, file0, file1, output
    except AssertionError:
        print(f"Usage: python3 {sys.argv[0]} {CARDS}|{IMAGES} file0 file1 output")
        exit(1)


def main():
    category, file0, file1, output = check_args()

    imgs = [imread(file) for file in (file0, file1)]
    objs_list = []
    for img in imgs:
        binary = np.full(img.shape[:2], 1, dtype=bool)
        # apply different mask logic
        if category == CARDS:
            base_mask = ~circle_mask(img.shape[:2], (MagicNumbers.CIRCLE_CENTER_Y, MagicNumbers.CIRCLE_CENTER_X),
                                     MagicNumbers.CIRCLE_RADIUS)
            mask = np.logical_or(base_mask, np.logical_and.reduce(
                list(img[:, :, i] > MagicNumbers.CARDS_THRESHOLD for i in range(3))))
        else:
            mask = np.logical_and.reduce(list(img[:, :, i] > MagicNumbers.IMAGES_THRESHOLD for i in range(3)))
        binary[mask] = 0
        # process binary image
        label_img = sequential_label(binary)
        objs = extract_objs(label_img, MagicNumbers.MINIMAL_AREA)
        objs = merge_objs(img, objs, MagicNumbers.MAXIMAL_DISTANCE, MagicNumbers.MAXIMAL_MERGE_COLOR_DISTANCE)
        objs_list.append(objs)
    # perform match
    pair = match(imgs[0], objs_list[0], imgs[1], objs_list[1], MagicNumbers.MAXIMAL_MATCH_COLOR_DISTANCE)
    generated_mask = np.hstack(list(fill_contour(objs[index]) for objs, index in zip(objs_list, pair)))
    imwrite(output, np.uint8(generated_mask) * 255)


if __name__ == "__main__":
    main()

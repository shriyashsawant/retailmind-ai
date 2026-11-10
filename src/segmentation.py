"""Day 8: cut the product out of its picture (background removal) with OpenCV.

Two simple methods, both classical (no model to download):

  - segment_plain_background(): for pictures where the product sits on a plain background
    (like Fashion-MNIST, or a studio photo). It finds which pixels differ from the background
    colour (read from the picture's edge), then tidies the result.
  - segment_with_box(): for ordinary colour photos. You give it a rough box around the product
    (for example from the Day 7 detector) and GrabCut works out which pixels inside the box
    belong to the product.

A "mask" is a picture of the same size where 255 means "product" and 0 means "background".
put_on_background() uses a mask to paste the product on a clean uniform background.

Limits, stated honestly: these methods do not understand what a product is. A product with a
hole or a colour close to the background can be cut badly. A learned segmenter (U-Net, Mask R-CNN)
would do better on messy real photos; it needs labelled masks we do not have.
"""
import cv2
import numpy as np

DIFFERENCE_THRESHOLD = 8      # how different from the background a pixel must be to count as product (0-255)


def segment_plain_background(picture):
    """picture: grey (height, width) uint8 array. Returns a mask (255 = product)."""
    border = np.concatenate([picture[0], picture[-1], picture[:, 0], picture[:, -1]])
    background_level = np.median(border)                          # the edge of the picture shows the background colour

    # product = any pixel that differs from the background by more than a small amount.
    # (Otsu's automatic threshold was tried first, but it threw away dark products such as black coats.
    #  The value 8 was picked on training pictures: anything from 3 to 15 kept the classifier's accuracy.)
    # (No blurring or "closing" here: on a 28x28 picture they merged the narrow gap between trouser legs.)
    difference = np.abs(picture.astype(int) - background_level)
    mask = (difference > DIFFERENCE_THRESHOLD).astype("uint8") * 255
    return fill_enclosed_holes(keep_biggest_piece(mask))


def fill_enclosed_holes(mask):
    """Dark patches completely surrounded by the product (a black logo, a button) are part of it.
    Background that touches the picture's edge (the gap between trouser legs) is left alone."""
    outside = mask.copy()
    flood_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)    # OpenCV wants a border of 1 pixel
    height, width = mask.shape
    edge_points = [(x, 0) for x in range(width)] + [(x, height - 1) for x in range(width)] \
        + [(0, y) for y in range(height)] + [(width - 1, y) for y in range(height)]
    for x, y in edge_points:                                          # start from EVERY background pixel on the border,
        if outside[y, x] == 0:                                        # not just the corners (a gap can open onto the edge)
            cv2.floodFill(outside, flood_mask, (x, y), 255)           # paint everything reachable from the edge
    enclosed = cv2.bitwise_not(outside)                               # whatever was NOT reached is enclosed
    return cv2.bitwise_or(mask, enclosed)


def segment_with_box(photo, box):
    """photo: colour (height, width, 3) uint8 array (BGR, as OpenCV uses). box: (x1, y1, x2, y2) in pixels.

    Returns a mask (255 = product)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    rectangle = (x1, y1, x2 - x1, y2 - y1)                       # OpenCV wants (x, y, width, height)
    mask = np.zeros(photo.shape[:2], np.uint8)
    background_model = np.zeros((1, 65), np.float64)             # scratch space GrabCut needs
    product_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(photo, mask, rectangle, background_model, product_model, 5, cv2.GC_INIT_WITH_RECT)

    product = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    return keep_biggest_piece(product)


def keep_biggest_piece(mask):
    """A mask can have several separate blobs. Keep only the largest one (the product itself)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask                                              # nothing found, return it unchanged
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))    # label 0 is the background
    return np.where(labels == biggest, 255, 0).astype("uint8")


def put_on_background(picture, mask, background=255):
    """Paste the product on a uniform background. background: 255 = white, 0 = black (or a BGR tuple for colour)."""
    clean = np.full_like(picture, background)
    clean[mask > 0] = picture[mask > 0]
    return clean


def mask_area_share(mask):
    """How much of the picture the product covers (0 to 1). Handy sanity check."""
    return float((mask > 0).mean())

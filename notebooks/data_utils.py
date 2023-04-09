import torch
import numpy as np

from enum import IntEnum
from typing import List


__all__ = ["BirdViewProducer", "DEFAULT_HEIGHT", "DEFAULT_WIDTH"]


DEFAULT_HEIGHT = 336  # its 84m when density is 4px/m
DEFAULT_WIDTH = 150  # its 37.5m when density is 4px/m

BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)
RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)

COLOR_ON = 1


class RGB:
    VIOLET = (173, 127, 168)
    ORANGE = (252, 175, 62)
    CHOCOLATE = (233, 185, 110)
    CHAMELEON = (138, 226, 52)
    SKY_BLUE = (114, 159, 207)
    DIM_GRAY = (105, 105, 105)
    DARK_GRAY = (50, 50, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)



class BirdViewMasks(IntEnum):
    PEDESTRIANS = 7
    RED_LIGHTS = 6
    YELLOW_LIGHTS = 5
    GREEN_LIGHTS = 4
    AGENT = 3
    VEHICLES = 2
    #    CENTERLINES = 2
    LANES = 1
    ROAD = 0

    @staticmethod
    def top_to_bottom() -> List[int]:
        return list(BirdViewMasks)

    @staticmethod
    def bottom_to_top() -> List[int]:
        return list(reversed(BirdViewMasks.top_to_bottom()))


RGB_BY_MASK = {
    BirdViewMasks.PEDESTRIANS: RGB.VIOLET,
    BirdViewMasks.RED_LIGHTS: RGB.RED,
    BirdViewMasks.YELLOW_LIGHTS: RGB.YELLOW,
    BirdViewMasks.GREEN_LIGHTS: RGB.GREEN,
    BirdViewMasks.AGENT: RGB.CHAMELEON,
    BirdViewMasks.VEHICLES: RGB.ORANGE,
    # BirdViewMasks.CENTERLINES: RGB.CHOCOLATE,
    BirdViewMasks.LANES: RGB.WHITE,
    BirdViewMasks.ROAD: RGB.DIM_GRAY,
}

BIRDVIEW_SHAPE_CHW = (len(RGB_BY_MASK), DEFAULT_HEIGHT, DEFAULT_WIDTH)
BIRDVIEW_SHAPE_HWC = (DEFAULT_HEIGHT, DEFAULT_WIDTH, len(RGB_BY_MASK))



class BirdViewProducer:
    """Responsible for producing top-down view on the map, following agent's vehicle.

    About BirdView:
    - top-down view, fixed directly above the agent (including vehicle rotation), cropped to desired size
    - consists of stacked layers (masks), each filled with ones and zeros (depends on MaskMaskGenerator implementation).
        Example layers: road, vehicles, pedestrians. 0 indicates -> no presence in that pixel, 1 -> presence
    - convertible to RGB image
    - Rendering full road and lanes masks is computationally expensive, hence caching mechanism is used
    """

    def __init__(
        self
    ) -> None:
        pass
 
    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        # TODO: fix it for batch 
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            rgb_color = RGB_BY_MASK[mask_type]
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas


def to_rgb(binary_img):
    result = torch.tensor(np.array(np.transpose(BirdViewProducer.as_rgb(binary_img.sigmoid().detach().cpu().numpy() > 0.5), (2, 0, 1)))).float()
    return result


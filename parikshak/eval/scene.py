"""A synthetic camera looking at tags on the rack: images with known ground truth.

Fixtures, not runtime. The marker detector, the rack lock and the whole live
path can then be exercised on pixels whose true rack-frame answer is known to
the millimetre, instead of on a webcam pointed at whatever is on the desk. It
also backs `parikshak run --camera synthetic`, so the live loop can be shown and
smoke-tested with no camera and no printed tags.

Tags are pasted with a white quiet zone, as a printed tag has one - ArUco finds
nothing without it - and warped with INTER_AREA, which keeps the edge a detector
refines onto instead of aliasing it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from parikshak.perception.rackframe import CameraIntrinsics

#: Source resolution of one tag before it is warped into the scene.
_TAG_PX = 400
_QUIET_PX = 80


@dataclass(frozen=True)
class TagScene:
    """A pinhole camera at a known pose relative to the rack."""

    intrinsics: CameraIntrinsics
    width: int
    height: int
    #: rack -> camera, p_cam = R @ p_rack + t
    R: np.ndarray
    t: np.ndarray

    @classmethod
    def facing_rack(cls, *, width: int = 1280, height: int = 720, hfov_deg: float = 70.0,
                    distance_m: float = 1.5,
                    aim_rack: tuple[float, float] = (0.0, -0.075)) -> TagScene:
        """A camera square-on to the rack face, `distance_m` out along +Z.

        Camera +X is rack +X, camera +Y (image down) is rack -Y, and the camera
        looks down rack -Z - a crew-eye view of the face.
        """
        r = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        centre = np.array([aim_rack[0], aim_rack[1], distance_m])
        return cls(CameraIntrinsics.from_fov(width, height, hfov_deg), width, height,
                   r, -r @ centre)

    @property
    def camera_position_rack(self) -> np.ndarray:
        return -self.R.T @ self.t

    def project(self, points_rack) -> np.ndarray:
        pts = np.asarray(points_rack, dtype=float).reshape(-1, 3)
        return self.intrinsics.project((self.R @ pts.T).T + self.t)

    @staticmethod
    def square(centre, size_m: float) -> np.ndarray:
        """Corners of a tag facing +Z (toward the crew), in the layout's order:
        counter-clockwise from the bottom-left."""
        c = np.asarray(centre, dtype=float)
        h = size_m / 2.0
        return np.array([c + [-h, -h, 0.0], c + [h, -h, 0.0], c + [h, h, 0.0], c + [-h, h, 0.0]])

    def render(self, tags: Iterable[tuple[int, np.ndarray]], *,
               background: int = 255) -> np.ndarray:
        """A BGR image of the given tags. `tags` is (tag_id, corners (4, 3))."""
        import cv2

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        canvas = np.full((self.height, self.width, 3), background, dtype=np.uint8)
        for tag_id, corners in tags:
            marker = cv2.aruco.generateImageMarker(dictionary, int(tag_id), _TAG_PX)
            img = cv2.copyMakeBorder(marker, _QUIET_PX, _QUIET_PX, _QUIET_PX, _QUIET_PX,
                                     cv2.BORDER_CONSTANT, value=255)
            side = img.shape[0]
            bl, br, tr, tl = (np.asarray(c, dtype=float) for c in corners)
            centre = (bl + br + tr + tl) / 4.0
            grow = 1.0 + 2.0 * _QUIET_PX / _TAG_PX
            quiet = [centre + (c - centre) * grow for c in (tl, tr, br, bl)]
            dst = self.project(np.array(quiet))
            if np.any(np.isnan(dst)):
                continue
            src = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
            h = cv2.getPerspectiveTransform(src, np.float32(dst))
            size = (self.width, self.height)
            warped = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), h, size,
                                         flags=cv2.INTER_AREA)
            mask = cv2.warpPerspective(np.full((side, side), 255, np.uint8), h, size)
            canvas[mask > 0] = warped[mask > 0]
        return canvas

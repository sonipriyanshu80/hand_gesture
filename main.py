import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


def make_skin_mask(frame: np.ndarray) -> np.ndarray:
    """Return a smooth skin mask for the current frame."""
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)

    low_hsv = np.array([0, 25, 90], dtype=np.uint8)
    high_hsv = np.array([25, 255, 255], dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, low_hsv, high_hsv)

    low_ycrcb = np.array([0, 130, 90], dtype=np.uint8)
    high_ycrcb = np.array([255, 180, 140], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, low_ycrcb, high_ycrcb)

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return mask


def pick_hand_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Pick the biggest contour in the mask and treat it as the hand."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < 9000:
        return None
    return hand


def keep_spread_out_points(points: List[Tuple[int, int]], min_gap: float = 25.0) -> List[Tuple[int, int]]:
    """Drop points that sit too close to each other."""
    filtered: List[Tuple[int, int]] = []
    for pt in points:
        if all(math.hypot(pt[0] - other[0], pt[1] - other[1]) > min_gap for other in filtered):
            filtered.append(pt)
    return filtered


def count_fingers(
    contour: np.ndarray,
    frame: np.ndarray
) -> Tuple[int, List[Tuple[int, int]], List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]:
    """Measure raised fingers from convexity defects."""
    hull_ids = cv2.convexHull(contour, returnPoints=False)
    if hull_ids is None or len(hull_ids) < 3:
        return 0, [], []

    defects = cv2.convexityDefects(contour, hull_ids)
    if defects is None:
        return 0, [], []

    finger_spots: List[Tuple[int, int]] = []
    finger_links: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
    count = 0

    for s, e, f, depth in defects[:, 0]:
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        dip = tuple(contour[f][0])

        a = math.dist(contour[e][0], contour[f][0])
        b = math.dist(contour[s][0], contour[f][0])
        c = math.dist(contour[s][0], contour[e][0])
        if a == 0 or b == 0:
            continue

        angle = math.degrees(math.acos(min(1, max(-1, (a * a + b * b - c * c) / (2 * a * b)))))

        if angle <= 95 and depth > 15000:
            finger_spots.extend([start, end])
            finger_links.append((start, dip, end))
            count += 1
            cv2.circle(frame, dip, 5, (0, 0, 255), -1)

    finger_tips = keep_spread_out_points(finger_spots)
    finger_tips.sort(key=lambda p: p[1])
    finger_tips = finger_tips[:5]
    for tip in finger_tips:
        cv2.circle(frame, tip, 8, (0, 255, 0), -1)

    if count == 0 and finger_tips:
        return len(finger_tips), finger_tips, finger_links

    return min(max(len(finger_tips), count + 1), 5), finger_tips, finger_links


def label_gesture(
    finger_count: int,
    contour: np.ndarray,
    finger_tips: List[Tuple[int, int]]
) -> str:
    """Give a friendly name to the detected hand pose."""
    if contour is None:
        return "No Hand Detected"

    if finger_count == 0:
        return "Fist"
    if finger_count == 1:
        if thumb_is_up(contour, finger_tips):
            return "Thumbs Up"
        return "One Finger"
    if finger_count == 2:
        return "Victory"
    if finger_count == 5:
        return "Open Palm"
    return f"{finger_count} Fingers"


def thumb_is_up(contour: np.ndarray, finger_tips: List[Tuple[int, int]]) -> bool:
    """Check if the single raised finger looks like a thumb."""
    if not finger_tips:
        return False

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return False
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    tip = max(finger_tips, key=lambda p: cy - p[1])
    dx = tip[0] - cx
    dy = cy - tip[1]

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    if width == 0 or height == 0:
        return False

    ratio = max(width, height) / (min(width, height) + 1e-5)

    pointing_up = dy > abs(dx) * 0.5 and dy > 40
    sideways = ratio > 1.35

    return pointing_up and sideways


def draw_hand_edges(frame: np.ndarray, contour: np.ndarray) -> None:
    hull = cv2.convexHull(contour)
    cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
    cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)





def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Hand Gesture Recognition Started!")
    print("Press 'q' to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        mask = make_skin_mask(frame)
        hand = pick_hand_contour(mask)

        fingers = 0
        finger_tips: List[Tuple[int, int]] = []
        finger_links: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = []
        pose = "No Hand Detected"

        if hand is not None:
            draw_hand_edges(frame, hand)
            fingers, finger_tips, finger_links = count_fingers(hand, frame)
            pose = label_gesture(fingers, hand, finger_tips)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 90), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

        cv2.putText(
            frame,
            f"Gesture: {pose}",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Fingers: {fingers}",
            (18, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended")


if __name__ == "__main__":
    main()




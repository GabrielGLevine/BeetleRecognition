import csv
from typing import Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
from tqdm import tqdm


def find_circles(ref_frame: np.ndarray) -> np.ndarray:
    """
    Find circles in the reference frame using Hough Circle Transform.

    Args:
        ref_frame (np.ndarray): The reference frame image.

    Returns:
        np.ndarray: Sorted circles found in the reference frame.
    """
    gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 1)

    # Use the Hough Circle Transform to detect circles
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=400,
        param1=50,
        param2=50,
        minRadius=245,
        maxRadius=270,
    )

    if circles is None:
        print("No circles found")
        exit(1)

    circles = np.round(circles[0, :]).astype("int")
    # Define a tolerance for y-coordinates to group rows
    tolerance = 150
    # Round y values to the nearest integer based on tolerance
    rounded_y = np.round(circles[:, 1] / tolerance) * tolerance

    # Sort by adjusted y and then by x
    sorted_indices = np.lexsort((circles[:, 0], rounded_y))
    sorted_circles = circles[sorted_indices]

    return sorted_circles


def generate_mask(frame: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    """
    Generate a mask for a circular region in the frame.

    Args:
        frame (np.ndarray): The frame image.
        x (int): The x-coordinate of the circle center.
        y (int): The y-coordinate of the circle center.
        r (int): The radius of the circle.

    Returns:
        np.ndarray: The masked circular region.
    """
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.circle(mask, (x, y), r, 255, -1)
    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked[y - r : y + r, x - r : x + r]


def identify_rois(
    frame: np.ndarray, x: int, y: int, r: int
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Identify regions of interest (ROIs) in the frame.

    Args:
        frame (np.ndarray): The frame image.
        x (int): The x-coordinate of the circle center.
        y (int): The y-coordinate of the circle center.
        r (int): The radius of the circle.

    Returns:
        Optional[List[Tuple[int, int, int, int]]]: List of ROIs as tuples (x, y, w, h) or None if no ROIs are selected.
    """
    crop = generate_mask(frame, x, y, r)
    if len(crop) == 0:
        return None

    areas = []
    # Select ROI
    for i in range(3):
        r = cv2.selectROI(f"Select Area {i+1}", crop, showCrosshair=False)
        cv2.destroyAllWindows()
        if r == (0, 0, 0, 0):
            break
        areas.append(r)

    return areas


def process_region(
    frame: np.ndarray,
    circle: Tuple[int, int, int],
    rois: List[Tuple[int, int, int, int]],
) -> List[float]:
    """
    Process a region in the frame based on the given circle and ROIs.

    Args:
        frame (np.ndarray): The frame image.
        circle (Tuple[int, int, int]): The circle parameters (x, y, r).
        rois (List[Tuple[int, int, int, int]]): List of ROIs.

    Returns:
        List[float]: List of mean and sum values for each ROI.
    """
    x, y, r = circle
    region = generate_mask(frame, x, y, r)
    results = []
    for roi in rois:
        area = region[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ]
        results.append(np.mean(area, where=area > 0).item())
        results.append(np.sum(area).item())

    return results


def initialize_reference(ref_frame: np.ndarray) -> List[Dict[str, Tuple]]:
    """
    Initialize reference circles and ROIs in the reference frame.

    Args:
        ref_frame (np.ndarray): The reference frame image.

    Returns:
        List[Dict[str, Tuple]]: List of dictionaries containing circles and their ROIs.
    """
    sorted_circles = find_circles(ref_frame)
    reference = []

    for x, y, r in sorted_circles:
        cv2.circle(ref_frame, (x, y), r, (0, 255, 0), 2)
        cv2.imshow("Circle", ref_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        areas = identify_rois(ref_frame, x, y, r)
        if not areas:
            continue
        reference.append({"circle": (x, y, r), "rois": areas})

    return reference


def process_video(
    cap: cv2.VideoCapture,
    reference: List[Dict[str, Tuple]],
    output_file: str,
    start_time: int,
    end_time: int,
    skip_frames: int,
    fps: float,
) -> None:
    """
    Process the video and write the results to a CSV file.

    Args:
        cap (cv2.VideoCapture): Video capture object.
        reference (List[Dict[str, Tuple]]): List of reference circles and ROIs.
        output_file (str): The name of the output CSV file.
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.
        skip_frames (int): Number of frames to skip.
        fps (float): Frames per second of the video.
    """
    headers = ["Frame", "Time"]
    for r in range(len(reference)):
        for roi in range(3):
            headers.extend(
                [f"Arena {r+1} ROI {roi+1} Mean", f"Arena {r+1} ROI {roi+1} Sum"]
            )

    print(f"Frames per second: {fps}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame count: {frame_count}")
    print(f"Total time: {frame_count / fps} seconds")

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else frame_count

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    i = start_frame
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with tqdm(total=end_frame - start_frame) as pbar:
            while cap.isOpened() and i < end_frame:
                i += 1
                ret, frame = cap.read()
                if not ret:
                    break

                if not i % skip_frames == 0:
                    continue

                results = [i, i / fps]
                for region in reference:
                    result = process_region(frame, region["circle"], region["rois"])
                    results.extend(result)

                writer.writerow(results)
                pbar.update(skip_frames)


@click.command()
@click.argument("video")
@click.option("--output", default="results.csv", help="Name for output csv file")
@click.option("--start-time", default=0, help="Start time in seconds")
@click.option("--end-time", default=0, help="End time in seconds")
@click.option("--reference-time", default=0, help="Reference time in seconds")
@click.option("--skip-frames", default=1, help="Number of frames to skip")
def recognize(
    video: str,
    output: str,
    start_time: int,
    end_time: int,
    reference_time: int,
    skip_frames: int,
) -> None:
    """
    Recognize the regions of interest in a video.

    Args:
        video (str): Path to the video file.
        output (str): Name of the output CSV file.
        start_time (int): Start time in seconds.
        end_time (int): End time in seconds.
        reference_time (int): Reference time in seconds.
        skip_frames (int): Number of frames to skip.
    """
    try:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Read the reference frame at given time or first frame
        if reference_time > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(reference_time * fps))

        ret, ref_frame = cap.read()
        if not ret:
            exit(1)

        reference = initialize_reference(ref_frame)
        process_video(cap, reference, output, start_time, end_time, skip_frames, fps)
    except:
        raise
    finally:
        # Release the video objects
        cap.release()
        # Clean up the windows
        cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize()

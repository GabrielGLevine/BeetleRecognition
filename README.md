# BeetleRecognition Script

This script processes a video to recognize and analyze specific regions of interest (ROIs) using circle detection and manual ROI selection. The results are written to a CSV file.

## Requirements

- Python 3.7+
- OpenCV
- Click
- tqdm
- numpy

## Installation

Install the required packages using pip:

```bash
pipenv install
```

## Usage

Run the script with the following command:

```bash
python recognize.py <video_path> [options]
```

### Arguments

- `video`: Path to the input video file.

### Options

- `--output TEXT`: Name of the output CSV file (default: "results.csv").
- `--start-time INTEGER`: Start time in seconds (default: 0).
- `--end-time INTEGER`: End time in seconds (default: end of video).
- `--reference-time INTEGER`: Reference time in seconds (default: 0).
- `--skip-frames INTEGER`: Number of frames to skip. This is useful if you want to create a smaller output file but will only make the script marginally faster.

### Example

```bash
python recognize.py input_video.mp4 --output results.csv --start-time 10 --end-time 60 --reference-time 5 --skip-frames 5
```

## Functions

### `find_circles(ref_frame: np.ndarray) -> np.ndarray`

Find circles in the reference frame using the Hough Circle Transform.

**Arguments:**
- `ref_frame`: The reference frame image.

**Returns:**
- Sorted circles found in the reference frame.

### `generate_mask(frame: np.ndarray, x: int, y: int, r: int) -> np.ndarray`

Generate a mask for a circular region in the frame.

**Arguments:**
- `frame`: The frame image.
- `x`: The x-coordinate of the circle center.
- `y`: The y-coordinate of the circle center.
- `r`: The radius of the circle.

**Returns:**
- The masked circular region.

### `identify_rois(frame: np.ndarray, x: int, y: int, r: int) -> Optional[List[Tuple[int, int, int, int]]]`

Identify regions of interest (ROIs) in the frame.

**Arguments:**
- `frame`: The frame image.
- `x`: The x-coordinate of the circle center.
- `y`: The y-coordinate of the circle center.
- `r`: The radius of the circle.

**Returns:**
- List of ROIs as tuples (x, y, w, h) or None if no ROIs are selected.

### `process_region(frame: np.ndarray, circle: Tuple[int, int, int], rois: List[Tuple[int, int, int, int]]) -> List[float]`

Process a region in the frame based on the given circle and ROIs.

**Arguments:**
- `frame`: The frame image.
- `circle`: The circle parameters (x, y, r).
- `rois`: List of ROIs.

**Returns:**
- List of mean and sum values for each ROI.

### `initialize_reference(ref_frame: np.ndarray) -> List[Dict[str, Tuple]]`

Initialize reference circles and ROIs in the reference frame.

**Arguments:**
- `ref_frame`: The reference frame image.

**Returns:**
- List of dictionaries containing circles and their ROIs.

### `process_video(cap: cv2.VideoCapture, reference: List[Dict[str, Tuple]], output_file: str, start_time: int, end_time: int, skip_frames: int, fps: float) -> None`

Process the video and write the results to a CSV file.

**Arguments:**
- `cap`: Video capture object.
- `reference`: List of reference circles and ROIs.
- `output_file`: The name of the output CSV file.
- `start_time`: Start time in seconds.
- `end_time`: End time in seconds.
- `skip_frames`: Number of frames to skip.
- `fps`: Frames per second of the video.

### `recognize(video: str, output: str, start_time: int, end_time: int, reference_time: int, skip_frames: int) -> None`

Recognize the regions of interest in a video.

**Arguments:**
- `video`: Path to the video file.
- `output`: Name of the output CSV file.
- `start_time`: Start time in seconds.
- `end_time`: End time in seconds.
- `reference_time`: Reference time in seconds.
- `skip_frames`: Number of frames to skip.

## Notes

- The script will open windows to manually select ROIs (Regions of Interest). Select ROIs by dragging a box over the area and then hitting Space or Enter.
- The script uses the Hough Circle Transform to detect circles in the reference frame. It will occasionally select invalid circles. In that case, when prompted to select ROIs just click `c` to indicate you don't want to select a region. Any arenas with no regions selected will be skipped in the process.
- The script attempts to order the arenas in rows from top left to bottom right. However, where a circle is not fully aligned with a row it may be assigned in an odd place. Make a note when you're being shown what circle you're assigning ROIs for, so you have the right arena numbers for the results.
- Ensure the reference frame is properly initialized to get accurate results.

## License

This project is licensed under the MIT License. The author makes no guarantees as to the correctness of the data. It is presented as is.

Feel free to reach out to gabrielglevine@gmail.com for any issues or contributions.
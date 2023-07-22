import cv2
import time
import numpy as np
from ultralytics import YOLO
import argparse
from screeninfo import get_monitors

# Constants
extra_time = 0
priority_time = 0
reduced_time = 0
timer = 30 + extra_time + priority_time - reduced_time
model = YOLO('yolov8m.pt')
ambulance_cls = 0

def check_emergency_vehicle(frames):
    for i in range(4):
        try:
            if ambulance_cls in frames[i][0].boxes.cls:
                return True, i
        except:
            pass
    return False, 0

def count_vehicle(proc_frames):
    count = np.zeros(4, dtype=int)
    for i in range(4):
        try:
            count[i] = proc_frames[i][0].boxes.id.shape[0]
        except:
            pass
    return count

red = (0, 0, 255)
green = (0, 255, 0)
each_timer = [0, 0, 0, 0]


def update_signal_colors(signal_state):
    colors = [red for _ in range(4)]
    for i in range(4):
        if signal_state[i] == 1:
            colors[i] = green
    return colors

def main(video_paths, red, green, each_timer, extra_time, priority_time, reduced_time, timer, screen_size):
    # Video captures
    caps = [cv2.VideoCapture(path) for path in video_paths]

    # Check if video captures are opened successfully
    for cap in caps:
        if not cap.isOpened():
            print("Error opening video file:", cap)
            return

    # Read initial frames
    initial_frames = [cap.read()[1].copy() for cap in caps]

    signal_state = [1, 0, 0, 0]
    each_timer[0] = timer
    road = 0

    # Define camera labels
    camera_labels = ['Cam 1', 'Cam 2', 'Cam 3', 'Cam 4']

    while True:
        try:
            frames = [cap.read()[1] for cap in caps]
        except:
            frames = initial_frames.copy()

        start_time = time.time()

        # Resize frames based on screen size
        resized_frames = [cv2.resize(frame, screen_size) for frame in frames]

        proc_frames = [model.track(f, device=0) for f in resized_frames]
        processed_frames = [annotated[0].plot() for annotated in proc_frames]
        count = count_vehicle(proc_frames)

        if timer != 0:
            timer -= 1
            each_timer[road] = timer

        if timer == 0:
            emergency_vehicle, lane = check_emergency_vehicle(proc_frames)
            if not emergency_vehicle:
                road += 1
                road %= 4
            else:
                road = lane

            signal_state = [0, 0, 0, 0]
            each_timer = [0, 0, 0, 0]
            signal_state[road] = 1

            extra_time = 30 if count[road] > 50 else 15 if 20 < count[road] <= 50 else 0
            reduced_time = 20 if count[road] < 10 else 0
            timer = 30 + extra_time - reduced_time
            each_timer[road] = timer

        colors = update_signal_colors(signal_state)

        for i in range(4):
            cv2.putText(processed_frames[i],"Count "+str(int(count[i])),
                        (int(screen_size[0] * 0.8), int(screen_size[1] * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 5)
            cv2.putText(processed_frames[i], camera_labels[i], (int(screen_size[0] * 0.8), int(screen_size[1] * 0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 5)
            cv2.circle(processed_frames[i], (int(screen_size[0] * 0.1), int(screen_size[1] * 0.1)), int(screen_size[0] * 0.05),
                       colors[i], -1)

        horizontal_1 = np.concatenate((processed_frames[0], processed_frames[1]), axis=1)
        horizontal_2 = np.concatenate((processed_frames[2], processed_frames[3]), axis=1)
        vertical = np.concatenate((horizontal_1, horizontal_2), axis=0)
        resized_frame = cv2.resize(vertical, screen_size)

        
        # Replace the lines separating frames with rectangles
        cv2.rectangle(resized_frame, (0, 0), (int(screen_size[0] * 0.5), int(screen_size[1] * 0.5)), colors[0], 4)
        cv2.rectangle(resized_frame, (int(screen_size[0] * 0.5), 0), (int(screen_size[0]), int(screen_size[1] * 0.5)), colors[1], 4)
        cv2.rectangle(resized_frame, (0, int(screen_size[1] * 0.5)), (int(screen_size[0] * 0.5), int(screen_size[1])), colors[2], 4)
        cv2.rectangle(resized_frame, (int(screen_size[0] * 0.5), int(screen_size[1] * 0.5)), (int(screen_size[0]), int(screen_size[1])), colors[3], 4)


        cv2.imshow('Frame', resized_frame)
        processing_time = time.time() - start_time
        desired_delay = max(1, int((1000 / caps[0].get(cv2.CAP_PROP_FPS)) - processing_time))
        if cv2.waitKey(desired_delay) & 0xFF == ord('q'):
            break

    # Release video captures and destroy windows
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Traffic Signal Control')

    # Add arguments
    parser.add_argument('videos', nargs='+', help='Paths to video files')

    # Parse the command line arguments
    args = parser.parse_args()

    # Get the screen size
    screen_size = (get_monitors()[0].width, get_monitors()[0].height)

    # Call the main function with the provided video paths and screen size
    main(args.videos, red, green, each_timer, extra_time, priority_time, reduced_time, timer, screen_size)

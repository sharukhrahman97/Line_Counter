import cv2
import sys
import time
import queue
import torch
import numpy as np
from pathlib import Path
import openvino as ov
from ultralytics.solutions import ObjectCounter
from ultralytics import YOLO
from ultralytics import YOLO
from utils.video import MulticamCapture
from utils.visualization import get_target_size
from threading import Thread


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


class RunInference:
    def __init__(self, inputs, device="CPU", out_feed=False):
        self.sources = MulticamCapture(inputs, False)
        self.device = device
        self.out_feed = out_feed

        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        DET_MODEL_NAME = "yolov8n"

        self.det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
        label_map = self.det_model.model.names

        res = self.det_model()
        self.det_model_path = (
            models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
        )
        if not self.det_model_path.exists():
            self.det_model.export(format="openvino", dynamic=True, half=True)

        # assuming all feeds have same resolution
        cap_width = self.sources.captures[0].cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        cap_height = self.sources.captures[0].cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        line_points = [(0, cap_height / 2), (cap_width, cap_height / 2)]

        self.classes_to_count = [0]
        self.counter = [
            ObjectCounter(
                view_img=False,
                reg_pts=line_points,
                classes_names=self.det_model.names,
                draw_tracks=True,
                line_thickness=1,
                view_in_counts=True,
                view_out_counts=True,
            )
            for _ in range(len(self.sources.captures))
        ]

        self.thread_body = FramesThreadBody(
            self.sources, max_queue_length=len(self.sources.captures) * 2
        )
        self.frames_thread = Thread(target=self.thread_body)

    def make_inference(
        self,
        frames,
        max_window_size=(1920, 1080),
        stack_frames="vertical",
    ):
        vis = None
        for i in range(len(frames)):
            tracks = self.det_model.track(
                frames[i],
                persist=True,
                show=False,
                classes=self.classes_to_count,
                verbose=False,
            )
            frame = self.counter[i].start_counting(frames[i], tracks)

            if vis is not None:
                if stack_frames == "vertical":
                    vis = np.vstack([vis, frame])
                elif stack_frames == "horizontal":
                    vis = np.hstack([vis, frame])
            else:
                vis = frame

        target_width, target_height = get_target_size(
            frames, vis, max_window_size, stack_frames
        )
        return cv2.resize(vis, (target_width, target_height))

    def end_inference(self) -> tuple[int, int]:
        self.thread_body.process = False
        self.frames_thread.join()
        in_counts = []
        out_counts = []
        for i in range(len(self.sources.captures)):
            in_counts.append(self.counter[i].in_counts)
            out_counts.append(self.counter[i].out_counts)
            self.sources.captures[i].cap.release()
        cv2.destroyAllWindows()
        print("in_count", in_counts)
        print("out_count", out_counts)
        return (in_counts, out_counts)

    def start_inference(self):
        core = ov.Core()

        det_ov_model = core.read_model(self.det_model_path)
        ov_config = {}

        if self.device != "CPU":
            det_ov_model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in self.device or (
            "AUTO" in self.device and "GPU" in core.available_devices
        ):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        compiled_model = core.compile_model(det_ov_model, self.device, ov_config)

        def infer(*args):
            result = compiled_model(args)
            return torch.from_numpy(result[0])

        self.det_model.predictor.inference = infer
        self.det_model.predictor.model.pt = False

        self.frames_thread.start()

        prev_frames = self.thread_body.frames_queue.get()

        try:
            while self.thread_body.process:
                try:
                    frames = self.thread_body.frames_queue.get_nowait()
                except queue.Empty:
                    frames = None

                if frames is None:
                    continue

                frame = self.make_inference(frames=prev_frames)

                if self.out_feed:
                    cv2.imshow("queue", frame)

                prev_frames, frames = frames, prev_frames

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("Interrupted")

        self.end_inference()


def main():
    inputs = ["test/cam1.mp4", "test/cam2.mp4"]
    # inputs = ["rtsp://localhost:8554/cam1", "rtsp://localhost:8555/cam2"]
    RunInference(inputs=inputs, out_feed=True).start_inference()


if __name__ == "__main__":
    sys.exit(main() or 0)

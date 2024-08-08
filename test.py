from pathlib import Path
from ultralytics import YOLO
from ultralytics import YOLO
from ultralytics.solutions import ObjectCounter
import cv2
import time
import collections
import numpy as np
import torch
import openvino as ov


def run_inference(source, device: str, det_model: YOLO, det_model_path: Path):
    core = ov.Core()

    det_ov_model = core.read_model(det_model_path)
    ov_config = {}

    if device != "CPU":
        det_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(det_ov_model, device, ov_config)

    def infer(*args):
        result = compiled_model(args)
        return torch.from_numpy(result[0])

    det_model.predictor.inference = infer
    det_model.predictor.model.pt = False

    cap_width = None
    cap_height = None

    try:
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), "Error reading video file"

        if cap_width == None or cap_height == None:
            cap_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        line_points = [(0, cap_height / 2), (cap_width, cap_height / 2)]
        classes_to_count = [0]

        counter = ObjectCounter(
            view_img=False,
            reg_pts=line_points,
            classes_names=det_model.names,
            draw_tracks=True,
            line_thickness=2,
            view_in_counts=True,
            view_out_counts=True,
        )
        processing_times = collections.deque(maxlen=200)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            start_time = time.time()
            tracks = det_model.track(
                frame, persist=True, show=False, classes=classes_to_count, verbose=False
            )
            frame = counter.start_counting(frame, tracks)
            stop_time = time.time()

            processing_times.append(stop_time - start_time)

            _, f_width = frame.shape[:2]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            fontScale = 0.75
            fontFace = cv2.FONT_HERSHEY_COMPLEX
            thickness = 2

            # cv2.putText(
            #     img=frame,
            #     text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            #     org=(20, 40),
            #     fontFace=fontFace,
            #     fontScale=fontScale,
            #     color=(0, 0, 255),
            #     thickness=thickness,
            #     lineType=cv2.LINE_AA,
            # )

            in_counts = counter.in_counts
            out_counts = counter.out_counts

            in_count_text = f"in_Count: {in_counts}"
            out_count_text = f"out_Count: {out_counts}"

            (in_text_width, in_text_height), _ = cv2.getTextSize(
                in_count_text, fontFace, fontScale, thickness
            )
            (out_text_width, out_text_height), _ = cv2.getTextSize(
                out_count_text, fontFace, fontScale, thickness
            )

            in_top_right_corner = (frame.shape[1] - in_text_width - 20, 40)
            out_top_right_corner = (frame.shape[1] - out_text_width - 20, 80)
            # cv2.putText(
            #     img=frame,
            #     text=in_count_text,
            #     org=(in_top_right_corner[0], in_top_right_corner[1]),
            #     fontFace=fontFace,
            #     fontScale=fontScale,
            #     color=(0, 0, 255),
            #     thickness=thickness,
            #     lineType=cv2.LINE_AA,
            # )
            # cv2.putText(
            #     img=frame,
            #     text=out_count_text,
            #     org=(out_top_right_corner[0], out_top_right_corner[1]),
            #     fontFace=fontFace,
            #     fontScale=fontScale,
            #     color=(0, 0, 255),
            #     thickness=thickness,
            #     lineType=cv2.LINE_AA,
            # )

            cv2.imshow("Line_Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    DET_MODEL_NAME = "yolov8n"

    det_model = YOLO(models_dir / f"{DET_MODEL_NAME}.pt")
    label_map = det_model.model.names

    res = det_model()
    det_model_path = (
        models_dir / f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml"
    )
    if not det_model_path.exists():
        det_model.export(format="openvino", dynamic=True, half=True)

    WEBCAM_INFERENCE = False

    if WEBCAM_INFERENCE:
        VIDEO_SOURCE = 0
    else:
        VIDEO_SOURCE = "./test/cam1.mp4"
        # VIDEO_SOURCE = "rtsp://localhost:8554/cam1"

    run_inference(
        source=VIDEO_SOURCE,
        device="CPU",
        det_model=det_model,
        det_model_path=det_model_path,
    )

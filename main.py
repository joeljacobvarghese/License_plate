import os
import cv2
import torch
import numpy as np
from collections import Counter
from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference

license_plate_details = {
    "DL10CG4057": {"name": "John Doe", "car": "Toyota Innova"},
    "UP37U3276": {"name": "Augustin Magnus", "car": "Maruti Alto"},
}

def run_single_frame(od_model, lprnet, image):
    original_image = image.copy()
    image = preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = od_model(image)
        scores, classes, boxes = out
        boxes = [
            [int(i[0]), int(i[1]), int(i[2]), int(i[3])]
            for i in boxes[0].cpu().numpy().tolist()
        ]
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()
    if len(boxes) == 0:
        return None
    plate_images = []
    for b in boxes:
        plate_image = original_image[b[1]: b[3], b[0]: b[2], :]
        im = cv2.resize(plate_image, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        plate_images.append(im)

    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images, 0))
    out_dict = {}

    for idx, (box, label) in enumerate(zip(boxes, plate_labels)):
        out_dict.update({idx: {"boxes": box, "label": label}})

    return out_dict

def plot_single_frame_from_out_dict(im, out_dict, line_thickness=3, color=(255, 0, 0)):
    if out_dict:
        for _, v in out_dict.items():
            box, label = v["boxes"], v["label"]
            if len(box) < 4:
                continue
            tl = (
                line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
            )
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            if label:
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
                cv2.putText(
                    im,
                    label,
                    (c1[0], c1[1] - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )
    return im

def create_video_with_status(video_path, od_model, lprnet, output_video_path, timestamps, num_frames=10):
    current_video = cv2.VideoCapture(video_path)
    fps = current_video.get(cv2.CAP_PROP_FPS)
    frame_width = int(current_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(current_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width * 2, frame_height)
    )
    total_frames = int(current_video.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps_in_frames = [int(ts * fps) for ts in timestamps]
    pause_duration = 1
    pause_frames = int(pause_duration * fps)

    frame_idx = 0
    while True:
        success, frame = current_video.read()
        if not success:
            break

        frame_idx += 1
        status_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255

        if frame_idx in timestamps_in_frames:
            cv2.putText(status_frame, "GATE SENSOR TRIGGERED", (50, int(frame_height // 3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            license_plate_counts = []
            frames_to_save = []
            for i in range(num_frames):
                current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx + i)
                success, snap_frame = current_video.read()
                if not success:
                    break
                out_dict = run_single_frame(od_model, lprnet, snap_frame)
                if out_dict:
                    snap_frame = plot_single_frame_from_out_dict(snap_frame, out_dict)
                    plate_labels = [v['label'] for v in out_dict.values()]
                    license_plate_counts.extend(plate_labels)
                    frames_to_save.append((snap_frame, out_dict))

            if license_plate_counts:
                most_common_plate = Counter(license_plate_counts).most_common(1)[0][0]
                car_details = license_plate_details.get(most_common_plate, {"name": "Unknown", "car": "Unknown"})
                for snap_frame, out_dict in frames_to_save:
                    for k, v in out_dict.items():
                        if v['label'] == most_common_plate:
                            plotted_image = plot_single_frame_from_out_dict(snap_frame, out_dict)
                            cv2.putText(status_frame, f"Plate: {most_common_plate}", (50, int(frame_height // 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(status_frame, f"Name: {car_details['name']}", (50, int(frame_height // 2) + 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(status_frame, f"Car: {car_details['car']}", (50, int(frame_height // 2) + 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                            cv2.putText(status_frame, "ENTRY AUTHORIZED", (50, int(frame_height // 2) + 180),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                            for _ in range(pause_frames):
                                out_video.write(np.hstack((plotted_image, status_frame)))
                            break
        else:
            out_video.write(np.hstack((frame, status_frame)))

    current_video.release()
    out_video.release()

if __name__ == "__main__":
    video_path = "Entry.mp4"
    output_video_path = "output_video.mp4"
    timestamps = [4, 12]

    od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    od_model.load_state_dict(
        torch.load(
            "weights/best_od.pth",
            map_location=torch.device("cpu"),
        )
    )

    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(
        torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
    )

    if torch.cuda.is_available():
        od_model = od_model.cuda()
        lprnet = lprnet.cuda()

    create_video_with_status(video_path, od_model, lprnet, output_video_path, timestamps)
    print("Processing done")

import os
import cv2
import torch
import json
import argparse
from tqdm import tqdm
from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference
import numpy as np

def run_single_frame(od_model, lprnet, image):
    original_image = image.copy()
    image = preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = od_model(image)
        boxes = [
            [int(i[0]), int(i[1]), int(i[2]), int(i[3])]
            for i in out[2][0].cpu().numpy().tolist()
        ]
    if len(boxes) == 0:
        return None
    plate_images = [
        torch.from_numpy(np.transpose(cv2.resize(original_image[b[1]:b[3], b[0]:b[2], :], (94, 24)).astype("float32") * 0.0078125 - 127.5, (2, 0, 1)))
        for b in boxes
    ]
    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images, 0))
    return {idx: {"boxes": box, "label": label} for idx, (box, label) in enumerate(zip(boxes, plate_labels))}

def plot_single_frame(im, out_dict, color=(255, 0, 0), line_thickness=3):
    for v in out_dict.values():
        box, label = v["boxes"], v["label"]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(im, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)
        if label:
            t_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3, thickness=max(line_thickness - 1, 1))[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, line_thickness / 3, [225, 255, 255], thickness=max(line_thickness - 1, 1), lineType=cv2.LINE_AA)
    return im

def process_video(video_path, od_model, lprnet, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 30
    final_dict = {}
    out_video = None

    print(f'Processing {video_path}...')
    for idx in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret or idx % frame_interval != 0:
            continue

        if out_video is None:
            out_video = cv2.VideoWriter(
                os.path.join(output_dir, "output.avi"),
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps / frame_interval,
                (frame.shape[1], frame.shape[0]),
            )

        out_dict = run_single_frame(od_model, lprnet, frame)
        if out_dict:
            final_dict[idx] = out_dict
            out_video.write(plot_single_frame(frame, out_dict))

    cap.release()
    if out_video:
        out_video.release()

    with open(os.path.join(output_dir, "output.json"), "w") as f:
        json.dump(final_dict, f)

    print(f"Processing of {video_path} complete. Output saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an MP4 video for license plate detection.")
    parser.add_argument("--source", type=str, required=True, help="Path to the .mp4 video file")
    parser.add_argument("--output_path", type=str, default="./", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    od_model.load_state_dict(torch.load("weights/best_od.pth", map_location=torch.device("cpu")))

    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu")))

    if torch.cuda.is_available():
        od_model = od_model.cuda()
        lprnet = lprnet.cuda()

    process_video(args.source, od_model, lprnet, args.output_path)

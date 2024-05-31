import os
import cv2
import numpy as np
import json
import argparse


class FrameExtractor:
    def __init__(self, video_paths, output_folders):
        self.video_paths = video_paths
        self.output_folders = output_folders

    def extract_frames_at_2fps(self):
        for folder in self.output_folders:
            os.makedirs(folder, exist_ok=True)

        videos = [cv2.VideoCapture(path) for path in self.video_paths]
        frame_rates = [video.get(cv2.CAP_PROP_FPS) for video in videos]
        skip_frames = [int(rate // 2) for rate in frame_rates]

        frame_count = 0
        while True:
            frames = []
            for video, skip in zip(videos, skip_frames):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
                for _ in range(skip - 1):
                    ret, _ = video.read()
                    if not ret:
                        break

            if len(frames) != len(videos):
                break

            for i, frame in enumerate(frames):
                output_path = os.path.join(self.output_folders[i], f"frame_{frame_count}.jpg")
                cv2.imwrite(output_path, frame)

            frame_count += 1

        for video in videos:
            video.release()


class HomographyGenerator:
    @staticmethod
    def detect_and_compute_keypoints(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    @staticmethod
    def match_keypoints(descriptors1, descriptors2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda x: x.distance)

    @staticmethod
    def find_homography(keypoints1, keypoints2, matches):
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return H

    @staticmethod
    def compute_homographies(images):
        base_image = images[0]
        base_keypoints, base_descriptors = HomographyGenerator.detect_and_compute_keypoints(base_image)
        homographies = [np.eye(3).tolist()]

        for image in images[1:]:
            keypoints, descriptors = HomographyGenerator.detect_and_compute_keypoints(image)
            matches = HomographyGenerator.match_keypoints(base_descriptors, descriptors)
            H = HomographyGenerator.find_homography(keypoints, base_keypoints, matches)
            homographies.append(H.tolist())

        return homographies


class MetadataGenerator:
    def __init__(self, sequence_num, video_info, homographies):
        self.sequence_num = sequence_num
        self.video_info = video_info
        self.homographies = homographies

    def generate_metainfo(self, output_path):
        metainfo = {
            f"sequence{self.sequence_num}": {
                "name": f"sequence{self.sequence_num}",
                "annot_fn_pattern": "",
                "video_fn_pattern": "",
                "cam_nbr": self.video_info["cam_nbr"],
                "video_frame_nbr": self.video_info["video_frame_nbr"],
                "valid_frames_range": self.video_info["valid_frames_range"],
                "frame_width": self.video_info["frame_width"],
                "frame_height": self.video_info["frame_height"],
                "homography": self.homographies,
                "train_ratio": 0.9,
                "eval_ratio": 0.2,
                "test_ratio": 0.1
            }
        }
        with open(output_path, 'w') as outfile:
            json.dump(metainfo, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video folder for frame extraction and metadata generation.")
    parser.add_argument("videos_folder", type=str, help="Path to the folder containing video files.")
    parser.add_argument("sequence_num", type=int, help="Sequence number for naming the output folders.")
    args = parser.parse_args()

    output_folder = "./datasets/SelfDataset"
    videos_folder = args.videos_folder
    sequence_num = args.sequence_num
    video_paths = [os.path.join(videos_folder, f) for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.mov', '.avi'))]
    output_folders = [os.path.join(output_folder,f"sequence{sequence_num}", "src", "Image_subsets", f"C{idx+1}") for idx in range(len(video_paths))]

    frame_extractor = FrameExtractor(video_paths, output_folders)
    frame_extractor.extract_frames_at_2fps()

    images = []
    for path in video_paths:
        video = cv2.VideoCapture(path)
        ret, frame = video.read()
        if ret:
            images.append(frame)
        video.release()
    
    homographies = HomographyGenerator.compute_homographies(images)
    # images = [cv2.imread(os.path.join(folder, os.listdir(folder)[0])) for folder in output_folders]
    images.clear()
    images = [cv2.imread(os.path.join(folder, f)) for folder in output_folders for f in os.listdir(folder)]
    no_cams = len(video_paths)
    no_frames = len(images) // no_cams

    video_info = {
        "cam_nbr": no_cams,
        "video_frame_nbr": no_frames,
        "valid_frames_range": [0, no_frames],
        "frame_width": images[0].shape[1],
        "frame_height": images[0].shape[0],
    }
    output_filepath = "./datasets/SelfDataset/metainfo.json"
    metadata_generator = MetadataGenerator(sequence_num, video_info, homographies)
    metadata_generator.generate_metainfo(output_filepath)

"""
MediaPipe Holistic 키포인트 추출 스크립트

`--data-dir` 아래의 모든 영상을 재귀적으로 탐색하여 프레임 단위 Holistic 키포인트를 추출합니다.
결과는 원본 폴더 구조를 보존하여 `--keypoints-dir` 아래에 .npy로 저장합니다.

사용 예시:
  python process_data.py --data-dir video \
                         --keypoints-dir holistic_keypoints_data \
                         --overwrite

옵션:
  --overwrite  기존 출력이 있어도 덮어쓰기
"""


"""
python3 /Users/parkjaehyun/Desktop/AI캡스톤디자인/process_data.py \
  --data-dir /Users/parkjaehyun/Desktop/AI캡스톤디자인/clip \
  --keypoints-dir /Users/parkjaehyun/Desktop/AI캡스톤디자인/holistic_keypoints \
  --overwrite
"""

import os
import sys
import argparse
from typing import List, Tuple
from tqdm import tqdm
import numpy as np
import cv2
import os
os.environ['GLOG_minloglevel'] = '2'  # 0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR

try:
    import mediapipe as mp
except ImportError as exc:
    raise SystemExit(
        "mediapipe is required. Install with: pip install mediapipe"
    ) from exc


# 1. 기본 설정 및 상수 
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


def list_files_recursively(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    """루트 폴더를 재귀적으로 탐색하여 특정 확장자에 해당하는 파일 경로 목록을 반환합니다."""
    matched_paths: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(exts):
                matched_paths.append(os.path.join(dirpath, filename))
    matched_paths.sort()
    return matched_paths


def ensure_parent_dir(path: str) -> None:
    """파일 경로의 상위 디렉터리를 생성합니다(없으면 생성)."""
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)


def extract_video_keypoints(video_path: str, holistic: "mp.solutions.holistic.Holistic") -> np.ndarray:
    """단일 영상에서 프레임별 Holistic 키포인트를 추출하여 (프레임 수, 차원) 형태의 배열로 반환합니다."""
    cap = cv2.VideoCapture(video_path)
    frames_keypoints: List[np.ndarray] = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # 포즈(33, 4: x, y, z, visibility), 얼굴(468, 3), 왼손(21, 3), 오른손(21, 3)
        pose = (
            np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark], dtype=np.float32)
            if results.pose_landmarks is not None
            else np.zeros((33, 4), dtype=np.float32)
        )
        face = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32)
            if results.face_landmarks is not None
            else np.zeros((468, 3), dtype=np.float32)
        )
        left_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
            if results.left_hand_landmarks is not None
            else np.zeros((21, 3), dtype=np.float32)
        )
        right_hand = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
            if results.right_hand_landmarks is not None
            else np.zeros((21, 3), dtype=np.float32)
        )

        # 하나의 1D 벡터로 결합하여 프레임 리스트에 추가
        frame_keypoints = np.concatenate([
            pose.reshape(-1),
            face.reshape(-1),
            left_hand.reshape(-1),
            right_hand.reshape(-1),
        ])
        frames_keypoints.append(frame_keypoints)

    cap.release()

    if len(frames_keypoints) == 0:
        return np.zeros((0, 33 * 4 + 468 * 3 + 21 * 3 + 21 * 3), dtype=np.float32)

    return np.stack(frames_keypoints).astype(np.float32)


def extract_holistic_for_all_videos(data_dir: str, keypoints_dir: str, overwrite: bool = False) -> None:
    """1단계: `data_dir`의 모든 영상을 처리하여 Holistic 키포인트 .npy를 `keypoints_dir`에 저장합니다."""
    video_paths = list_files_recursively(data_dir, VIDEO_EXTENSIONS)
    if not video_paths:
        print(f"No videos found in '{data_dir}'. Supported: {VIDEO_EXTENSIONS}")
        return

    print(f"Found {len(video_paths)} videos under '{data_dir}'. Extracting keypoints → '{keypoints_dir}'")

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for video_path in tqdm(video_paths, desc="Extracting Holistic"):
            # 원본 경로 구조를 보존하여 .npy 저장 경로 구성
            rel_path = os.path.relpath(video_path, data_dir)
            save_path = os.path.join(
                keypoints_dir,
                os.path.splitext(rel_path)[0] + ".npy",
            )

            if not overwrite and os.path.exists(save_path):
                continue

            keypoints_seq = extract_video_keypoints(video_path, holistic)
            ensure_parent_dir(save_path)
            np.save(save_path, keypoints_seq)

    print(f"Keypoint extraction complete → '{keypoints_dir}'")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """CLI 인자들을 정의하고 파싱하여 반환합니다."""
    parser = argparse.ArgumentParser(description="Video → MediaPipe Holistic keypoints extractor")
    parser.add_argument("--data-dir", default="video", help="Root folder containing input videos (scanned recursively)")
    parser.add_argument("--keypoints-dir", default="holistic_keypoints_data", help="Output folder for keypoint .npy files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """엔드 투 엔드 실행 함수: 키포인트 추출을 수행합니다."""
    args = parse_args(argv)
    extract_holistic_for_all_videos(
        data_dir=args.data_dir,
        keypoints_dir=args.keypoints_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
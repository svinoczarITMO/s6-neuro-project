from fer import FER
import cv2
from loguru import logger
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def analyze_emotions(test_image):
    test_image = cv2.rotate(test_image, cv2.ROTATE_90_CLOCKWISE)

    emo_detector = FER(mtcnn=True)
    captured_emotions = emo_detector.detect_emotions(test_image)
    logger.debug(captured_emotions)

    if captured_emotions:
        return captured_emotions[0]['emotions']
    return None

def process_batch(batch_frames, batch_times):
    batch_emotions = {}
    batch_results = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_emotions, batch_frames))

    for frame, emotions, time in zip(batch_frames, results, batch_times):
        # logger.debug(emotions)
        if emotions:
            for emotion, value in emotions.items():
                batch_emotions[emotion] = batch_emotions.get(emotion, 0) + value
            batch_results.append((frame, emotions, time))

    if batch_emotions:
        dominant_emotion = max(batch_emotions, key=batch_emotions.get)
        dominant_frame = max(batch_results, key=lambda x: x[1].get(dominant_emotion, 0))

        return {
            "dominant_emotion": dominant_emotion,
            "emotion_details": batch_emotions,
            "dominant_frame_time": dominant_frame[2],
            "dominant_frame_emotion_details": dominant_frame[1],
            "dominant_frame": dominant_frame[0]
        }
    return None

def process_video(video_path, frames_per_second=5, batch_size=10, progress_callback=None):
    logger.info('Analyzing video...')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    batch_frames = []
    batch_times = []

    count = 0
    results = []
    while count < duration * frames_per_second:
        video.set(cv2.CAP_PROP_POS_MSEC, count * 1000 / frames_per_second)
        success, frame = video.read()
        if not success:
            break

        batch_frames.append(frame)
        batch_times.append(count / frames_per_second)
        count += 1

        if len(batch_frames) == batch_size:
            result = process_batch(batch_frames, batch_times)
            logger.info(result)
            if result:
                results.append(result)
            batch_frames = []
            batch_times = []

        if progress_callback:
            progress_callback(count, duration * frames_per_second)

    video.release()
    logger.info('Video analyzed...')
    return results

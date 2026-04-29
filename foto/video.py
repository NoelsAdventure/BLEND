import cv2
import requests
import numpy as np
import threading
import time
from datetime import datetime

time.sleep(30)

# Base URL and stream dictionary
base_url = "http://127.0.0.1"
# Max is 3
urls = {
    "detection_tflite": f"{base_url}/video_raw/detection_tflite",
}

# Shared stop signal
stop_event = threading.Event()

# Shared frame counters (mutable + thread-safe with a lock)
frame_counters = {key: 0 for key in urls}
counter_lock = threading.Lock()

# Function to process each stream
def process_stream(stream_url, stream_name):
    stream = None
    bytes_chunk = b""
    try:
        stream = requests.get(stream_url, stream=True, timeout=10)
        time.sleep(1/30)
        # Iterate over the streamed bytes until told to stop
        for chunk in stream.iter_content(chunk_size=1024):
            if stop_event.is_set():
                break
            if not chunk:
                continue

            bytes_chunk += chunk
            a = bytes_chunk.find(b'\xff\xd8')  # JPEG start
            b = bytes_chunk.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1 and b > a:
                jpg = bytes_chunk[a:b+2]
                bytes_chunk = bytes_chunk[b+2:]

                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    with counter_lock:
                        idx = frame_counters[stream_name]
                        frame_counters[stream_name] += 1

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(f"{stream_name}_frame_{idx:04d}_{ts}.jpg", img)

    except requests.exceptions.RequestException as e:
        print(f"[{stream_name}] Stream error: {e}")
    finally:
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

# Create and start a thread for each stream
threads = []
for stream_name, stream_url in urls.items():
    thread = threading.Thread(target=process_stream, args=(stream_url, stream_name), daemon=True)
    threads.append(thread)
    thread.start()

# Stop after 10 minutes
time.sleep(10 * 60)
stop_event.set()

# Wait briefly for threads to exit
for thread in threads:
    thread.join(timeout=5)

cv2.destroyAllWindows()

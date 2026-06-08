from pathlib import Path
import sys

# Lấy đường dẫn tuyệt đối của file hiện tại
file_path = Path(__file__).resolve()

# Lấy thư mục cha của file hiện tại
root_path = file_path.parent

# Thêm root_path vào sys.path nếu chưa có
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Lấy đường dẫn tương đối của thư mục gốc
ROOT = root_path.relative_to(Path.cwd())

# ──────────────────────────────────────────────
# Nguồn đầu vào
# ──────────────────────────────────────────────
IMAGE  = 'Image'
WEBCAM = 'Webcam'

SOURCES_LIST = [IMAGE, WEBCAM]

# ──────────────────────────────────────────────
# Cấu hình ảnh
# ──────────────────────────────────────────────
IMAGES_DIR            = ROOT / 'img'
DEFAULT_IMAGE         = IMAGES_DIR / 'imagedefault.jpg'
DEFAULT_DETECT_IMAGE  = IMAGES_DIR / 'imageclassfication.png'

# ──────────────────────────────────────────────
# Cấu hình mô hình ML
# ──────────────────────────────────────────────
MODEL_DIR       = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'

# ──────────────────────────────────────────────
# Webcam
# ──────────────────────────────────────────────
WEBCAM_PATH = 0          # 0 = webcam mặc định; đổi thành 1, 2… nếu có nhiều webcam

# ──────────────────────────────────────────────
# Giới hạn FPS webcam (để tránh CPU/GPU 100%)
# ──────────────────────────────────────────────
WEBCAM_FPS_LIMIT = 15    # Số frame tối đa xử lý mỗi giây

# ──────────────────────────────────────────────
# Cấu hình giọng nói (TTS)
# ──────────────────────────────────────────────
# True  → dùng pyttsx3 (offline, không cần internet)
# False → tắt giọng nói
TTS_ENABLED        = True
TTS_ANNOUNCE_EVERY = 3   # Thông báo lại sau mỗi N giây khi vẫn còn rác
# Dán THÊM 3 dòng này vào CUỐI file settings.py cũ của bạn
# (hoặc thay toàn bộ bằng file settings.py mới đã tải về)


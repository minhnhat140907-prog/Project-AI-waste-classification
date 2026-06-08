"""
helper.py – Các hàm hỗ trợ chính cho ứng dụng phân loại rác thải YOLO.

Sửa lỗi so với phiên bản cũ:
  1. load_model() giờ dùng đúng model_path được truyền vào (không hardcode nữa).
  2. play_webcam() có nút Dừng thực sự (dùng st.session_state).
  3. display_tracker_options() chuyển vào sidebar để giao diện gọn hơn.
  4. _display_detected_frames() trả về danh sách nhãn phát hiện để dùng cho TTS.
  5. Thêm FPS throttle để không ăn CPU/GPU 100%.
  6. Thêm giọng nói TTS (pyttsx3) thông báo khi phát hiện rác thải.
  7. Đảm bảo vid_cap.release() luôn được gọi dù có lỗi (dùng finally).
"""

from ultralytics import YOLO
import streamlit as st
import cv2
import time
import threading
import settings


# ──────────────────────────────────────────────────────────────────────────────
# TTS – Giọng nói thông báo (chạy nền, không chặn luồng chính)
# ──────────────────────────────────────────────────────────────────────────────

def _init_tts():
    """
    Khởi tạo engine TTS (pyttsx3) một lần duy nhất và lưu vào session_state.
    Nếu pyttsx3 chưa cài, trả về None và tắt TTS.
    """
    if "tts_engine" not in st.session_state:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)    # Tốc độ đọc (từ/phút)
            engine.setProperty('volume', 1.0)  # Âm lượng (0.0 – 1.0)
            st.session_state["tts_engine"] = engine
        except Exception:
            # pyttsx3 chưa cài hoặc không có driver âm thanh → tắt TTS
            st.session_state["tts_engine"] = None
    return st.session_state["tts_engine"]


def _speak_async(text: str):
    """
    Đọc `text` bằng giọng nói trong một thread riêng để không chặn luồng hiển thị.
    """
    if not settings.TTS_ENABLED:
        return
    engine = _init_tts()
    if engine is None:
        return

    def _run():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass  # Bỏ qua lỗi âm thanh, không crash app

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ──────────────────────────────────────────────────────────────────────────────
# Tải mô hình
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_path):
    """
    Tải mô hình YOLO từ đường dẫn được truyền vào.

    BUG CŨ: phiên bản trước hardcode './weights/best.pt' và bỏ qua model_path.
    FIX: dùng đúng model_path được truyền vào.

    Tham số:
        model_path (str | Path): Đường dẫn tới file trọng số YOLO (.pt).

    Trả về:
        YOLO: Đối tượng mô hình đã tải.
    """
    model = YOLO(str(model_path))   # ← FIX: dùng model_path thay vì hardcode
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Tùy chọn tracker (đặt trong sidebar)
# ──────────────────────────────────────────────────────────────────────────────

def display_tracker_options():
    """
    Hiển thị tùy chọn tracker trong SIDEBAR (phiên bản cũ đặt ở main area – sai).

    Trả về:
        is_display_tracker (bool): Có bật tracker hay không.
        tracker_type (str | None): Tên file cấu hình tracker, hoặc None.
    """
    # BUG CŨ: dùng st.radio() → nằm giữa trang chính, trông lộn xộn.
    # FIX: dùng st.sidebar.radio()
    display_tracker = st.sidebar.radio("🎯 Bật theo dõi đối tượng", ('Có', 'Không'))
    is_display_tracker = display_tracker == 'Có'

    if is_display_tracker:
        tracker_type = st.sidebar.radio(
            "📡 Chọn tracker",
            ("bytetrack.yaml", "botsort.yaml")
        )
        return is_display_tracker, tracker_type
    return is_display_tracker, None


# ──────────────────────────────────────────────────────────────────────────────
# Xử lý từng frame
# ──────────────────────────────────────────────────────────────────────────────

def _display_detected_frames(conf, model, st_frame, image,
                              is_display_tracking=False, tracker=None):
    """
    Phát hiện đối tượng trên một frame, vẽ kết quả lên st_frame và
    trả về danh sách tên lớp được phát hiện.

    Tham số:
        conf (float):              Ngưỡng độ tin cậy.
        model (YOLO):              Mô hình YOLO.
        st_frame (st.empty):       Vùng hiển thị Streamlit.
        image (np.ndarray):        Frame ảnh (BGR).
        is_display_tracking (bool):Có dùng tracker không.
        tracker (str | None):      Tên file tracker yaml.

    Trả về:
        list[str]: Danh sách tên nhãn phát hiện được (có thể rỗng).
    """
    # Resize về 720 × 405 (16:9) – chiều cao = 720*9/16 = 405
    h = int(720 * 9 / 16)  # = 405
    image = cv2.resize(image, (720, h))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    # Lấy tên các nhãn phát hiện được
    detected_labels = []
    if res and res[0].boxes is not None and len(res[0].boxes) > 0:
        class_ids = res[0].boxes.cls.tolist()
        detected_labels = [model.names[int(c)] for c in class_ids]

    # Vẽ bounding box lên frame và hiển thị
    res_plotted = res[0].plot()
    st_frame.image(
        res_plotted,
        caption='📹 Video trực tiếp – kết quả phát hiện',
        channels="BGR",
        use_container_width=True
    )

    return detected_labels


# ──────────────────────────────────────────────────────────────────────────────
# Webcam chính
# ──────────────────────────────────────────────────────────────────────────────

def play_webcam(conf, model):
    """
    Chạy webcam realtime với phát hiện YOLO.

    Cải tiến so với phiên bản cũ:
    - Nút Dừng hoạt động thật sự (dùng session_state).
    - FPS throttle (settings.WEBCAM_FPS_LIMIT) để không ăn CPU 100%.
    - Giọng nói TTS thông báo khi phát hiện rác thải.
    - vid_cap.release() được gọi trong finally → không bao giờ bị leak.
    - Hiển thị số lượng và tên đối tượng phát hiện ngay trên sidebar.
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()

    # Khởi tạo session_state để điều khiển vòng lặp
    if "webcam_running" not in st.session_state:
        st.session_state["webcam_running"] = False

    # ── Nút bắt đầu / dừng ────────────────────────────────────────────────────
    col_start, col_stop = st.sidebar.columns(2)
    with col_start:
        start_btn = st.button("▶️ Bắt đầu", use_container_width=True)
    with col_stop:
        stop_btn  = st.button("⏹️ Dừng",    use_container_width=True)

    if start_btn:
        st.session_state["webcam_running"] = True
    if stop_btn:
        st.session_state["webcam_running"] = False

    if not st.session_state["webcam_running"]:
        st.info("📷 Nhấn **Bắt đầu** để khởi động webcam.")
        return

    # ── Chạy webcam ───────────────────────────────────────────────────────────
    vid_cap = None
    try:
        # Đoạn code mới đã sửa:
        vid_cap = cv2.VideoCapture(source_webcam, cv2.CAP_DSHOW)
        if not vid_cap.isOpened():
            st.sidebar.error(
                "❌ Không thể mở webcam. "
                "Hãy kiểm tra kết nối thiết bị hoặc thay đổi WEBCAM_PATH trong settings.py."
            )
            st.session_state["webcam_running"] = False
            return

        st_frame       = st.empty()          # Vùng hiển thị frame
        info_box       = st.sidebar.empty()  # Thông tin số đối tượng phát hiện
        tts_cooldown   = 0.0                 # Thời điểm lần nói gần nhất

        frame_interval = 1.0 / settings.WEBCAM_FPS_LIMIT  # Giây giữa 2 frame

        while st.session_state.get("webcam_running", False):
            t0 = time.time()

            success, image = vid_cap.read()
            if not success:
                st.sidebar.warning("⚠️ Không đọc được frame. Đang thử lại…")
                time.sleep(0.1)
                continue

            detected_labels = _display_detected_frames(
                conf, model, st_frame, image,
                is_display_tracker, tracker
            )

            # ── Cập nhật thông tin sidebar ────────────────────────────────────
            if detected_labels:
                from collections import Counter
                counts = Counter(detected_labels)
                summary = "  \n".join(
                    f"• **{label}**: {cnt}" for label, cnt in counts.items()
                )
                info_box.markdown(
                    f"**🗑️ Phát hiện {len(detected_labels)} đối tượng:**  \n{summary}"
                )

                # ── Giọng nói TTS ─────────────────────────────────────────────
                now = time.time()
                if now - tts_cooldown >= settings.TTS_ANNOUNCE_EVERY:
                    label_text = ", ".join(counts.keys())
                    _speak_async(f"Phát hiện rác: {label_text}")
                    tts_cooldown = now
            else:
                info_box.markdown("✅ **Không phát hiện rác thải**")

            # ── Throttle FPS ──────────────────────────────────────────────────
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        st.sidebar.error(f"❌ Lỗi webcam: {e}")
        st.session_state["webcam_running"] = False
    finally:
        # BUG CŨ: release() chỉ được gọi trong nhánh success==False,
        # nếu có exception giữa chừng webcam sẽ bị giữ mãi.
        # FIX: luôn release trong finally.
        if vid_cap is not None:
            vid_cap.release()
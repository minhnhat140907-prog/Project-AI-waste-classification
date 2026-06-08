"""
app.py – Giao diện chính Streamlit cho ứng dụng phân loại rác thải YOLO.

Cải tiến so với phiên bản cũ:
  - Kiểm tra pyttsx3 ngay khi khởi động và hướng dẫn cài đặt nếu thiếu.
  - Hiển thị rõ trạng thái TTS (Giọng nói) trong sidebar.
  - Phần hiển thị kết quả ảnh được cải thiện: đếm từng loại rác,
    đọc to tên rác qua TTS ngay sau khi nhận diện ảnh tĩnh.
  - Cấu trúc rõ ràng hơn, tách biệt từng chế độ.
"""

from pathlib import Path
import PIL

import streamlit as st
import settings
print("settings path:", settings.__file__)
print("TTS_ENABLED:", settings.TTS_ENABLED)
import settings
import helper

# ──────────────────────────────────────────────────────────────────────────────
# Cấu hình trang
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Phân Loại Rác Thải – YOLO",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Tiêu đề
# ──────────────────────────────────────────────────────────────────────────────
st.title("♻️ Phân loại rác thải bằng YOLO")
st.markdown(
    "Ứng dụng phát hiện và phân loại rác thải theo thời gian thực "
    "sử dụng mô hình YOLOv8."
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Cấu hình mô hình
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Cấu hình mô hình")

model_type = st.sidebar.radio("🔍 Chế độ", ['Phát hiện'])
confidence = float(
    st.sidebar.slider("📊 Độ tin cậy (%)", 25, 100, 40)
) / 100

# Chọn đường dẫn mô hình
if model_type == 'Phát hiện':
    model_path = Path(settings.DETECTION_MODEL)

# Tải mô hình
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Không thể tải mô hình: `{model_path}`")
    st.error(ex)
    st.stop()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Trạng thái TTS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("🔊 Giọng nói (TTS)")

if settings.TTS_ENABLED:
    try:
        import pyttsx3
        st.sidebar.success("✅ Giọng nói: **Đang bật** (pyttsx3)")
    except ImportError:
        st.sidebar.warning(
            "⚠️ pyttsx3 chưa được cài đặt.  \n"
            "Chạy lệnh sau để cài:  \n"
            "```\npip install pyttsx3\n```"
        )
else:
    st.sidebar.info("🔇 Giọng nói: **Đã tắt** (xem settings.py → TTS_ENABLED)")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Chọn nguồn đầu vào
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.header("📸 Nguồn đầu vào")
source_radio = st.sidebar.radio("🖼️ Chọn nguồn", settings.SOURCES_LIST)

# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ ẢNH TĨNH
# ══════════════════════════════════════════════════════════════════════════════
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "📂 Tải ảnh lên…", type=("jpg", "jpeg", "png", "bmp", "webp")
    )

    col1, col2 = st.columns(2)

    # Cột trái – Ảnh gốc
    with col1:
        st.subheader("🖼️ Ảnh đầu vào")
        try:
            if source_img is None:
                st.image(
                    str(settings.DEFAULT_IMAGE),
                    caption="Ảnh mặc định",
                    use_container_width=True
                )
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Ảnh đã tải lên", use_container_width=True)
        except Exception as ex:
            st.error("⚠️ Không mở được ảnh gốc.")
            st.error(ex)

    # Cột phải – Kết quả phát hiện
    with col2:
        st.subheader("📍 Kết quả phát hiện")
        if source_img is None:
            st.image(
                str(settings.DEFAULT_DETECT_IMAGE),
                caption="Ảnh mẫu sau phát hiện",
                use_container_width=True
            )
        else:
            detect_btn = st.sidebar.button("🚀 Phát hiện đối tượng", use_container_width=True)

            if detect_btn:
                with st.spinner("🔄 Đang phân tích…"):
                    try:
                        res        = model.predict(uploaded_image, conf=confidence)
                        boxes      = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]   # BGR → RGB

                        st.image(res_plotted, caption="Kết quả", use_container_width=True)

                        # ── Chi tiết kết quả ──────────────────────────────────
                        with st.expander("📋 Chi tiết các đối tượng phát hiện", expanded=True):
                            if len(boxes) == 0:
                                st.info("Không phát hiện rác thải nào đạt ngưỡng tin cậy.")
                            else:
                                from collections import Counter

                                class_ids = boxes.cls.tolist()
                                label_list = [model.names[int(c)] for c in class_ids]
                                counts     = Counter(label_list)

                                st.markdown(f"**Tổng cộng: {len(boxes)} đối tượng**")
                                for lbl, cnt in counts.items():
                                    st.markdown(f"- 🗑️ **{lbl}**: {cnt}")

                                # ── TTS – đọc kết quả ─────────────────────────
                                tts_text = (
                                    f"Phát hiện {len(boxes)} đối tượng: "
                                    + ", ".join(f"{cnt} {lbl}" for lbl, cnt in counts.items())
                                )
                                helper._speak_async(tts_text)

                    except Exception as ex:
                        st.error("⚠️ Lỗi trong quá trình xử lý mô hình.")
                        st.error(ex)

# ══════════════════════════════════════════════════════════════════════════════
# CHẾ ĐỘ WEBCAM
# ══════════════════════════════════════════════════════════════════════════════
elif source_radio == settings.WEBCAM:
    st.subheader("📹 Phát hiện qua Webcam")
    st.markdown(
        "Nhấn **Bắt đầu** trong thanh bên để khởi động webcam.  \n"
        "Nhấn **Dừng** để tắt và giải phóng camera."
    )
    helper.play_webcam(confidence, model)

# ══════════════════════════════════════════════════════════════════════════════
# Nguồn không hợp lệ
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.error("⚠️ Vui lòng chọn loại nguồn hợp lệ!")
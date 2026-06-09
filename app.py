# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Cấu hình giao diện trang
st.set_page_config(
    page_title="Phân Loại Rác Thải bằng YOLO",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tiêu đề chính
st.title("🌍 Phân loại rác thải bằng YOLO")

# Thanh bên
st.sidebar.header("⚙️ Cấu hình mô hình học máy")

# Tùy chọn Mô hình
model_type = st.sidebar.radio("🔍 Chọn chế độ", ['Phát hiện'])
confidence = float(st.sidebar.slider("📊 Chọn độ tin cậy (%)", 25, 100, 40)) / 100

# Lựa chọn Phát hiện hoặc Phân đoạn
if model_type == 'Phát hiện':
    model_path = Path(settings.DETECTION_MODEL)

# Tải mô hình Máy Học đã Huấn luyện trước
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"❌ Không thể tải mô hình. Vui lòng kiểm tra đường dẫn: {model_path}")
    st.error(ex)
    st.stop() # Dừng chương trình nếu không tải được mô hình

# Cấu hình hình ảnh/video
st.sidebar.header("📸 Cấu hình ảnh/Video")
source_radio = st.sidebar.radio("🖼️ Chọn nguồn", settings.SOURCES_LIST)

source_img = None

# Nếu người dùng chọn tải ảnh
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "📂 Chọn một ảnh...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                st.image(default_image_path, caption="📎 Ảnh Mặc Định", use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="📎 Ảnh Đã Tải Lên", use_container_width=True)
        except Exception as ex:
            st.error("⚠️ Đã xảy ra lỗi khi mở ảnh gốc.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            st.image(default_detected_image_path, caption='📍 Ảnh mẫu sau phát hiện', use_container_width=True)
        else:
            # Tạo nút bấm để kích hoạt nhận diện
            detect_btn = st.sidebar.button('🚀 Phát hiện đối tượng')
            
            if detect_btn:
                with st.spinner("🔄 Đang phân tích hình ảnh..."):
                    try:
                        # Dự đoán từ mô hình YOLO
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1] # Chuyển đổi màu từ BGR sang RGB
                        
                        # Hiển thị ảnh đã vẽ bounding box
                        st.image(res_plotted, caption='📍 Kết quả phát hiện', use_container_width=True)
                        
                        # Hiển thị thông tin chi tiết các box phát hiện được
                        with st.expander("📋 Chi tiết các đối tượng phát hiện"):
                            if len(boxes) == 0:
                                st.info("Không phát hiện thấy rác thải nào đạt độ tin cậy yêu cầu.")
                            else:
                                for box in boxes:
                                    st.write(box.data)
                                    
                    except Exception as ex:
                        st.error("⚠️ Đã xảy ra lỗi trong quá trình xử lý mô hình.")
                        st.error(ex)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

else:
    st.error("⚠️ Vui lòng chọn loại nguồn hợp lệ!")
#streamlit run app.py

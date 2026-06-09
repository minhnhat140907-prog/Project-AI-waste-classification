from ultralytics import YOLO
import streamlit as st
import cv2
import pafy
import pickle
import settings
import os
from gtts import gTTS
import base64
import time
pafy.backend = "yt-dlp"
def speak_result(text):
    """
    Chuyển đổi văn bản thành giọng nói Tiếng Anh và hiển thị trình phát nhạc trên Sidebar.
    """
    try:
        # Tạo file âm thanh từ chữ bằng gTTS (Tiếng Anh)
        tts = gTTS(text=text, lang='en', slow=False)
        filename = "voice.mp3"
        tts.save(filename)
        
        # Đọc dữ liệu file âm thanh
        with open(filename, "rb") as f:
            audio_bytes = f.read()
            
        # Sử dụng trình phát âm thanh chính thức của Streamlit thay vì nhúng HTML ẩn
        # Cách này giúp bypass (vượt qua) lệnh chặn của trình duyệt 100%
        st.sidebar.audio(audio_bytes, format="audio/mp3", autoplay=True)
        
        # Xóa file tạm
        os.remove(filename)
    except Exception as e:
        pass
def load_model(model_path):
    """
    Tải mô hình phát hiện đối tượng YOLO từ đường dẫn được chỉ định.

    Tham số:
        model_path (str): Đường dẫn tới tệp mô hình YOLO.

    Trả về:
        Một mô hình phát hiện đối tượng YOLO.
    """
    model = YOLO('./weights/best.pt')
    return model

def display_tracker_options():
    """
    Hiển thị tùy chọn cho việc bật/tắt tính năng theo dõi đối tượng và lựa chọn kiểu theo dõi.

    Trả về:
        is_display_tracker (bool): Có bật tính năng theo dõi hay không.
        tracker_type (str): Loại trình theo dõi được chọn.
    """
    display_tracker = st.radio("Hiển thị trình theo dõi", ('Có', 'Không'))
    is_display_tracker = True if display_tracker == 'Có' else False
    if is_display_tracker:
        tracker_type = st.radio("Trình theo dõi", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Hiển thị các đối tượng được phát hiện và quản lý thời gian phát giọng nói (không bị nghẹn luồng).
    """
    # 1. Khởi tạo bộ đếm thời gian trễ trong bộ nhớ Streamlit (nếu chưa có)
    if 'last_speech_time' not in st.session_state:
        st.session_state.last_speech_time = 0

    # 2. Thay đổi kích thước hình ảnh về kích thước tiêu chuẩn
    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    # 3. Dự đoán hoặc theo dõi đối tượng bằng mô hình YOLO
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    # 4. Vẽ các bounding box lên khung hình (YOLO trả về hệ màu BGR)
    res_plotted = res[0].plot()
    
    # 5. Ép chuyển đổi hệ màu từ BGR sang RGB để tránh lỗi xanh màn hình
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)

    # 6. Hiển thị khung hình lên giao diện Streamlit
    st_frame.image(res_plotted_rgb,
                   caption='Video Detecting...',
                   channels="RGB",
                   width="stretch"
                   )

    # 7. Xử lý giọng nói có bộ lọc thời gian trễ (Cooldown)
    boxes = res[0].boxes
    if len(boxes) > 0:
        current_time = time.time()
        
        # Tăng thời gian trễ lên 5 giây để người nghe kịp nghe hết câu trước khi có câu mới
        if current_time - st.session_state.last_speech_time > 5.0:
            for box in boxes:
                cls_id = int(box.cls[0])
                label_name = model.names[cls_id]
                
                clean_label = label_name.replace("_", " ") 
                text_to_speak = f"Detected {clean_label}"
                
                # Tạo một khu vực hiển thị chữ và loa trực quan trên Sidebar
                st.sidebar.markdown("---")
                st.sidebar.markdown(f"### 🔊 **Scanning:** {clean_label.title()}")
                
                # Phát âm thanh ra loa (Sẽ hiện một thanh Player nhỏ ở Sidebar)
                speak_result(text_to_speak)
                
                # Cập nhật lại mốc thời gian vừa phát âm thanh
                st.session_state.last_speech_time = current_time
                break
def play_webcam(conf, model):
    """
    Phát luồng webcam thông minh: Tự động dò tìm camera phù hợp trên mọi máy tính.
    """
    is_display_tracker, tracker = display_tracker_options()
    
    if st.sidebar.button('Phát hiện rác thải'):
        # --- THUẬT TOÁN TỰ ĐỘNG DÒ TÌM CAMERA THÍCH NGHI ---
        vid_cap = None
        detected_source = None
        
        # Danh sách các nguồn camera khả thi: 
        # 0 (Webcam thật), 1 và 2 (DroidCam/Iriun ảo hoặc Cam USB)
        possible_sources = [0, 1, 2] 
        
        for source in possible_sources:
            try:
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    # Đọc thử 1 khung hình xem có bị lỗi màn hình xanh đặc không
                    success, test_frame = cap.read()
                    if success and test_frame is not None:
                        # Kiểm tra xem ảnh có phải là một màu xanh lá đặc (lỗi DroidCam) không
                        # Nếu giá trị trung bình kênh Green quá cao, các kênh khác bằng 0 -> Bỏ qua
                        mean_channels = cv2.mean(test_frame)
                        if mean_channels[1] > 200 and mean_channels[0] < 20 and mean_channels[2] < 20:
                            cap.release()
                            continue # Tìm camera tiếp theo
                        
                        vid_cap = cap
                        detected_source = source
                        break
                cap.release()
            except:
                continue

        # Nếu không tìm thấy camera phần cứng nào chạy tốt, thử quét luồng IP DroidCam mặc định
        if vid_cap is None:
            try:
                # Bạn có thể đổi địa chỉ IP này trùng với điện thoại hiện tại của bạn
                ip_url = "http://192.168.5.101:4747/video" 
                cap = cv2.VideoCapture(ip_url)
                if cap.isOpened():
                    vid_cap = cap
                    detected_source = "DroidCam IP Stream"
            except:
                pass

        # --- KIỂM TRA KẾT QUẢ DÒ TÌM ---
        if vid_cap is None or not vid_cap.isOpened():
            st.sidebar.error("❌ Không tìm thấy camera nào khả dụng! Vui lòng kết nối DroidCam hoặc kiểm tra webcam.")
            return
        else:
            st.sidebar.success(f"🎥 Đã kết nối thành công: Nguồn {detected_source}")

        # --- VÒNG LẶP HIỂN THỊ VIDEO ---
        try:
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                    # Thêm độ trễ nhỏ để tránh nghẽn luồng và đứng hình
                    cv2.waitKey(1)
                else:
                    break
            vid_cap.release()
        except Exception as e:
            st.sidebar.error("Lỗi khi chạy luồng video: " + str(e))
            if vid_cap:
                vid_cap.release()
# def play_stored_video(conf, model):
#     """
#     Phát video từ tệp đã lưu. Theo dõi và phát hiện các đối tượng trong thời gian thực bằng mô hình phát hiện đối tượng YOLOv8.

#     Tham số:
#         conf: Độ tin cậy của mô hình YOLOv8.
#         model: Một thực thể của lớp `YOLOv8` chứa mô hình YOLOv8.

#     Trả về:
#         Không trả về giá trị.

#     Gây ra:
#         Exception: Nếu có lỗi khi tải video.
#     """
#     source_vid = st.sidebar.selectbox(
#         "Chọn một video...", settings.VIDEOS_DICT.keys())

#     is_display_tracker, tracker = display_tracker_options()

#     with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
#         video_bytes = video_file.read()
#     if video_bytes:
#         st.video(video_bytes)

#     if st.sidebar.button('Phát hiện rác thải trong video'):
#         try:
#             vid_cap = cv2.VideoCapture(
#                 str(settings.VIDEOS_DICT.get(source_vid)))
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Lỗi khi tải video: " + str(e))

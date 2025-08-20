# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

# ========================
# 1. Config page
# ========================
st.set_page_config(page_title="🩺 OCR Bệnh Án", layout="wide")
st.title("🩺 OCR Bệnh Án - Full Pipeline")
st.write("Upload ảnh bệnh án → Tiền xử lý → OCR → Nhận dạng ký tự tiếng Việt")

# ========================
# 2. Utility Functions
# ========================
def auto_invert_if_needed(img):
    mean_intensity = np.mean(img)
    if mean_intensity > 127:
        return cv2.bitwise_not(img)
    return img

def noise_removal(image):
    return cv2.medianBlur(image, 3)

def thin_font(image):
    img = image.copy()
    kernel = np.ones((2,2), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def thick_font(image):
    img = image.copy()
    kernel = np.ones((2,2), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

def sharpness_score(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def best_sharp_image(img_list):
    scores = [sharpness_score(i) for i in img_list]
    return img_list[np.argmax(scores)]

def overlay_result(image, boxes, txts):
    for (box, txt) in zip(boxes, txts):
        pts = np.array(box).astype(int).reshape((-1,1,2))
        cv2.polylines(image, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(image, txt, tuple(pts[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return image

# ========================
# 3. Load Models
# ========================
@st.cache_resource
def load_models():
    ocr_det = PaddleOCR(lang='en')  # detection
    config = Cfg.load_config_from_name('vgg_transformer')
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    vietocr = Predictor(config)
    return ocr_det, vietocr

ocr_det, vietocr = load_models()

# ========================
# 4. Upload image
# ========================
uploaded_file = st.file_uploader("📂 Tải ảnh bệnh án", type=["png","jpg","jpeg","webp"])
if uploaded_file is not None:
    # Load ảnh
    img = Image.open(uploaded_file).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    st.image(cv_img, caption="Ảnh gốc", use_column_width=True)

    # ========================
    # 5. Preprocessing
    # ========================
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inverted = auto_invert_if_needed(gray)
    bw = cv2.adaptiveThreshold(inverted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)
    no_noise = noise_removal(bw)
    thin = thin_font(no_noise)
    thick = thick_font(no_noise)

    # chọn ảnh rõ nhất
    best = best_sharp_image([no_noise, thin, thick])

    col1, col2, col3, col4 = st.columns(4)
    col1.image(inverted, caption="Invert", use_column_width=True)
    col2.image(bw, caption="B/W", use_column_width=True)
    col3.image(no_noise, caption="No Noise", use_column_width=True)
    col4.image(best, caption="Best Sharp", use_column_width=True)

    # ========================
    # 6. Detection (PaddleOCR)
    # ========================
    result = ocr_det.ocr(best, cls=False)[0]
    boxes = [line[0] for line in result]
    crops = [best[int(min(y for x,y in box)):int(max(y for x,y in box)),
                 int(min(x for x,y in box)):int(max(x for x,y in box))] for box in boxes]

    # ========================
    # 7. Recognition (VietOCR)
    # ========================
    txts = []
    for crop in crops:
        if crop.size > 0:
            pil_crop = Image.fromarray(crop)
            txt = vietocr.predict(pil_crop)
            txts.append(txt)
        else:
            txts.append("")

    # ========================
    # 8. Overlay result
    # ========================
    overlay = cv_img.copy()
    overlay = overlay_result(overlay, boxes, txts)
    st.image(overlay, caption="Kết quả OCR", use_column_width=True)

    # ========================
    # 9. Xuất kết quả text
    # ========================
    st.subheader("📄 Văn bản OCR")
    st.write("\n".join(txts))

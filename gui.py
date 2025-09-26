"""
app_with_camera.py

Ứng dụng GUI (ttkbootstrap + tkinter) nhận dạng biển báo:
- Upload ảnh / Nhận dạng
- Live camera -> chụp ảnh -> nhận dạng
- Lịch sử (ảnh thu nhỏ + nhãn + timestamp + confidence + nút Mở/Xoá)
- Xuất CSV / Xoá lịch sử

Theme: superhero (dark)
Footer: thông tin nhóm / học phần / GVHD
"""

import os
import time
import threading
from datetime import datetime
import subprocess
import platform

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageOps
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel

from tensorflow.keras.models import load_model

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = r"D:\KI 1_2025-2026\NM_HOC_MAY\traffic-sign-classification\traffic-sign-classification-main\traffic_sign_model.h5"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

IMG_SIZE = (30, 30)            # kích thước input model
DISPLAY_SIZE = (300, 300)      # kích thước preview hiển thị lớn
CAM_PREVIEW_SIZE = (480, 360)  # kích thước stream preview nhỏ
CONF_THRESHOLD = 0.35

# ---------------------------
# Load model (sẽ in trạng thái)
# ---------------------------
print("⏳ Đang load model, vui lòng chờ...")
try:
    model = load_model(MODEL_PATH)
    print("✅ Model đã load thành công.")
except Exception as e:
    raise RuntimeError(f"Không thể load model từ {MODEL_PATH}\nLỗi: {e}")

# ---------------------------
# Labels
# ---------------------------
classes = {
    0: 'Giới hạn tốc độ (20km/h)',
    1: 'Giới hạn tốc độ (30km/h)',
    2: 'Giới hạn tốc độ (50km/h)',
    3: 'Giới hạn tốc độ (60km/h)',
    4: 'Giới hạn tốc độ (70km/h)',
    5: 'Giới hạn tốc độ (80km/h)',
    6: 'Hết giới hạn tốc độ (80km/h)',
    7: 'Giới hạn tốc độ (100km/h)',
    8: 'Giới hạn tốc độ (120km/h)',
    9: 'Cấm vượt',
    10: 'Cấm xe > 3.5 tấn vượt',
    11: 'Được ưu tiên tại ngã tư',
    12: 'Đường ưu tiên',
    13: 'Nhường đường',
    14: 'Dừng lại',
    15: 'Cấm xe',
    16: 'Cấm xe > 3.5 tấn',
    17: 'Cấm đi vào',
    18: 'Chú ý nguy hiểm',
    19: 'Đường cong nguy hiểm bên trái',
    20: 'Đường cong nguy hiểm bên phải',
    21: 'Đường cong kép',
    22: 'Đường gồ ghề',
    23: 'Đường trơn trượt',
    24: 'Đường hẹp bên phải',
    25: 'Đường đang thi công',
    26: 'Tín hiệu giao thông',
    27: 'Người đi bộ',
    28: 'Trẻ em qua đường',
    29: 'Xe đạp qua đường',
    30: 'Chú ý băng tuyết',
    31: 'Động vật hoang dã qua đường',
    32: 'Hết hạn chế tốc độ và cấm vượt',
    33: 'Rẽ phải phía trước',
    34: 'Rẽ trái phía trước',
    35: 'Chỉ đi thẳng',
    36: 'Đi thẳng hoặc rẽ phải',
    37: 'Đi thẳng hoặc rẽ trái',
    38: 'Đi bên phải',
    39: 'Đi bên trái',
    40: 'Bắt buộc đi vòng xuyến',
    41: 'Hết cấm vượt',
    42: 'Hết cấm vượt xe > 3.5 tấn'
}

# ---------------------------
# Helpers: open file cross-platform
# ---------------------------
def open_with_default_app(path):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path])
        else:
            subprocess.run(["xdg-open", path])
    except Exception as e:
        messagebox.showinfo("Mở ảnh", f"Không mở được ảnh: {e}\nĐường dẫn: {path}")

# ---------------------------
# Preprocess & Predict
# ---------------------------
def preprocess_pil(img_pil):
    """
    Tiền xử lý: convert RGB, resize to IMG_SIZE, normalize.
    Return numpy array shape (1, H, W, C)
    """
    img = img_pil.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_pil(img_pil):
    """
    Trả về (label_str, confidence_float)
    """
    arr = preprocess_pil(img_pil)
    preds = model.predict(arr)  # (1, n_classes)
    probs = preds[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = classes.get(idx, "Unknown")
    return label, conf

# ---------------------------
# App state
# ---------------------------
history = []      # list of dicts: {'thumb': PhotoImage, 'label':..., 'conf':..., 'time':..., 'path':...}
cam_thread = None
cam_running = False
cap = None

# UI globals (will be set in build_app)
top = None
display_image_label = None
label_result = None
history_list_frame = None

# ---------------------------
# Camera stream (threaded)
# ---------------------------
def start_camera_stream(label_widget, preview_size=CAM_PREVIEW_SIZE):
    global cam_running, cap
    cam_running = True
    # Try DirectShow backend on Windows for better compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        label_widget.after(0, lambda: messagebox.showerror("Camera lỗi", "Không thể mở camera. Kiểm tra driver/thiết bị."))
        cam_running = False
        return

    while cam_running:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # mirror for UX
        # For display: convert BGR->RGB and make PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil = ImageOps.fit(pil, preview_size, Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil)
        # update safe from thread
        label_widget.after(0, lambda im=imgtk, w=label_widget: (w.configure(image=im), setattr(w, "imgtk", im)))
        time.sleep(0.03)

    # cleanup
    if cap is not None:
        cap.release()
        cap = None

def stop_camera_stream():
    global cam_running
    cam_running = False

# ---------------------------
# GUI Action Handlers
# ---------------------------
def on_upload():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not path:
        return
    try:
        pil = Image.open(path).convert("RGB")
    except Exception as e:
        messagebox.showerror("File lỗi", f"Không thể mở ảnh: {e}")
        return

    # scale/crop to a consistent large preview so the sign is visible
    display = ImageOps.fit(pil, DISPLAY_SIZE, Image.LANCZOS)
    top.current_pil_image = pil  # store original (or converted) for prediction
    top.current_image_path = path

    imgtk = ImageTk.PhotoImage(display)
    display_image_label.configure(image=imgtk)
    display_image_label.image = imgtk
    label_result.configure(text="Ảnh đã tải lên. Nhấn 'Nhận dạng' để phân loại.", bootstyle=INFO)

def on_open_camera():
    global cam_thread, cam_running
    if cam_running:
        stop_camera_stream()
        label_result.configure(text="Camera dừng.", bootstyle=SECONDARY)
        return
    cam_thread = threading.Thread(target=start_camera_stream, args=(display_image_label,), daemon=True)
    cam_thread.start()
    label_result.configure(text="Camera đang hoạt động. Nhấn 'Chụp từ Camera' để chụp ảnh.", bootstyle=INFO)

def on_capture_from_camera():
    """
    Chụp frame hiện tại từ cap, hiển thị lớn và lưu vào top.current_pil_image để nhận dạng.
    """
    global cap
    if cap is None:
        messagebox.showwarning("Camera", "Camera chưa mở. Nhấn 'Mở Camera' trước.")
        return
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Lỗi camera", "Không thể chụp ảnh từ camera.")
        return
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    display = ImageOps.fit(pil, DISPLAY_SIZE, Image.LANCZOS)
    top.current_pil_image = pil
    top.current_image_path = None

    imgtk = ImageTk.PhotoImage(display)
    display_image_label.configure(image=imgtk)
    display_image_label.image = imgtk
    label_result.configure(text="Ảnh đã chụp từ camera. Nhấn 'Nhận dạng' để phân loại.", bootstyle=INFO)

def on_recognize_current():
    if getattr(top, "current_pil_image", None) is None:
        messagebox.showwarning("Chưa có ảnh", "Vui lòng upload ảnh hoặc chụp ảnh từ camera trước.")
        return
    try:
        label_text, conf = predict_pil(top.current_pil_image)
    except Exception as e:
        messagebox.showerror("Lỗi dự đoán", f"Không thể dự đoán: {e}")
        return

    conf_pct = conf * 100.0
    if conf < CONF_THRESHOLD:
        display = f"🔎 Không chắc chắn ({conf_pct:.1f}%). Kết quả: {label_text}"
        label_result.configure(text=display, bootstyle=WARNING)
    else:
        display = f"✅ {label_text}  —  Độ tin cậy: {conf_pct:.1f}%"
        label_result.configure(text=display, bootstyle=SUCCESS)

    # save record to history (save full-size captured/uploaded image)
    tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"capture_{tstamp}.png"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        top.current_pil_image.save(save_path)
    except Exception:
        top.current_pil_image.convert("RGB").save(save_path)

    thumb = top.current_pil_image.copy()
    thumb.thumbnail((80, 80), Image.LANCZOS)
    thumb_tk = ImageTk.PhotoImage(thumb)
    entry = {
        'thumb': thumb_tk,
        'label': label_text,
        'conf': conf,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'path': save_path
    }
    history.insert(0, entry)
    refresh_history_panel()

def refresh_history_panel():
    # clear
    for w in history_list_frame.winfo_children():
        w.destroy()
    # repopulate
    for i, entry in enumerate(history, start=1):
        frame = ttk.Frame(history_list_frame, bootstyle="light", padding=6)
        frame.pack(fill="x", pady=6, padx=6)

        img_lbl = ttk.Label(frame)
        img_lbl.configure(image=entry['thumb'])
        img_lbl.image = entry['thumb']
        img_lbl.pack(side="left")

        txt = f"Lần {i}: {entry['label']}\n{entry['time']}\n{entry['conf']*100:.1f}%"
        ttk.Label(frame, text=txt, font=("Helvetica", 10), wraplength=220).pack(side="left", padx=8)

        btns = ttk.Frame(frame)
        btns.pack(side="right")
        ttk.Button(btns, text="Mở", bootstyle=PRIMARY, width=6, command=lambda p=entry['path']: open_with_default_app(p)).pack(pady=2)
        ttk.Button(btns, text="Xoá", bootstyle=DANGER, width=6, command=lambda e=entry: delete_history_entry(e)).pack(pady=2)

def delete_history_entry(entry):
    try:
        if os.path.exists(entry['path']):
            os.remove(entry['path'])
    except Exception:
        pass
    try:
        history.remove(entry)
    except ValueError:
        pass
    refresh_history_panel()

def on_clear_history():
    if not history:
        return
    if messagebox.askyesno("Xoá lịch sử", "Bạn có chắc muốn xoá toàn bộ lịch sử?"):
        for e in history:
            try:
                if os.path.exists(e['path']):
                    os.remove(e['path'])
            except Exception:
                pass
        history.clear()
        refresh_history_panel()
        label_result.configure(text="Đã xoá lịch sử.", bootstyle=SECONDARY)

def on_show_history():
    if not history:
        messagebox.showinfo("Lịch sử", "Chưa có dữ liệu nhận dạng.")
        return
    win = Toplevel(top)
    win.title("📜 Lịch sử nhận dạng (chi tiết)")
    win.geometry("700x600")

    frame = ttk.Frame(win, padding=8)
    frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(frame, bg=win.cget("bg"))
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    inner = ttk.Frame(canvas)
    inner_id = canvas.create_window((0,0), window=inner, anchor="nw")

    def on_config(e):
        canvas.configure(scrollregion=canvas.bbox("all"))
    inner.bind("<Configure>", on_config)

    # populate
    for i, entry in enumerate(history, start=1):
        row = ttk.Frame(inner, padding=6)
        row.pack(fill="x", pady=6)
        limg = ttk.Label(row)
        limg.configure(image=entry['thumb'])
        limg.image = entry['thumb']
        limg.pack(side="left")
        txt = f"Lần {i}: {entry['label']}\n{entry['time']}\nĐộ tin cậy: {entry['conf']*100:.1f}%"
        ttk.Label(row, text=txt, font=("Helvetica", 11)).pack(side="left", padx=8)
        ttk.Button(row, text="Mở ảnh", bootstyle=INFO, command=lambda p=entry['path']: open_with_default_app(p)).pack(side="right")

def on_save_history_csv():
    if not history:
        messagebox.showinfo("Lưu CSV", "Không có mục lịch sử để lưu.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV file","*.csv")])
    if not save_path:
        return
    rows = [{'time': e['time'], 'label': e['label'], 'confidence': float(e['conf']), 'image_path': e['path']} for e in history]
    try:
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        messagebox.showinfo("Lưu CSV", f"Đã lưu {len(rows)} mục vào {save_path}")
    except Exception as ex:
        messagebox.showerror("Lỗi lưu CSV", str(ex))

# ---------------------------
# Build GUI
# ---------------------------
def build_app():
    global top, display_image_label, label_result, history_list_frame

    # use dark theme superhero
    top = ttk.Window(themename="superhero")
    top.title("🚦 Nhận dạng biển báo giao thông (Camera + Upload)")
    top.geometry("1200x800")

    # Header / top bar
    header = ttk.Frame(top, padding=10)
    header.pack(fill="x", padx=16, pady=8)

    ttk.Label(header, text="🚦 Nhận dạng biển báo giao thông", font=("Helvetica", 20, "bold"), bootstyle=INFO).pack(side="left")
    btn_frame = ttk.Frame(header)
    btn_frame.pack(side="right")
    ttk.Button(btn_frame, text="📂 Upload", bootstyle=WARNING, command=on_upload).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="📷 Mở Camera", bootstyle=PRIMARY, command=on_open_camera).pack(side="left", padx=6)
    ttk.Button(btn_frame, text="📜 Lịch sử", bootstyle=SECONDARY, command=on_show_history).pack(side="left", padx=6)

    # Main content (left preview + right history)
    content = ttk.Frame(top)
    content.pack(fill="both", expand=True, padx=16, pady=10)

    left = ttk.Frame(content)
    left.pack(side="left", fill="both", expand=True)

    preview_card = ttk.Frame(left, padding=12)
    preview_card.pack(fill="both", expand=True, pady=8)

    ttk.Label(preview_card, text="Preview", font=("Helvetica", 14, "bold")).pack(anchor="nw")

    # large preview
    display_image_label = ttk.Label(preview_card, bootstyle="dark", width=DISPLAY_SIZE[0])
    display_image_label.pack(pady=8)

    label_result = ttk.Label(preview_card, text="Chưa có dữ liệu", font=("Helvetica", 14))
    label_result.pack(pady=8)

    controls = ttk.Frame(preview_card)
    controls.pack(pady=6)
    ttk.Button(controls, text="🔍 Nhận dạng (Ảnh đang hiển thị)", bootstyle=SUCCESS, command=on_recognize_current).pack(side="left", padx=6)
    ttk.Button(controls, text="📸 Chụp từ Camera", bootstyle=INFO, command=on_capture_from_camera).pack(side="left", padx=6)
    ttk.Button(controls, text="🗑️ Xoá lịch sử", bootstyle=DANGER, command=on_clear_history).pack(side="left", padx=6)
    ttk.Button(controls, text="💾 Lưu lịch sử (CSV)", bootstyle=SECONDARY, command=on_save_history_csv).pack(side="left", padx=6)

    # right panel - history quick list
    right = ttk.Frame(content, width=360)
    right.pack(side="right", fill="y")
    ttk.Label(right, text="Lịch sử gần đây", font=("Helvetica", 14, "bold")).pack(anchor="nw", pady=6)
    history_canvas = ttk.Frame(right)
    history_canvas.pack(fill="both", expand=True)
    history_list_frame = tk.Frame(history_canvas, bg=top.cget("bg"))
    history_list_frame.pack(fill="both", expand=True)

       # footer với nền nổi bật và chữ dễ đọc hơn
    sep = ttk.Separator(top, orient="horizontal")
    sep.pack(fill="x", padx=16, pady=8)

    footer = ttk.Frame(top, padding=12, bootstyle="dark")  # nền tối khác biệt
    footer.pack(fill="x", padx=16, pady=(0,12))

    ttk.Label(
        footer,
        text="👨‍💻 Nhóm 16 - CNTT 17-07  |  Thành viên: Trần Văn Lâm, Nguyễn Văn Thuyết",
        font=("Helvetica", 12, "bold"),
        bootstyle="success"
    ).pack(pady=2)

    ttk.Label(
        footer,
        text="📘 Học phần: Nhập môn Học máy  |  GVHD: ThS. Lê Thị Thùy Trang",
        font=("Helvetica", 12, "bold"),
        bootstyle="info"
    ).pack(pady=2)


    # internal state
    top.current_pil_image = None
    top.current_image_path = None

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    build_app()
    top.protocol("WM_DELETE_WINDOW", lambda: (stop_camera_stream(), top.destroy()))
    top.mainloop()

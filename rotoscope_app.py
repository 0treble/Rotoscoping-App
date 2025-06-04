import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# === Custom Tooltip Class ===
class CustomToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self.show_tip)
        widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tip_window or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify='left', background="#333", foreground="white", relief='solid', borderwidth=1, font=("Arial", 9))
        label.pack(ipadx=5, ipady=3)

    def hide_tip(self, event=None):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

import tkinter as tk

class RotoscopeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Rotoscope App")
        self.geometry("1200x800")

        self.input_path = None
        self.original_img = None
        self.processed_img = None

        # Config defaults
        self.config = {
            "line_thickness": 2,
            "frame_step": 3,
            "output_fps": 10,
            "use_canny": True,
            "overlay_on_original": True,
            "use_sharpening": True,
            "bilateral_sigma_color": 50,
            "bilateral_sigma_space": 50,
            "scharr_channel_weights": [1.0, 1.0, 1.0],
            "canny_channel_weights": [1.0, 1.0, 1.0]
        }

        # === UI LAYOUT ===
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        self.preview_frame = ctk.CTkFrame(self.main_frame)
        self.preview_frame.pack(side="left", fill="both", expand=True)

        self.controls_frame = ctk.CTkScrollableFrame(self.main_frame, width=300)
        self.controls_frame.pack(side="right", fill="y")

        self.original_canvas = ctk.CTkLabel(self.preview_frame, text="Original")
        self.processed_canvas = ctk.CTkLabel(self.preview_frame, text="Processed")
        self.original_canvas.pack(pady=10)
        self.processed_canvas.pack(pady=10)

        # === CONTROL PANEL ===
        ctk.CTkButton(self.controls_frame, text="Load Video", command=self.load_video).pack(pady=10)

        self.create_slider("Line Thickness", "line_thickness", 1, 10, "Thickness of contour lines")
        self.create_slider("Output FPS", "output_fps", 1, 30, "Frames per second in exported video")
        self.create_slider("Frame Step", "frame_step", 1, 10, "How often frames are processed")

        self.create_slider("Bilateral Sigma Color", "bilateral_sigma_color", 1, 150, "Color sensitivity of bilateral filter")
        self.create_slider("Bilateral Sigma Space", "bilateral_sigma_space", 1, 150, "Spatial blur range of bilateral filter")

        self.create_slider("Canny Red Weight", "canny_channel_weights", 0, 2, "Weight of red channel in Canny", index=0)
        self.create_slider("Canny Green Weight", "canny_channel_weights", 0, 2, "Weight of green channel in Canny", index=1)
        self.create_slider("Canny Blue Weight", "canny_channel_weights", 0, 2, "Weight of blue channel in Canny", index=2)

        self.create_slider("Scharr Red Weight", "scharr_channel_weights", 0, 2, "Weight of red channel in Scharr", index=0)
        self.create_slider("Scharr Green Weight", "scharr_channel_weights", 0, 2, "Weight of green channel in Scharr", index=1)
        self.create_slider("Scharr Blue Weight", "scharr_channel_weights", 0, 2, "Weight of blue channel in Scharr", index=2)

        self.canny_switch = ctk.CTkSwitch(self.controls_frame, text="Use Canny", command=self.toggle_canny)
        self.canny_switch.select()
        self.canny_switch.pack(pady=10)
        CustomToolTip(self.canny_switch, "Switch between Canny and Scharr")

        self.overlay_switch = ctk.CTkSwitch(self.controls_frame, text="Overlay on Original", command=self.toggle_overlay)
        self.overlay_switch.select()
        self.overlay_switch.pack(pady=10)
        CustomToolTip(self.overlay_switch, "Overlay edges over the original image")

        self.sharpen_switch = ctk.CTkSwitch(self.controls_frame, text="Use Sharpening", command=self.toggle_sharpening)
        self.sharpen_switch.select()
        self.sharpen_switch.pack(pady=10)
        CustomToolTip(self.sharpen_switch, "Apply sharpening before edge detection")

        ctk.CTkButton(self.controls_frame, text="Export Video", command=self.export_video).pack(pady=20)
        
    def create_slider(self, label_text, config_key, min_val, max_val, tooltip_text, index=None):
        label = ctk.CTkLabel(self.controls_frame, text=label_text)
        label.pack()
        slider = ctk.CTkSlider(self.controls_frame, from_=min_val, to=max_val, command=lambda v, k=config_key, i=index: self.slider_update(v, k, i))
        slider.set(self.config[config_key][index] if index is not None else self.config[config_key])
        slider.pack(pady=5)
        CustomToolTip(slider, tooltip_text)

    def slider_update(self, value, key, index):
        value = float(value)
        if index is not None:
            self.config[key][index] = value
        else:
            self.config[key] = int(value)
        self.update_preview()

    def toggle_canny(self):
        self.config["use_canny"] = self.canny_switch.get()
        self.update_preview()

    def toggle_overlay(self):
        self.config["overlay_on_original"] = self.overlay_switch.get()
        self.update_preview()

    def toggle_sharpening(self):
        self.config["use_sharpening"] = self.sharpen_switch.get()
        self.update_preview()

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if path:
            self.input_path = path
            cap = cv2.VideoCapture(self.input_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.original_img = frame
                self.update_preview()

    def update_preview(self):
        if self.original_img is None:
            return

        frame = self.original_img.copy()
        if self.config["use_sharpening"]:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)

        b, g, r = cv2.split(frame)

        if self.config["use_canny"]:
            edges_r = self.process_channel(r, "canny", self.config["canny_channel_weights"][0])
            edges_g = self.process_channel(g, "canny", self.config["canny_channel_weights"][1])
            edges_b = self.process_channel(b, "canny", self.config["canny_channel_weights"][2])
        else:
            edges_r = self.process_channel(r, "scharr", self.config["scharr_channel_weights"][0])
            edges_g = self.process_channel(g, "scharr", self.config["scharr_channel_weights"][1])
            edges_b = self.process_channel(b, "scharr", self.config["scharr_channel_weights"][2])

        combined = cv2.bitwise_or(cv2.bitwise_or(edges_r, edges_g), edges_b)
        kernel = np.ones((3, 3), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output = frame.copy() if self.config["overlay_on_original"] else np.zeros_like(frame)

        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                cv2.drawContours(output, [cnt], -1, (0, 255, 255), self.config["line_thickness"])

        self.processed_img = output
        self.display_images()

    def process_channel(self, channel, method, weight):
        if weight == 0:
            return np.zeros_like(channel)
        if method == "canny":
            blurred = cv2.bilateralFilter(channel, 9, self.config["bilateral_sigma_color"], self.config["bilateral_sigma_space"])
            return cv2.Canny(blurred, 50, 150)
        elif method == "scharr":
            grad_x = cv2.Scharr(channel, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(channel, cv2.CV_64F, 0, 1)
            grad_mag = cv2.magnitude(grad_x, grad_y)
            grad_mag = np.uint8(grad_mag / grad_mag.max() * 255)
            _, edges = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return edges

    def display_images(self):
        if self.original_img is not None:
            orig = self.resize_with_ratio(self.original_img, 500)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)))
            self.original_canvas.configure(image=img, text="")
            self.original_canvas.image = img

        if self.processed_img is not None:
            proc = self.resize_with_ratio(self.processed_img, 500)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)))
            self.processed_canvas.configure(image=img, text="")
            self.processed_canvas.image = img

        if self.original_img is not None:
            h, w = self.original_img.shape[:2]
            if w > h:
                self.original_canvas.pack(side="top", pady=5)
                self.processed_canvas.pack(side="top", pady=5)
            else:
                self.original_canvas.pack(side="left", padx=5)
                self.processed_canvas.pack(side="right", padx=5)

    def resize_with_ratio(self, img, max_size):
        h, w = img.shape[:2]
        scale = max_size / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))

    def export_video(self):
        if not self.input_path:
            return
        cap = cv2.VideoCapture(self.input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.splitext(self.input_path)[0] + '_rotoscoped.mp4'
        out = cv2.VideoWriter(out_path, fourcc, self.config["output_fps"], (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.config["frame_step"] == 0:
                self.original_img = frame
                self.update_preview()
                out.write(self.processed_img)
            frame_idx += 1

        cap.release()
        out.release()
        print(f"Export complete: {out_path}")
        messagebox.showinfo("Export Complete", f"Video exported successfully to:\n{out_path}")


if __name__ == "__main__":
    app = RotoscopeApp()
    app.mainloop()

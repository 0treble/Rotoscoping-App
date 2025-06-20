import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import torch
from transformers import pipeline

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
        label = tk.Label(tw, text=self.text, justify='left', background="#333", foreground="white", relief='solid',
                         borderwidth=1, font=("Arial", 9))
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
        self.depth_pipe = None
        self.depth_initialized = False

        # Config defaults
        self.config = {
            "line_thickness": 2,
            "frame_step": 3,
            "output_fps": 10,
            "edge_detection": "Canny",  # Changed from use_canny to edge_detection
            "use_sharpening": False,
            "use_depth": True,
            "depth_delta": 0.3,
            "output_background": "Original",
            "bilateral_sigma_color": 50,
            "bilateral_sigma_space": 50,
            "channel_weights": [1.0, 1.0, 1.0]
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
        # Top button row
        self.top_button_frame = ctk.CTkFrame(self.controls_frame)
        self.top_button_frame.pack(pady=10, fill="x")

        ctk.CTkButton(self.top_button_frame, text="Load Video", command=self.load_video).pack(side="left", padx=5,
                                                                                              expand=True)
        ctk.CTkButton(self.top_button_frame, text="Export Video", command=self.export_video).pack(side="right", padx=5,
                                                                                                  expand=True)

        # Edge detection dropdown
        self.edge_detection_frame = ctk.CTkFrame(self.controls_frame)
        self.edge_detection_frame.pack(pady=10, fill="x")
        ctk.CTkLabel(self.edge_detection_frame, text="Edge Detection:").pack(side="left", padx=5)
        self.edge_detection_combo = ctk.CTkComboBox(
            self.edge_detection_frame,
            values=["Canny", "Scharr"],
            command=self.change_edge_detection
        )
        self.edge_detection_combo.set("Canny")
        self.edge_detection_combo.pack(side="right", padx=5, expand=True)
        CustomToolTip(self.edge_detection_combo, "Choose edge detection method")

        # Create all sliders with value displays
        self.line_thickness_slider = self.create_slider_with_value("Line Thickness", "line_thickness", 1, 10,
                                                                   "Thickness of contour lines")
        self.output_fps_slider = self.create_slider_with_value("Output FPS", "output_fps", 1, 30,
                                                               "Frames per second in exported video")
        self.frame_step_slider = self.create_slider_with_value("Frame Step", "frame_step", 1, 10,
                                                               "How often frames are processed")
        self.depth_delta_slider = self.create_slider_with_value("Depth Delta", "depth_delta", 0.0, 1.0,
                                                                "Distance from the camera", is_float=True)

        self.bilateral_sigma_color_slider = self.create_slider_with_value("Bilateral Sigma Color",
                                                                          "bilateral_sigma_color", 1, 150,
                                                                          "Color sensitivity of bilateral filter")
        self.bilateral_sigma_space_slider = self.create_slider_with_value("Bilateral Sigma Space",
                                                                          "bilateral_sigma_space", 1, 150,
                                                                          "Spatial blur range of bilateral filter")

        # Channel weight sliders
        self.red_slider = self.create_slider_with_value("Red Weight", "channel_weights", 0, 1,
                                                              "Contribution of red channel", is_float=True, index=0)
        self.green_slider = self.create_slider_with_value("Green Weight", "channel_weights", 0, 1,
                                                                "Contribution of green channel", is_float=True,
                                                                index=1)
        self.blue_slider = self.create_slider_with_value("Blue Weight", "channel_weights", 0, 1,
                                                               "Contribution of blue channel", is_float=True,
                                                               index=2)

        # Switches
        self.sharpen_switch = ctk.CTkSwitch(self.controls_frame, text="Use Sharpening", command=self.toggle_sharpening)
        self.sharpen_switch.pack(pady=10)
        CustomToolTip(self.sharpen_switch, "Apply sharpening before edge detection")

        self.depth_switch = ctk.CTkSwitch(self.controls_frame, text="Use Depth", command=self.toggle_depth)
        self.depth_switch.pack(pady=10)
        CustomToolTip(self.depth_switch, "Use depth estimation for better edge detection")

        # Background selection
        self.bg_frame = ctk.CTkFrame(self.controls_frame)
        self.bg_frame.pack(pady=10)
        ctk.CTkLabel(self.bg_frame, text="Background:").pack(side="left", padx=5)
        self.bg_combobox = ctk.CTkComboBox(self.bg_frame,
                                           values=["Original", "Black", "Depth"],
                                           command=self.change_background)
        self.bg_combobox.set("Original")
        self.bg_combobox.pack(side="left", padx=5)
        CustomToolTip(self.bg_combobox, "Background type for output")

    def create_slider_with_value(self, label_text, config_key, min_val, max_val, tooltip_text, is_float=False,
                                 index=None):
        """Create a slider with a value display label"""
        frame = ctk.CTkFrame(self.controls_frame)
        frame.pack(fill="x", pady=2)

        # Get initial value
        if index is not None:
            initial_value = self.config[config_key][index]
        else:
            initial_value = self.config[config_key]

        # Create combined label with value
        value_text = f"{initial_value:.2f}" if is_float else f"{initial_value}"
        combined_label = f"{label_text}: {value_text}"
        label = ctk.CTkLabel(frame, text=combined_label, width=150)
        label.pack(side="left", padx=5)

        # Slider
        slider = ctk.CTkSlider(frame, from_=min_val, to=max_val)
        slider.set(initial_value)
        slider.pack(side="left", expand=True, fill="x", padx=5)

        # Configure command to update both config and label
        def update_value(value):
            value = float(value)
            if is_float:
                display_value = f"{value:.2f}"
            else:
                value = int(value)
                display_value = f"{value}"

            if index is not None:
                self.config[config_key][index] = value
            else:
                self.config[config_key] = value

            # Update the label text
            label.configure(text=f"{label_text}: {display_value}")
            self.update_preview()

        slider.configure(command=update_value)
        CustomToolTip(slider, tooltip_text)
        return slider

    def update_slider_value(self, value, key, index, label, is_float):
        """Update both the config value and the displayed label"""
        value = float(value)
        if is_float:
            display_value = f"{value:.2f}"
        else:
            value = int(value)
            display_value = f"{value}"

        label.configure(text=display_value)

        if index is not None:
            self.config[key][index] = value
        else:
            self.config[key] = value

        self.update_preview()

    def change_edge_detection(self, choice):
        """Handle edge detection method change"""
        self.config["edge_detection"] = choice
        self.update_preview()

    def toggle_sharpening(self):
        self.config["use_sharpening"] = self.sharpen_switch.get()
        self.update_preview()

    def toggle_depth(self):
        self.config["use_depth"] = self.depth_switch.get()
        if self.config["use_depth"] and not self.depth_initialized:
            self.initialize_depth_pipeline()
        self.update_preview()

    def change_background(self, choice):
        self.config["output_background"] = choice
        self.update_preview()

    def initialize_depth_pipeline(self):
        try:
            self.depth_pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if torch.cuda.is_available() else -1,
                use_fast=True
            )
            self.depth_initialized = True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize depth pipeline: {str(e)}")
            self.depth_switch.deselect()
            self.config["use_depth"] = False

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.input_path = path
            cap = cv2.VideoCapture(self.input_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.original_img = frame
                self.update_preview()

    def get_foreground_mask_and_depth(self, frame, delta=0.5):
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        depth_output = self.depth_pipe(pil_img)["depth"]
        depth_np = np.array(depth_output)

        # Normalize
        norm_depth = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

        # Auto-threshold: select region near minimum depth (closer object)
        min_val = norm_depth.min()
        mask = (norm_depth <= min_val + delta).astype(np.uint8) * 255

        mask = cv2.bitwise_not(mask)

        # Resize to match frame
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        depth_vis = cv2.resize((norm_depth * 255).astype(np.uint8), (frame.shape[1], frame.shape[0]))
        return mask, depth_vis

    def update_preview(self):
        if self.original_img is None:
            return

        frame = self.original_img.copy()
        if self.config["use_sharpening"]:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            frame = cv2.filter2D(frame, -1, kernel)

        if self.config["use_depth"] and self.depth_initialized:
            # Depth-based processing
            fg_mask, depth_vis = self.get_foreground_mask_and_depth(frame, self.config["depth_delta"])

            if self.config["edge_detection"] == "Canny":
                # Process depth map for edges
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                depth_clahe = clahe.apply(depth_vis)
                blurred = cv2.GaussianBlur(depth_clahe, (0, 0), sigmaX=1.0)
                depth_sharp = cv2.addWeighted(depth_clahe, 1.5, blurred, -0.5, 0)
                edges = cv2.Canny(depth_sharp, 50, 150)
                edges = cv2.bitwise_and(edges, fg_mask)
            else:
                # Process each channel with Scharr on the original frame but masked
                b, g, r = cv2.split(frame)
                edges_r = self.process_channel(r, "Scharr", self.config["channel_weights"][0])
                edges_g = self.process_channel(g, "Scharr", self.config["channel_weights"][1])
                edges_b = self.process_channel(b, "Scharr", self.config["channel_weights"][2])
                edges = cv2.bitwise_or(cv2.bitwise_or(edges_r, edges_g), edges_b)
                edges = cv2.bitwise_and(edges, fg_mask)

            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.config["output_background"] == "Depth":
                background = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
                background = cv2.bitwise_and(background, background, mask=fg_mask)
            elif self.config["output_background"] == "Original":
                background = frame.copy()
            else:  # black
                background = np.zeros_like(frame)

            output = background.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    cv2.drawContours(output, [cnt], -1, (0, 255, 255), self.config["line_thickness"])
        else:
            # Original processing without depth
            b, g, r = cv2.split(frame)
            if self.config["edge_detection"] == "Canny":
                edges_r = self.process_channel(r, "Canny", self.config["channel_weights"][0])
                edges_g = self.process_channel(g, "Canny", self.config["channel_weights"][1])
                edges_b = self.process_channel(b, "Canny", self.config["channel_weights"][2])
            else:
                edges_r = self.process_channel(r, "Scharr", self.config["channel_weights"][0])
                edges_g = self.process_channel(g, "Scharr", self.config["channel_weights"][1])
                edges_b = self.process_channel(b, "Scharr", self.config["channel_weights"][2])

            edges = cv2.bitwise_or(cv2.bitwise_or(edges_r, edges_g), edges_b)
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            background = frame.copy() if self.config["output_background"] == "Original" else \
                np.zeros_like(frame) if self.config["output_background"] == "Black" else frame.copy()

            output = background.copy()
            for cnt in contours:
                if cv2.contourArea(cnt) > 50:
                    cv2.drawContours(output, [cnt], -1, (0, 255, 255), self.config["line_thickness"])

            # Display on GUI
        self.show_image_on_canvas(self.original_canvas, self.original_img)
        self.show_image_on_canvas(self.processed_canvas, output)
        self.processed_img = output

    def process_channel(self, channel, method, weight=1.0):
        if weight == 0.0:
            return np.zeros_like(channel)

        if method == "Canny":
            return cv2.Canny(channel, 100, 200)
        elif method == "Scharr":
            grad_x = cv2.Scharr(channel, cv2.CV_64F, 1, 0)
            grad_y = cv2.Scharr(channel, cv2.CV_64F, 0, 1)
            magnitude = cv2.magnitude(grad_x, grad_y)
            norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            return (norm * weight).astype(np.uint8)
        return np.zeros_like(channel)

    def show_image_on_canvas(self, canvas, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_pil.thumbnail((500, 500))  # Resize for GUI
        img_tk = ImageTk.PhotoImage(image_pil)
        canvas.configure(image=img_tk)
        canvas.image = img_tk  # Keep a reference!

    def export_video(self):
        if self.input_path is None:
            messagebox.showerror("Error", "Please load a video first.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4")])
        if not save_path:
            return

        cap = cv2.VideoCapture(self.input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(save_path, fourcc, self.config["output_fps"], (width, height))

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
        messagebox.showinfo("Export Complete", f"Video exported to:\n{save_path}")


if __name__ == "__main__":
    app = RotoscopeApp()
    app.mainloop()

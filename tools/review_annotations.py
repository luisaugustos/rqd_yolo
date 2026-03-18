#!/usr/bin/env python
"""
Interactive annotation review tool.
View auto-generated annotations and easily correct them.
"""

import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os

class AnnotationReviewer:
    """Interactive GUI for reviewing and correcting YOLO annotations."""

    def __init__(self, root, images_dir: str, labels_dir: str):
        self.root = root
        self.root.title("YOLO Annotation Reviewer")
        self.root.geometry("1200x800")

        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        # Get list of images
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))
        self.current_idx = 0

        # Create UI
        self.create_widgets()
        self.load_image()

    def create_widgets(self):
        """Create UI elements."""
        # Top frame
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(top_frame, text="Image:").pack(side=tk.LEFT)
        self.label_info = tk.Label(top_frame, text="", font=("Arial", 10))
        self.label_info.pack(side=tk.LEFT, padx=10)

        tk.Button(top_frame, text="← Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Next →", command=self.next_image).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Jump to...", command=self.jump_to).pack(side=tk.LEFT, padx=5)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        # Bottom frame
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        tk.Button(bottom_frame, text="✓ Approve", command=self.approve, bg="green", fg="white").pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="✎ Edit Labels", command=self.edit_labels).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="✗ Clear All", command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        tk.Button(bottom_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        self.status_label = tk.Label(bottom_frame, text="", font=("Arial", 9), fg="blue")
        self.status_label.pack(side=tk.LEFT, padx=10)

    def load_image(self):
        """Load and display current image with annotations."""
        if not self.image_files:
            messagebox.showinfo("No Images", "No images found!")
            return

        image_file = self.image_files[self.current_idx]
        self.current_image_file = image_file
        self.label_file = self.labels_dir / f"{image_file.stem}.txt"

        # Update info
        self.label_info.config(text=f"{self.current_idx + 1}/{len(self.image_files)} - {image_file.name}")

        # Read image
        img = cv2.imread(str(image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.original_image = img

        # Draw annotations
        self.draw_annotations()

    def draw_annotations(self):
        """Draw bounding boxes on image."""
        img = self.original_image.copy()
        h, w = img.shape[:2]

        # Read annotations
        annotations = []
        if self.label_file.exists():
            with open(self.label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id, x_c, y_c, box_w, box_h = map(float, parts)
                            annotations.append((int(cls_id), x_c, y_c, box_w, box_h))

        # Draw boxes
        for cls_id, x_c, y_c, box_w, box_h in annotations:
            x1 = int((x_c - box_w / 2) * w)
            y1 = int((y_c - box_h / 2) * h)
            x2 = int((x_c + box_w / 2) * w)
            y2 = int((y_c + box_h / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "fracture", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display detections count
        self.status_label.config(text=f"Detections: {len(annotations)}")

        # Resize for display
        display_h, display_w = 700, 1000
        img_resized = cv2.resize(img, (display_w, display_h))

        # Convert to PhotoImage
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.canvas.image = photo
        self.photo = photo

    def prev_image(self):
        """Go to previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_image()

    def next_image(self):
        """Go to next image."""
        if self.current_idx < len(self.image_files) - 1:
            self.current_idx += 1
            self.load_image()

    def jump_to(self):
        """Jump to specific image number."""
        dlg = simpledialog.askinteger("Jump to", f"Enter image number (1-{len(self.image_files)}):")
        if dlg and 1 <= dlg <= len(self.image_files):
            self.current_idx = dlg - 1
            self.load_image()

    def approve(self):
        """Mark annotation as approved and move to next."""
        messagebox.showinfo("Approved", f"Image approved: {self.current_image_file.name}")
        self.next_image()

    def edit_labels(self):
        """Open label file for editing."""
        os.startfile(str(self.label_file))
        messagebox.showinfo("Edit", "Label file opened. Make changes and save, then reload the image.")

    def clear_annotations(self):
        """Clear all annotations for current image."""
        if messagebox.askyesno("Confirm", "Clear all annotations for this image?"):
            self.label_file.unlink(missing_ok=True)
            self.load_image()

    def on_canvas_click(self, event):
        """Handle canvas clicks (for future interactive editing)."""
        pass

def main():
    root = tk.Tk()
    images_dir = "C:/Users/luisb/dev/rqd_yolo/data/annotated/dataset_hp_v2/images"
    labels_dir = "C:/Users/luisb/dev/rqd_yolo/data/annotated/dataset_hp_v2/labels"

    # Create directories if they don't exist
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    app = AnnotationReviewer(root, images_dir, labels_dir)
    root.mainloop()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Tinder-Style Niqab Face Review Interface
======================================

This script creates a Tinder-style interface for reviewing aligned niqab faces.
You can swipe through images and decide which ones to keep or delete.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import shutil
from pathlib import Path
import random

class NiqabFaceReviewer:
    def __init__(self, root, image_dir):
        self.root = root
        self.image_dir = Path(image_dir)
        self.keep_dir = self.image_dir.parent / "kept_faces"
        self.delete_dir = self.image_dir.parent / "deleted_faces"
        
        # Create directories
        self.keep_dir.mkdir(exist_ok=True)
        self.delete_dir.mkdir(exist_ok=True)
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            self.image_files.extend(self.image_dir.glob(f"*{ext}"))
            self.image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        # Shuffle for random order
        random.shuffle(self.image_files)
        
        self.current_index = 0
        self.total_images = len(self.image_files)
        self.kept_count = 0
        self.deleted_count = 0
        
        self.setup_ui()
        self.load_current_image()
    
    def setup_ui(self):
        """Setup the Tinder-style interface"""
        self.root.title("Niqab Face Reviewer - Tinder Style")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        # Progress info
        self.progress_label = tk.Label(
            main_frame, 
            text=f"Image 1 of {self.total_images}",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0'
        )
        self.progress_label.pack(pady=(0, 10))
        
        # Stats frame
        stats_frame = tk.Frame(main_frame, bg='#f0f0f0')
        stats_frame.pack(pady=(0, 10))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Kept: 0 | Deleted: 0",
            font=('Arial', 10),
            bg='#f0f0f0'
        )
        self.stats_label.pack()
        
        # Image frame
        self.image_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        self.image_frame.pack(expand=True, fill='both', pady=(0, 20))
        
        # Image label
        self.image_label = tk.Label(self.image_frame, bg='white')
        self.image_label.pack(expand=True, fill='both')
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#f0f0f0')
        button_frame.pack(pady=(0, 10))
        
        # Delete button (red, left side)
        self.delete_btn = tk.Button(
            button_frame,
            text="❌ DELETE",
            font=('Arial', 16, 'bold'),
            bg='#ff4444',
            fg='white',
            width=12,
            height=2,
            command=self.delete_image
        )
        self.delete_btn.pack(side='left', padx=(0, 20))
        
        # Keep button (green, right side)
        self.keep_btn = tk.Button(
            button_frame,
            text="✅ KEEP",
            font=('Arial', 16, 'bold'),
            bg='#44ff44',
            fg='white',
            width=12,
            height=2,
            command=self.keep_image
        )
        self.keep_btn.pack(side='right', padx=(20, 0))
        
        # Navigation buttons
        nav_frame = tk.Frame(main_frame, bg='#f0f0f0')
        nav_frame.pack(pady=(0, 10))
        
        self.prev_btn = tk.Button(
            nav_frame,
            text="← Previous",
            font=('Arial', 12),
            command=self.previous_image,
            state='disabled'
        )
        self.prev_btn.pack(side='left', padx=(0, 10))
        
        self.next_btn = tk.Button(
            nav_frame,
            text="Next →",
            font=('Arial', 12),
            command=self.next_image
        )
        self.next_btn.pack(side='left')
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="Click KEEP to save the image, DELETE to remove it",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#666666'
        )
        instructions.pack(pady=(10, 0))
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.delete_image())
        self.root.bind('<Right>', lambda e: self.keep_image())
        self.root.bind('<space>', lambda e: self.next_image())
        self.root.focus_set()
    
    def load_current_image(self):
        """Load and display the current image"""
        if self.current_index >= len(self.image_files):
            self.show_completion()
            return
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Calculate size to fit in frame (max 600x400)
            max_width, max_height = 600, 400
            img_width, img_height = img.size
            
            # Calculate scaling factor
            scale = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.photo)
            
            # Update progress
            self.progress_label.configure(text=f"Image {self.current_index + 1} of {self.total_images}")
            
            # Update navigation buttons
            self.prev_btn.configure(state='normal' if self.current_index > 0 else 'disabled')
            self.next_btn.configure(state='normal' if self.current_index < self.total_images - 1 else 'disabled')
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
            self.next_image()
    
    def keep_image(self):
        """Keep the current image"""
        if self.current_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Copy to kept directory
            shutil.copy2(image_path, self.keep_dir / image_path.name)
            self.kept_count += 1
            self.update_stats()
            self.next_image()
        except Exception as e:
            messagebox.showerror("Error", f"Could not keep image: {e}")
    
    def delete_image(self):
        """Delete the current image"""
        if self.current_index >= len(self.image_files):
            return
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Move to deleted directory
            shutil.move(str(image_path), str(self.delete_dir / image_path.name))
            self.deleted_count += 1
            self.update_stats()
            self.next_image()
        except Exception as e:
            messagebox.showerror("Error", f"Could not delete image: {e}")
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            self.load_current_image()
    
    def previous_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()
    
    def update_stats(self):
        """Update statistics display"""
        self.stats_label.configure(text=f"Kept: {self.kept_count} | Deleted: {self.deleted_count}")
    
    def show_completion(self):
        """Show completion message"""
        messagebox.showinfo(
            "Review Complete!",
            f"Review completed!\n\n"
            f"Kept: {self.kept_count} images\n"
            f"Deleted: {self.deleted_count} images\n\n"
            f"Kept images saved to: {self.keep_dir}\n"
            f"Deleted images moved to: {self.delete_dir}"
        )
        self.root.quit()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Tinder-style niqab face reviewer')
    parser.add_argument('image_dir', help='Directory containing aligned niqab images')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist!")
        return
    
    # Create and run the interface
    root = tk.Tk()
    app = NiqabFaceReviewer(root, args.image_dir)
    root.mainloop()

if __name__ == "__main__":
    main()

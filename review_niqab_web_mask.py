#!/usr/bin/env python3
"""
Web-based Niqab Face Segmentation Tool
=====================================

This script creates a web-based tool for freehand segmentation of niqab faces.
You can draw where the eyes are and create segmentation masks.
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image, ImageDraw
import io
import base64
import cv2

app = Flask(__name__)

class NiqabFaceSegmenter:
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.masks_dir = Path("/home/maass/code/niqab/train/masks")
        self.masks_dir.mkdir(exist_ok=True)
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            self.image_files.extend(self.image_dir.glob(f"*{ext}"))
            self.image_files.extend(self.image_dir.glob(f"*{ext.upper()}"))
        
        # Shuffle for random order
        random.shuffle(self.image_files)
        
        self.current_index = 0
        self.total_images = len(self.image_files)
        self.segmented_count = 0
        self.skipped_count = 0
        
        print(f"Found {self.total_images} images to segment")
    
    def get_current_image(self):
        """Get the current image data"""
        if self.current_index >= len(self.image_files):
            return None, None, None
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Load and resize image
            img = Image.open(image_path)
            original_size = img.size
            
            # Calculate size to fit in browser (max 800x600)
            max_width, max_height = 800, 600
            img_width, img_height = img.size
            
            # Calculate scaling factor
            scale = min(max_width / img_width, max_height / img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convert to base64 for web display
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str, image_path.name, self.current_index + 1, original_size, (new_width, new_height)
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None, None, None, None
    
    def save_segmentation_mask(self, mask_data, original_size, display_size):
        """Save segmentation mask"""
        if self.current_index >= len(self.image_files):
            return False
        
        image_path = self.image_files[self.current_index]
        mask_filename = image_path.stem + "_mask.png"
        mask_path = self.masks_dir / mask_filename
        
        try:
            # Convert mask data to numpy array
            mask_array = np.array(mask_data, dtype=np.uint8)
            
            # Resize mask back to original image size
            mask_resized = cv2.resize(mask_array, original_size, interpolation=cv2.INTER_NEAREST)
            
            # Convert to proper format: 0 = black (background), 255 = white (face area)
            mask_final = mask_resized * 255
            
            # Save as PNG
            mask_image = Image.fromarray(mask_final, mode='L')
            mask_image.save(mask_path)
            
            self.segmented_count += 1
            return True
        except Exception as e:
            print(f"Error saving mask: {e}")
            return False
    
    def skip_image(self):
        """Skip current image"""
        self.skipped_count += 1
        return True
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < self.total_images - 1:
            self.current_index += 1
            return True
        return False
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'current': self.current_index + 1,
            'total': self.total_images,
            'segmented': self.segmented_count,
            'skipped': self.skipped_count
        }

# Global segmenter instance
segmenter = None

@app.route('/')
def index():
    """Main page"""
    return render_template('niqab_segmenter.html')

@app.route('/api/current')
def get_current():
    """Get current image"""
    global segmenter
    if segmenter is None:
        return jsonify({'error': 'Segmenter not initialized'}), 500
    
    img_data, filename, image_num, original_size, display_size = segmenter.get_current_image()
    
    if img_data is None:
        return jsonify({
            'done': True,
            'stats': segmenter.get_stats()
        })
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_data}',
        'filename': filename,
        'image_num': image_num,
        'original_size': original_size,
        'display_size': display_size,
        'stats': segmenter.get_stats()
    })

@app.route('/api/save_mask', methods=['POST'])
def save_mask():
    """Save segmentation mask"""
    global segmenter
    if segmenter is None:
        return jsonify({'error': 'Segmenter not initialized'}), 500
    
    data = request.json
    mask_data = data.get('mask_data')
    original_size = data.get('original_size')
    display_size = data.get('display_size')
    
    if not mask_data or not original_size or not display_size:
        return jsonify({'error': 'Missing mask data'}), 400
    
    success = segmenter.save_segmentation_mask(mask_data, original_size, display_size)
    if success:
        # Move to next image
        if not segmenter.next_image():
            return jsonify({
                'done': True,
                'stats': segmenter.get_stats()
            })
    
    return jsonify({
        'success': success,
        'stats': segmenter.get_stats()
    })

@app.route('/api/skip', methods=['POST'])
def skip_image():
    """Skip current image"""
    global segmenter
    if segmenter is None:
        return jsonify({'error': 'Segmenter not initialized'}), 500
    
    success = segmenter.skip_image()
    if success:
        # Move to next image
        if not segmenter.next_image():
            return jsonify({
                'done': True,
                'stats': segmenter.get_stats()
            })
    
    return jsonify({
        'success': success,
        'stats': segmenter.get_stats()
    })

@app.route('/api/delete', methods=['POST'])
def delete_image():
    """Delete current image"""
    global segmenter
    if segmenter is None:
        return jsonify({'error': 'Segmenter not initialized'}), 500
    
    if segmenter.current_index >= len(segmenter.image_files):
        return jsonify({'error': 'No image to delete'}), 400
    
    try:
        # Delete the current image file
        image_path = segmenter.image_files[segmenter.current_index]
        os.remove(image_path)
        
        # Remove from the list
        segmenter.image_files.pop(segmenter.current_index)
        segmenter.total_images = len(segmenter.image_files)
        
        # Adjust current index if needed
        if segmenter.current_index >= segmenter.total_images:
            segmenter.current_index = max(0, segmenter.total_images - 1)
        
        return jsonify({
            'success': True,
            'stats': segmenter.get_stats()
        })
    except Exception as e:
        return jsonify({'error': f'Failed to delete image: {e}'}), 500

@app.route('/api/next', methods=['POST'])
def next_image():
    """Go to next image"""
    global segmenter
    if segmenter is None:
        return jsonify({'error': 'Segmenter not initialized'}), 500
    
    if segmenter.next_image():
        return jsonify({'success': True, 'stats': segmenter.get_stats()})
    else:
        return jsonify({
            'done': True,
            'stats': segmenter.get_stats()
        })

def create_html_template():
    """Create the HTML template"""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Niqab Face Segmenter</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1000px;
            width: 100%;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .progress {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        
        .stats {
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }
        
        .canvas-container {
            text-align: center;
            margin: 20px 0;
            background: #f8f8f8;
            border-radius: 8px;
            padding: 20px;
            position: relative;
            display: inline-block;
        }
        
        #canvas {
            border: 2px solid #ddd;
            border-radius: 8px;
            cursor: crosshair;
            max-width: 100%;
            height: auto;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 10px 20px;
            font-size: 14px;
            font-weight: bold;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .btn-primary {
            background-color: #007bff;
            color: white;
        }
        
        .btn-primary:hover {
            background-color: #0056b3;
        }
        
        .btn-success {
            background-color: #28a745;
            color: white;
        }
        
        .btn-success:hover {
            background-color: #1e7e34;
        }
        
        .btn-warning {
            background-color: #ffc107;
            color: black;
        }
        
        .btn-warning:hover {
            background-color: #e0a800;
        }
        
        .btn-danger {
            background-color: #dc3545;
            color: white;
        }
        
        .btn-danger:hover {
            background-color: #c82333;
        }
        
        .instructions {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .loading {
            text-align: center;
            color: #666;
            font-size: 16px;
        }
        
        .done {
            text-align: center;
            color: #22aa22;
            font-size: 18px;
            font-weight: bold;
        }
        
        .brush-size {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 10px 0;
        }
        
        .brush-size input {
            width: 100px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Niqab Face Segmenter</h1>
            <div class="progress" id="progress">Loading...</div>
            <div class="stats" id="stats">Segmented: 0 | Skipped: 0</div>
        </div>
        
        <div class="canvas-container" id="canvasContainer">
            <div class="loading">Loading image...</div>
        </div>
        
        <div class="controls">
            <div class="brush-size">
                <label for="brushSize">Brush Size:</label>
                <input type="range" id="brushSize" min="1" max="50" value="10">
                <span id="brushSizeValue">10</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="clearCanvas()">Clear</button>
            <button class="btn btn-success" onclick="fillArea()">Fill Area</button>
            <button class="btn btn-success" onclick="saveMask()">Save Mask</button>
            <button class="btn btn-warning" onclick="skipImage()">Skip</button>
            <button class="btn btn-danger" onclick="deleteImage()">Delete Image</button>
            <button class="btn btn-primary" onclick="nextImage()">Next</button>
        </div>
        
        <div class="instructions">
            <strong>Instructions:</strong><br>
            1. Draw white dots/lines on the visible face area (eyes, forehead)<br>
            2. Click "Fill Area" to fill the entire area you marked<br>
            3. Use the brush size slider to adjust drawing precision<br>
            4. Click "Clear" to start over<br>
            5. Click "Save Mask" to save the segmentation and move to next image<br>
            6. Click "Skip" to skip this image without saving<br>
            7. Click "Delete Image" to permanently remove this image<br>
            8. Click "Next" to move to next image without saving current work
        </div>
    </div>

    <script>
        let currentImage = null;
        let canvas = null;
        let ctx = null;
        let isDrawing = false;
        let brushSize = 10;
        
        function loadCurrentImage() {
            fetch('/api/current')
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        document.getElementById('canvasContainer').innerHTML = 
                            '<div class="done">Segmentation Complete!<br><br>' +
                            'Segmented: ' + data.stats.segmented + ' images<br>' +
                            'Skipped: ' + data.stats.skipped + ' images</div>';
                        return;
                    }
                    
                    if (data.error) {
                        document.getElementById('canvasContainer').innerHTML = 
                            '<div class="loading">Error: ' + data.error + '</div>';
                        return;
                    }
                    
                    currentImage = data;
                    document.getElementById('progress').textContent = 
                        'Image ' + data.image_num + ' of ' + data.stats.total;
                    document.getElementById('stats').textContent = 
                        'Segmented: ' + data.stats.segmented + ' | Skipped: ' + data.stats.skipped;
                    
                    // Create canvas
                    createCanvas(data.display_size[0], data.display_size[1]);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('canvasContainer').innerHTML = 
                        '<div class="loading">Error loading image</div>';
                });
        }
        
        function createCanvas(width, height) {
            const container = document.getElementById('canvasContainer');
            container.innerHTML = `
                <canvas id="canvas" width="${width}" height="${height}"></canvas>
                <img id="backgroundImage" style="display: none;" />
            `;
            
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            
            // Load background image
            const img = document.getElementById('backgroundImage');
            img.onload = function() {
                ctx.drawImage(img, 0, 0, width, height);
            };
            img.src = currentImage.image;
            
            // Set up drawing events
            setupDrawingEvents();
        }
        
        function setupDrawingEvents() {
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            canvas.addEventListener('mouseout', stopDrawing);
            
            // Touch events for mobile
            canvas.addEventListener('touchstart', handleTouch);
            canvas.addEventListener('touchmove', handleTouch);
            canvas.addEventListener('touchend', stopDrawing);
        }
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = 'rgba(255, 255, 255, 1.0)'; // White for face area
            ctx.beginPath();
            ctx.arc(x, y, brushSize, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                            e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }
        
        function clearCanvas() {
            if (!canvas || !currentImage) return;
            
            // Redraw background image
            const img = document.getElementById('backgroundImage');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
        
        function fillArea() {
            if (!canvas || !currentImage) return;
            
            // Get current canvas data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Create a mask of white pixels
            const mask = Array(canvas.height).fill().map(() => Array(canvas.width).fill(0));
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    const i = (y * canvas.width + x) * 4;
                    const r = data[i];
                    const g = data[i + 1];
                    const b = data[i + 2];
                    const a = data[i + 3];
                    
                    if (r > 200 && g > 200 && b > 200 && a > 50) {
                        mask[y][x] = 1;
                    }
                }
            }
            
            // Count white pixels
            let whiteCount = 0;
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    if (mask[y][x] === 1) whiteCount++;
                }
            }
            
            if (whiteCount === 0) {
                alert('No white pixels found. Draw some white areas first.');
                return;
            }
            
            // Use scanline fill algorithm
            const filled = Array(canvas.height).fill().map(() => Array(canvas.width).fill(0));
            
            for (let y = 0; y < canvas.height; y++) {
                let inside = false;
                let startX = -1;
                
                for (let x = 0; x < canvas.width; x++) {
                    if (mask[y][x] === 1) {
                        if (!inside) {
                            inside = true;
                            startX = x;
                        } else {
                            // Fill from startX to x-1
                            for (let fillX = startX; fillX < x; fillX++) {
                                filled[y][fillX] = 1;
                            }
                            inside = false;
                        }
                    }
                }
            }
            
            // Apply the fill to canvas
            ctx.fillStyle = 'rgba(255, 255, 255, 1.0)';
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    if (filled[y][x] === 1) {
                        ctx.fillRect(x, y, 1, 1);
                    }
                }
            }
            
            console.log(`Filled ${whiteCount} boundary pixels`);
        }
        
        function floodFill(mask, width, height) {
            // Simple approach: just return the mask as-is
            // The user draws the boundary, we keep it as the mask
            return mask.map(row => [...row]);
        }
        
        function saveMask() {
            if (!canvas || !currentImage) return;
            
            // Get canvas data
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Create mask array (0 for background, 1 for drawn areas)
            const maskData = [];
            for (let i = 0; i < data.length; i += 4) {
                // Check if pixel has white color (drawn area)
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const a = data[i + 3];
                
                // If pixel has white color and alpha > 0, mark as 1, otherwise 0
                if (r > 200 && g > 200 && b > 200 && a > 50) {
                    maskData.push(1);
                } else {
                    maskData.push(0);
                }
            }
            
            // Reshape to 2D array
            const mask2D = [];
            for (let i = 0; i < canvas.height; i++) {
                mask2D.push(maskData.slice(i * canvas.width, (i + 1) * canvas.width));
            }
            
            // Count drawn pixels for debugging
            let drawnPixels = 0;
            for (let y = 0; y < canvas.height; y++) {
                for (let x = 0; x < canvas.width; x++) {
                    if (mask2D[y][x] === 1) {
                        drawnPixels++;
                    }
                }
            }
            
            console.log(`Detected ${drawnPixels} drawn pixels`);
            
            // Fill the boundary using flood fill
            const filledMask = floodFill(mask2D, canvas.width, canvas.height);
            
            // Send to server
            fetch('/api/save_mask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    mask_data: filledMask,
                    original_size: currentImage.original_size,
                    display_size: currentImage.display_size
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.done) {
                    loadCurrentImage();
                } else if (data.success) {
                    loadCurrentImage();
                } else {
                    alert('Error saving mask');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error saving mask');
            });
        }
        
        function skipImage() {
            if (!currentImage) return;
            
            fetch('/api/skip', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        loadCurrentImage();
                    } else if (data.success) {
                        loadCurrentImage();
                    } else {
                        alert('Error skipping image');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error skipping image');
                });
        }
        
        function deleteImage() {
            if (!currentImage) return;
            
            if (confirm('Are you sure you want to delete this image permanently?')) {
                fetch('/api/delete', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        if (data.done) {
                            loadCurrentImage();
                        } else if (data.success) {
                            loadCurrentImage();
                        } else {
                            alert('Error deleting image');
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('Error deleting image');
                    });
            }
        }
        
        function nextImage() {
            if (!currentImage) return;
            
            fetch('/api/next', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        loadCurrentImage();
                    } else if (data.success) {
                        loadCurrentImage();
                    } else {
                        alert('Error moving to next image');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error moving to next image');
                });
        }
        
        // Brush size control
        document.getElementById('brushSize').addEventListener('input', function(e) {
            brushSize = parseInt(e.target.value);
            document.getElementById('brushSizeValue').textContent = brushSize;
        });
        
        // Load first image
        loadCurrentImage();
    </script>
</body>
</html>
    """
    
    with open(template_dir / "niqab_segmenter.html", "w") as f:
        f.write(html_content)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based niqab face segmenter')
    parser.add_argument('image_dir', help='Directory containing aligned niqab images')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001, help='Port to bind to (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist!")
        return
    
    # Create HTML template
    create_html_template()
    
    # Initialize segmenter
    global segmenter
    segmenter = NiqabFaceSegmenter(args.image_dir)
    
    print(f"Starting segmentation server on {args.host}:{args.port}")
    print(f"Open your browser and go to: http://{args.host}:{args.port}")
    print(f"Masks will be saved to: {segmenter.masks_dir}")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Web-based Tinder-Style Niqab Face Review Interface
=================================================

This script creates a web-based Tinder-style interface for reviewing aligned niqab faces.
You can access it through your browser on your HPC.
"""

import os
import sys
import json
import random
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import io
import base64

app = Flask(__name__)

class NiqabFaceReviewer:
    def __init__(self, image_dir):
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
        
        print(f"Found {self.total_images} images to review")
    
    def get_current_image(self):
        """Get the current image data"""
        if self.current_index >= len(self.image_files):
            return None, None, None
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Calculate size to fit in browser (max 600x400)
            max_width, max_height = 600, 400
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
            
            return img_str, image_path.name, self.current_index + 1
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return None, None, None
    
    def keep_image(self):
        """Keep the current image"""
        if self.current_index >= len(self.image_files):
            return False
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Copy to kept directory
            import shutil
            shutil.copy2(image_path, self.keep_dir / image_path.name)
            self.kept_count += 1
            return True
        except Exception as e:
            print(f"Error keeping image: {e}")
            return False
    
    def delete_image(self):
        """Delete the current image"""
        if self.current_index >= len(self.image_files):
            return False
        
        image_path = self.image_files[self.current_index]
        
        try:
            # Move to deleted directory
            import shutil
            shutil.move(str(image_path), str(self.delete_dir / image_path.name))
            self.deleted_count += 1
            return True
        except Exception as e:
            print(f"Error deleting image: {e}")
            return False
    
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
            'kept': self.kept_count,
            'deleted': self.deleted_count
        }

# Global reviewer instance
reviewer = None

@app.route('/')
def index():
    """Main page"""
    return render_template('niqab_reviewer.html')

@app.route('/api/current')
def get_current():
    """Get current image"""
    global reviewer
    if reviewer is None:
        return jsonify({'error': 'Reviewer not initialized'}), 500
    
    img_data, filename, image_num = reviewer.get_current_image()
    
    if img_data is None:
        return jsonify({
            'done': True,
            'stats': reviewer.get_stats()
        })
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_data}',
        'filename': filename,
        'image_num': image_num,
        'stats': reviewer.get_stats()
    })

@app.route('/api/keep', methods=['POST'])
def keep_image():
    """Keep current image"""
    global reviewer
    if reviewer is None:
        return jsonify({'error': 'Reviewer not initialized'}), 500
    
    success = reviewer.keep_image()
    if success:
        # Move to next image
        if not reviewer.next_image():
            return jsonify({
                'done': True,
                'stats': reviewer.get_stats()
            })
    
    return jsonify({
        'success': success,
        'stats': reviewer.get_stats()
    })

@app.route('/api/delete', methods=['POST'])
def delete_image():
    """Delete current image"""
    global reviewer
    if reviewer is None:
        return jsonify({'error': 'Reviewer not initialized'}), 500
    
    success = reviewer.delete_image()
    if success:
        # Move to next image
        if not reviewer.next_image():
            return jsonify({
                'done': True,
                'stats': reviewer.get_stats()
            })
    
    return jsonify({
        'success': success,
        'stats': reviewer.get_stats()
    })

@app.route('/api/next', methods=['POST'])
def next_image():
    """Go to next image"""
    global reviewer
    if reviewer is None:
        return jsonify({'error': 'Reviewer not initialized'}), 500
    
    if reviewer.next_image():
        return jsonify({'success': True, 'stats': reviewer.get_stats()})
    else:
        return jsonify({
            'done': True,
            'stats': reviewer.get_stats()
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
    <title>Niqab Face Reviewer - Tinder Style</title>
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
            max-width: 800px;
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
        
        .image-container {
            text-align: center;
            margin: 20px 0;
            background: #f8f8f8;
            border-radius: 8px;
            padding: 20px;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .buttons {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
        }
        
        .btn {
            padding: 15px 30px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            min-width: 120px;
        }
        
        .btn-delete {
            background-color: #ff4444;
            color: white;
        }
        
        .btn-delete:hover {
            background-color: #ff2222;
            transform: scale(1.05);
        }
        
        .btn-keep {
            background-color: #44ff44;
            color: white;
        }
        
        .btn-keep:hover {
            background-color: #22ff22;
            transform: scale(1.05);
        }
        
        .btn-next {
            background-color: #4444ff;
            color: white;
        }
        
        .btn-next:hover {
            background-color: #2222ff;
            transform: scale(1.05);
        }
        
        .instructions {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 20px;
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
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Niqab Face Reviewer - Tinder Style</h1>
            <div class="progress" id="progress">Loading...</div>
            <div class="stats" id="stats">Kept: 0 | Deleted: 0</div>
        </div>
        
        <div class="image-container" id="imageContainer">
            <div class="loading">Loading image...</div>
        </div>
        
        <div class="buttons" id="buttons">
            <button class="btn btn-delete" onclick="deleteImage()">❌ DELETE</button>
            <button class="btn btn-next" onclick="nextImage()">⏭️ NEXT</button>
            <button class="btn btn-keep" onclick="keepImage()">✅ KEEP</button>
        </div>
        
        <div class="instructions">
            Click KEEP to save the image, DELETE to remove it, or NEXT to skip
        </div>
    </div>

    <script>
        let currentImage = null;
        
        function loadCurrentImage() {
            fetch('/api/current')
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        document.getElementById('imageContainer').innerHTML = 
                            '<div class="done">Review Complete!<br><br>' +
                            'Kept: ' + data.stats.kept + ' images<br>' +
                            'Deleted: ' + data.stats.deleted + ' images</div>';
                        document.getElementById('buttons').style.display = 'none';
                        return;
                    }
                    
                    if (data.error) {
                        document.getElementById('imageContainer').innerHTML = 
                            '<div class="loading">Error: ' + data.error + '</div>';
                        return;
                    }
                    
                    currentImage = data;
                    document.getElementById('progress').textContent = 
                        'Image ' + data.image_num + ' of ' + data.stats.total;
                    document.getElementById('stats').textContent = 
                        'Kept: ' + data.stats.kept + ' | Deleted: ' + data.stats.deleted;
                    document.getElementById('imageContainer').innerHTML = 
                        '<img src="' + data.image + '" alt="' + data.filename + '">';
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('imageContainer').innerHTML = 
                        '<div class="loading">Error loading image</div>';
                });
        }
        
        function keepImage() {
            if (!currentImage) return;
            
            fetch('/api/keep', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.done) {
                        loadCurrentImage();
                    } else if (data.success) {
                        loadCurrentImage();
                    } else {
                        alert('Error keeping image');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('Error keeping image');
                });
        }
        
        function deleteImage() {
            if (!currentImage) return;
            
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
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {
            if (event.key === 'ArrowLeft') {
                deleteImage();
            } else if (event.key === 'ArrowRight') {
                keepImage();
            } else if (event.key === ' ') {
                event.preventDefault();
                nextImage();
            }
        });
        
        // Load first image
        loadCurrentImage();
    </script>
</body>
</html>
    """
    
    with open(template_dir / "niqab_reviewer.html", "w") as f:
        f.write(html_content)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Web-based Tinder-style niqab face reviewer')
    parser.add_argument('image_dir', help='Directory containing aligned niqab images')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory '{args.image_dir}' does not exist!")
        return
    
    # Create HTML template
    create_html_template()
    
    # Initialize reviewer
    global reviewer
    reviewer = NiqabFaceReviewer(args.image_dir)
    
    print(f"Starting web server on {args.host}:{args.port}")
    print(f"Open your browser and go to: http://{args.host}:{args.port}")
    print("Use arrow keys: Left=Delete, Right=Keep, Space=Next")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()

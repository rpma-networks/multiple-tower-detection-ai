from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import time
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Configuration (env-overridable)
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
ALLOWED_EXTENSIONS = {
    ext.strip().lower() for ext in os.environ.get(
        'ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,bmp,tiff'
    ).split(',') if ext.strip()
}
try:
    MAX_FILE_SIZE = int(os.environ.get('MAX_FILE_SIZE', str(10 * 1024 * 1024)))
except ValueError:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # fallback 10MB

# Optional API key for simple auth (no-op if not set)
API_KEY = os.environ.get('API_KEY')

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv11 model path
MODEL_PATH = os.environ.get('MODEL_PATH', 'best.pt')
model = None

# Classes that must be excluded from recognition/output
EXCLUDED_LABEL_KEY = 'antenna-damaged'

def _normalize_label_key(name):
    """Normalize label to a canonical lowercase key (spaces collapsed, hyphens unified)."""
    try:
        s = ' '.join(str(name).strip().lower().split())
    except Exception:
        return ''
    # unify various dash variants and remove surrounding spaces
    s = s.replace(' – ', '-').replace(' — ', '-')
    s = s.replace(' - ', '-').replace(' –', '-').replace('– ', '-')
    s = s.replace(' —', '-').replace('— ', '-')
    s = s.replace('–', '-').replace('—', '-')
    return s

def is_excluded_class(name):
    """Return True if the provided class name should be ignored entirely."""
    key = _normalize_label_key(name)
    return key == EXCLUDED_LABEL_KEY

def load_model():
    """Load the YOLOv11 model from best.pt"""
    global model
    try:
        print(f"Attempting to load model from: {MODEL_PATH}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Model file exists: {os.path.exists(MODEL_PATH)}")
        
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
            print(f"Model type: {type(model)}")
            
            # Get model information
            if hasattr(model, 'names'):
                print(f"Model classes ({len(model.names)}): {model.names}")
                print("Available classes:")
                for class_id, class_name in model.names.items():
                    print(f"  {class_id}: {class_name}")
            else:
                print("Warning: Model names not available")
                
            # Test model with a dummy image to ensure it's working
            import numpy as np
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            test_results = model(dummy_image, verbose=False)
            print("Model test successful - ready for inference")
            
        else:
            print(f"ERROR: Model file not found at {MODEL_PATH}")
            print("Please ensure model file exists or set MODEL_PATH env var")
            model = None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("Please check that the .pt file is a valid YOLO model file")
        model = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_image_mimetype(file_obj):
    """Best-effort server-side MIME check for images."""
    mimetype = getattr(file_obj, 'mimetype', '') or ''
    return mimetype.startswith('image/')

def normalize_class_name(name):
    """Normalize class names for consistent presentation without changing model logic."""
    if not isinstance(name, str):
        return name
    
    # First, strip leading/trailing whitespace and collapse multiple spaces
    cleaned = ' '.join(name.strip().split())
    
    # Handle underscores (replace with spaces)
    cleaned = cleaned.replace('_', ' ')
    cleaned = ' '.join(cleaned.split())  # Collapse spaces again
    
    # Handle special cases with dashes and spaces
    # Normalize "antenna - Damaged" to "Antenna - Damaged" etc.
    if ' - ' in cleaned:
        parts = cleaned.split(' - ')
        cleaned = ' - '.join(part.strip() for part in parts)
    
    # Handle hyphenated words (like "surface-damage")
    if '-' in cleaned and ' - ' not in cleaned:
        # Replace single hyphens with spaces for better formatting
        cleaned = cleaned.replace('-', ' ')
        cleaned = ' '.join(cleaned.split())
    
    # Convert to lowercase first for consistent processing
    cleaned_lower = cleaned.lower()
    
    # Title-case but preserve acronyms and special formatting
    normalized = cleaned.title()
    
    # Preserve known acronyms (uppercase) - check in original cleaned string
    acronyms = ['GSM', 'BTS', 'LT', 'RRU']
    for acronym in acronyms:
        # Replace case-insensitive - simple approach
        normalized_lower = normalized.lower()
        acronym_lower = acronym.lower()
        if acronym_lower in normalized_lower:
            # Find and replace preserving word boundaries
            words = normalized.split()
            for i, word in enumerate(words):
                if word.lower() == acronym_lower:
                    words[i] = acronym
            normalized = ' '.join(words)
    
    # Specific class name normalizations based on model classes
    # Use lowercase keys for case-insensitive matching
    replacements_lower = {
        # Tower classes
        'mobile tower': 'Mobile Tower',
        'small tower': 'Small Tower',
        'tower base': 'Tower Base',
        'tower': 'Tower',
        'tower lattice': 'Tower Lattice',
        'tower tucohy': 'Tower Tucohy',
        'tower wooden': 'Tower Wooden',
        
        # Antenna classes
        'gsm antenna': 'GSM Antenna',
        'microwave antenna': 'Microwave Antenna',
        'panel antenna': 'Panel Antenna',
        'dirty antenna': 'Dirty Antenna',
        'antenna': 'Antenna',
        'antenna - damaged': 'Antenna - Damaged',
        'antenna - not damaged': 'Antenna - Not Damaged',
        
        # Equipment classes
        'bts': 'BTS',
        'control box': 'Control Box',
        'generator': 'Generator',
        'solar panel': 'Solar Panel',
        'microwave dish': 'Microwave Dish',
        'remote radio unit': 'Remote Radio Unit',
        
        # Damage/condition classes
        'discoloration': 'Discoloration',
        'surface damage': 'Surface Damage',
        'corrosion': 'Corrosion',
        'dirty equipment': 'Dirty Equipment',
        'rusty mounts and bolts': 'Rusty Mounts and Bolts',
        'rusty bolts': 'Rusty Bolts',
        'rusty rod and bolts': 'Rusty Rod and Bolts',
        'break': 'Break',
        'thunderbolt': 'Thunderbolt',
        'wear': 'Wear',
        'loose': 'Loose',
        'twist': 'Twist',
        'uneven': 'Uneven',
        
        # Cable classes
        'shielded information cable': 'Shielded Information Cable',
        'information cable': 'Information Cable',
        'cable': 'Cable',
        'wire': 'Wire',
        
        # Other classes
        'nest': 'Nest',
        'joint': 'Joint',
        'side': 'Side',
        'head': 'Head',
        'anchor': 'Anchor',
        'gasket': 'Gasket',
        'void': 'Void',
    }
    
    # Check case-insensitive match
    if cleaned_lower in replacements_lower:
        return replacements_lower[cleaned_lower]
    
    # Default: return properly formatted version with acronyms preserved
    return normalized

def maybe_require_api_key():
    """Enforce API key if configured via env. No-op if API_KEY not set."""
    if not API_KEY:
        return None  # allow
    provided = request.headers.get('X-API-Key') or request.headers.get('Authorization')
    if provided and provided.startswith('Bearer '):
        provided = provided.split(' ', 1)[1]
    if provided != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401
    return None

def preprocess_image(image_path):
    """Preprocess image for YOLO inference"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Could not read image")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

def run_inference(image):
    """Run YOLO inference with fixed optimal parameters for tower detection"""
    try:
        if model is None:
            raise ValueError("Model not loaded. Please ensure best.pt is available.")
        
        # Run inference with fixed optimal YOLO parameters
        results = model(image, 
                       conf=0.20,    # Fixed confidence threshold: 0.20
                       iou=0.50,     # Fixed IoU threshold: 0.50
                       max_det=300)  # Fixed max detections: 300
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Convert to xywh format
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Additional confidence filtering for antenna classes (dynamic based on model)
                    if hasattr(model, 'names') and model.names:
                        class_name_lower = model.names[class_id].lower()
                        if 'antenna' in class_name_lower and confidence < 0.18:
                            continue
                    
                    # Validate class_id and get class name
                    if hasattr(model, 'names') and class_id in model.names:
                        class_name = model.names[class_id]
                    elif hasattr(model, 'names'):
                        class_name = f'unknown_class_{class_id}'
                    else:
                        class_name = 'object'
                    
                    # Exclude specific class completely
                    if is_excluded_class(class_name):
                        continue

                    detections.append({
                        'bbox': [float(x), float(y), float(w), float(h)],
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        # Post-process detections to filter false positives
        detections = filter_false_positives(detections)
        
        return detections
    except Exception as e:
        raise ValueError(f"Error during inference: {e}")

def filter_false_positives(detections):
    """Filter out likely false positive detections with optimized thresholds"""
    if not detections:
        return detections
    
    filtered_detections = []
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        bbox = detection['bbox']
        
        # Exclude specific class completely
        if is_excluded_class(class_name):
            continue

        # Filter out very small detections (likely false positives)
        if bbox[2] < 15 or bbox[3] < 15:  # Width or height less than 15 pixels
            continue
            
        # Filter out very low confidence antenna detections (adjusted for new range)
        if 'antenna' in class_name.lower() and confidence < 0.18:
            continue
            
        # Filter out detections that are too close to image edges (often false positives)
        if bbox[0] < 5 or bbox[1] < 5:  # Too close to top/left edge
            continue
            
        # Additional filtering for very low confidence detections
        if confidence < 0.16:  # Absolute minimum confidence
            continue
            
        filtered_detections.append(detection)
    
    return filtered_detections

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/detect', methods=['POST'])
def detect_tower():
    """API endpoint for tower detection - supports single image"""
    try:
        # Optional API key enforcement
        auth_resp = maybe_require_api_key()
        if auth_resp is not None:
            return auth_resp
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename) or not is_image_mimetype(file):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Preprocess image
            image = preprocess_image(file_path)
            
            # Run inference with fixed optimal parameters
            detections = run_inference(image)
            
            # Process results - always return success to display image
            if detections:
                # Return ALL detections, not just the best one
                response = {
                    'success': True,
                    'detections': [
                        {
                            **d,
                            'class_name': normalize_class_name(d.get('class_name'))
                        } for d in detections
                    ],
                    'total_detections': len(detections),
                    'message': f'Found {len(detections)} object(s) in the image'
                }
            else:
                # No detections found, but still return success to display image
                response = {
                    'success': True,
                    'detections': [],
                    'confidence': 0.0,
                    'class_name': 'No tower detected',
                    'bbox': None,
                    'total_detections': 0,
                    'message': 'Image uploaded successfully, but no tower detected'
                }
            
            return jsonify(response)
            
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-multiple', methods=['POST'])
def detect_towers_multiple():
    """API endpoint for multiple tower detection"""
    try:
        # Optional API key enforcement
        auth_resp = maybe_require_api_key()
        if auth_resp is not None:
            return auth_resp
        # Check if image files are present
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No files selected'}), 400
        
        results = []
        processed_files = []
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
                
            # Check file type
            if not allowed_file(file.filename):
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'Invalid file type',
                    'filename': file.filename
                })
                continue
            
            # Check file size
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size > MAX_FILE_SIZE:
                results.append({
                    'index': i,
                    'success': False,
                    'error': 'File too large',
                    'filename': file.filename
                })
                continue
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = str(int(time.time()))
            filename = f"{timestamp}_{i}_{filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            processed_files.append(file_path)
            
            try:
                # Preprocess image
                image = preprocess_image(file_path)
                
                # Run inference
                detections = run_inference(image)
                
                # Process results
                if detections:
                    # Return ALL detections for multiple image processing
                    result = {
                        'index': i,
                        'success': True,
                        'filename': file.filename,
                        'detections': [
                            {
                                **d,
                                'class_name': normalize_class_name(d.get('class_name'))
                            } for d in detections
                        ],
                        'total_detections': len(detections),
                        'message': f'Found {len(detections)} object(s) in {file.filename}'
                    }
                else:
                    result = {
                        'index': i,
                        'success': True,
                        'filename': file.filename,
                        'detections': [],
                        'confidence': 0.0,
                        'class_name': 'No tower detected',
                        'bbox': None,
                        'total_detections': 0,
                        'message': 'Image processed successfully, but no tower detected'
                    }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'filename': file.filename
                })
            finally:
                # Clean up uploaded file
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.get('success', False)]
        total_detections = sum(r.get('total_detections', 0) for r in successful_results)
        
        response = {
            'success': True,
            'results': results,
            'summary': {
                'total_files': len(files),
                'successful_files': len(successful_results),
                'failed_files': len(results) - len(successful_results),
                'total_detections': total_detections
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': time.time()
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available detection classes from the loaded model"""
    if model is not None and hasattr(model, 'names'):
        visible = {k: normalize_class_name(v) for k, v in model.names.items() if not is_excluded_class(v)}
        return jsonify({
            'success': True,
            'classes': visible,
            'total_classes': len(visible),
            'model_loaded': True
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please ensure best.pt is available.',
            'model_loaded': False
        })

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    if model is not None:
        model_info = {
            'success': True,
            'model_loaded': True,
            'model_path': MODEL_PATH,
            'model_type': str(type(model)),
            'total_classes': (len([1 for v in model.names.values() if not is_excluded_class(v)]) if hasattr(model, 'names') else 0),
            'classes': ({k: normalize_class_name(v) for k, v in model.names.items() if not is_excluded_class(v)} if hasattr(model, 'names') else {}),
            'timestamp': time.time()
        }
        
        # Add class categories if available
        if hasattr(model, 'names') and model.names:
            normalized_values = [normalize_class_name(name) for name in model.names.values() if not is_excluded_class(name)]
            antenna_classes = [name for name in normalized_values if 'antenna' in name.lower()]
            tower_classes = [name for name in normalized_values if 'tower' in name.lower()]
            damage_classes = [name for name in normalized_values if any(word in name.lower() for word in ['damage', 'rust', 'corrosion', 'dirty'])]
            
            model_info['class_categories'] = {
                'antenna_classes': antenna_classes,
                'tower_classes': tower_classes,
                'damage_classes': damage_classes
            }
        
        return jsonify(model_info)
    else:
        return jsonify({
            'success': False,
            'error': 'Model not loaded',
            'model_loaded': False
        })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413

if __name__ == '__main__':
    print("Loading YOLOv11 model...")
    load_model()
    print("Starting Flask server...")
    # Use 0.0.0.0 for Docker compatibility
    port = int(os.environ.get('PORT', '5002'))
    app.run(debug=False, host='0.0.0.0', port=port)

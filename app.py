from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import time
import os
from models.tone_converter import ToneConverter

app = Flask(__name__)
CORS(app)

# Initialize the tone converter model
tone_converter = ToneConverter()

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/convert', methods=['POST'])
def convert_text():
    """
    API endpoint to convert text to different tones
    Expected JSON payload:
    {
        "text": "input text",
        "mode": "polite|formal|informal|professional|friendly|neutral"
    }
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        input_text = data.get('text', '').strip()
        mode = data.get('mode', 'polite').lower()
        
        # Validation
        if not input_text:
            return jsonify({'error': 'Text is required'}), 400
        
        if len(input_text) < 3:
            return jsonify({'error': 'Text must be at least 3 characters long'}), 400
        
        if mode not in ['polite', 'formal', 'informal', 'professional', 'friendly', 'neutral']:
            return jsonify({'error': f'Invalid mode: {mode}'}), 400
        
        # Track processing time
        start_time = time.time()
        
        # Convert the text
        result = tone_converter.convert(input_text, mode)
        
        # Calculate processing time
        processing_time = round((time.time() - start_time) * 1000, 2)  # in milliseconds
        
        # Prepare enhanced response with detailed analysis
        analysis = result.get('analysis', {})
        response = {
            'success': True,
            'original_text': input_text,
            'converted_text': result['converted_text'],
            'mode': mode,
            'alternatives': result.get('alternatives', []),
            'dataset_info': {
                'model': result.get('model', 'Advanced Tone Converter v2.0'),
                'datasets': result.get('datasets_used', ['GYAFC', 'ParaNMT', 'Wikipedia Simple', 'Yelp Reviews']),
                'processing_time': f"{analysis.get('processing_time_ms', processing_time)}ms",
                'confidence': result.get('confidence', 'High'),
                'sentiment': analysis.get('original_sentiment', 'neutral'),
                'complexity': analysis.get('transformation_complexity', 'Moderate'),
                'notes': analysis.get('processing_notes', [])
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in convert_text: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': tone_converter.is_loaded(),
        'version': '1.0.0'
    }), 200

@app.route('/api/modes', methods=['GET'])
def get_modes():
    """Get available conversion modes"""
    modes = tone_converter.get_available_modes()
    return jsonify({
        'modes': modes
    }), 200

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure required directories exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)
    
    print("=" * 60)
    print("🚀 Starting Tone Converter Application")
    print("=" * 60)
    print(f"📍 Server running at: http://localhost:5000")
    print(f"📊 Model Status: {'Loaded' if tone_converter.is_loaded() else 'Loading...'}")
    print("=" * 60)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

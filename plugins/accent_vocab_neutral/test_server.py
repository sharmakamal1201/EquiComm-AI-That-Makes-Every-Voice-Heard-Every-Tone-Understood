from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys

# Add the plugin directory to Python path
sys.path.append(os.path.dirname(__file__))

try:
    from vocab_normalizer import normalize_vocabulary
except ImportError:
    # Fallback function if vocab_normalizer can't be imported
    def normalize_vocabulary(text):
        return {
            'original': text,
            'alternate': text,
            'changed': 'None'
        }

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def index():
    """Serve the test UI"""
    with open('test_ui.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/normalize', methods=['POST'])
def normalize_text():
    """API endpoint to normalize text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        input_text = data['text'].strip()
        
        if not input_text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Call the normalizer
        result = normalize_vocabulary(input_text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'original': data.get('text', ''),
            'alternate': data.get('text', ''),
            'changed': f'Error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Vocab normalizer API is running'})

if __name__ == '__main__':
    print("🌍 Starting Vocab Normalizer Test Server...")
    print("📱 Open your browser to: http://localhost:5000")
    print("🚀 Ready to test regional expression normalization!")
    app.run(debug=True, host='0.0.0.0', port=5000)
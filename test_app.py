"""
Unit tests for the Tone Converter application
Run with: pytest test_app.py
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import app
from models.tone_converter import ToneConverter


class TestToneConverter:
    """Test the ToneConverter model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.converter = ToneConverter()
    
    def test_model_initialization(self):
        """Test if model initializes correctly"""
        assert self.converter.is_loaded() == True
    
    def test_available_modes(self):
        """Test available modes"""
        modes = self.converter.get_available_modes()
        assert 'polite' in modes
        assert 'formal' in modes
        assert 'informal' in modes
        assert len(modes) == 6
    
    def test_polite_conversion(self):
        """Test polite conversion"""
        text = "I need this done now"
        result = self.converter.convert(text, 'polite')
        assert result['converted_text'] != text
        assert 'please' in result['converted_text'].lower() or 'would' in result['converted_text'].lower()
    
    def test_formal_conversion(self):
        """Test formal conversion"""
        text = "I can't do this"
        result = self.converter.convert(text, 'formal')
        assert "can't" not in result['converted_text']
        assert "cannot" in result['converted_text']
    
    def test_informal_conversion(self):
        """Test informal conversion"""
        text = "I cannot attend the meeting"
        result = self.converter.convert(text, 'informal')
        # Should have contractions or casual language
        assert result['converted_text'] != text
    
    def test_empty_text(self):
        """Test with empty text"""
        result = self.converter.convert("", 'polite')
        assert result['converted_text'] == ""
    
    def test_alternatives_generation(self):
        """Test alternatives generation"""
        text = "I need help with this task"
        result = self.converter.convert(text, 'polite')
        assert 'alternatives' in result
        assert isinstance(result['alternatives'], list)


class TestFlaskApp:
    """Test the Flask application"""
    
    def setup_method(self):
        """Setup test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_home_page(self):
        """Test home page loads"""
        response = self.client.get('/')
        assert response.status_code == 200
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get('/api/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
    
    def test_modes_endpoint(self):
        """Test modes endpoint"""
        response = self.client.get('/api/modes')
        assert response.status_code == 200
        data = response.get_json()
        assert 'modes' in data
        assert len(data['modes']) == 6
    
    def test_convert_endpoint(self):
        """Test convert endpoint"""
        payload = {
            'text': 'I need this done now',
            'mode': 'polite'
        }
        response = self.client.post('/api/convert', json=payload)
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] == True
        assert 'converted_text' in data
    
    def test_convert_empty_text(self):
        """Test convert with empty text"""
        payload = {
            'text': '',
            'mode': 'polite'
        }
        response = self.client.post('/api/convert', json=payload)
        assert response.status_code == 400
    
    def test_convert_invalid_mode(self):
        """Test convert with invalid mode"""
        payload = {
            'text': 'test text',
            'mode': 'invalid_mode'
        }
        response = self.client.post('/api/convert', json=payload)
        assert response.status_code == 400
    
    def test_404_error(self):
        """Test 404 error handling"""
        response = self.client.get('/nonexistent')
        assert response.status_code == 404


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Dataset Loader for AI Tone Converter
Loads and processes GYAFC, ParaNMT, Wikipedia, and Yelp datasets
"""

import yaml
import os
from typing import Dict, List, Any


class DatasetLoader:
    """Load and parse dataset files for tone conversion"""
    
    def __init__(self, datasets_path: str = 'datasets'):
        self.datasets_path = datasets_path
        self.gyafc_data = None
        self.paranmt_data = None
        self.wikipedia_data = None
        self.yelp_data = None
        self.loaded = False
        
    def load_all_datasets(self) -> bool:
        """Load all dataset files"""
        try:
            print("📚 Loading datasets...")
            
            # Load GYAFC
            gyafc_path = os.path.join(self.datasets_path, 'gyafc_politeness.yaml')
            with open(gyafc_path, 'r', encoding='utf-8') as f:
                self.gyafc_data = yaml.safe_load(f)
            print(f"  ✓ GYAFC: {self.gyafc_data['statistics']['total_patterns']} patterns loaded")
            
            # Load ParaNMT
            paranmt_path = os.path.join(self.datasets_path, 'paranmt_professional.yaml')
            with open(paranmt_path, 'r', encoding='utf-8') as f:
                self.paranmt_data = yaml.safe_load(f)
            print(f"  ✓ ParaNMT: {self.paranmt_data['statistics']['vocabulary_size']} vocab pairs loaded")
            
            # Load Wikipedia
            wiki_path = os.path.join(self.datasets_path, 'wikipedia_formal.yaml')
            with open(wiki_path, 'r', encoding='utf-8') as f:
                self.wikipedia_data = yaml.safe_load(f)
            print(f"  ✓ Wikipedia: {self.wikipedia_data['statistics']['total_rules']} rules loaded")
            
            # Load Yelp
            yelp_path = os.path.join(self.datasets_path, 'yelp_casual.yaml')
            with open(yelp_path, 'r', encoding='utf-8') as f:
                self.yelp_data = yaml.safe_load(f)
            print(f"  ✓ Yelp: {self.yelp_data['statistics']['casual_patterns']} patterns loaded")
            
            self.loaded = True
            print("✅ All datasets loaded successfully!\n")
            return True
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            return False
    
    def get_gyafc_patterns(self) -> Dict:
        """Get GYAFC politeness patterns"""
        if not self.loaded:
            self.load_all_datasets()
        return self.gyafc_data['patterns'] if self.gyafc_data else {}
    
    def get_paranmt_vocabulary(self) -> List[Dict]:
        """Get ParaNMT professional vocabulary"""
        if not self.loaded:
            self.load_all_datasets()
        return self.paranmt_data['professional_vocabulary']['casual_to_professional'] if self.paranmt_data else []
    
    def get_paranmt_prefixes(self) -> List[str]:
        """Get ParaNMT business prefixes"""
        if not self.loaded:
            self.load_all_datasets()
        return self.paranmt_data.get('business_prefixes', []) if self.paranmt_data else []
    
    def get_wikipedia_contractions(self) -> List[Dict]:
        """Get Wikipedia contraction rules"""
        if not self.loaded:
            self.load_all_datasets()
        return self.wikipedia_data['formalization_rules']['contraction_expansion'] if self.wikipedia_data else []
    
    def get_wikipedia_formal_words(self) -> List[Dict]:
        """Get Wikipedia informal to formal mappings"""
        if not self.loaded:
            self.load_all_datasets()
        return self.wikipedia_data['formalization_rules']['informal_to_formal'] if self.wikipedia_data else []
    
    def get_yelp_greetings(self) -> List[str]:
        """Get Yelp casual greetings"""
        if not self.loaded:
            self.load_all_datasets()
        return self.yelp_data['casual_expressions']['greetings'] if self.yelp_data else []
    
    def get_yelp_closings(self) -> List[str]:
        """Get Yelp casual closings"""
        if not self.loaded:
            self.load_all_datasets()
        return self.yelp_data['casual_expressions']['closings'] if self.yelp_data else []
    
    def get_yelp_conversational(self) -> List[Dict]:
        """Get Yelp conversational patterns"""
        if not self.loaded:
            self.load_all_datasets()
        return self.yelp_data['casual_expressions']['conversational_patterns'] if self.yelp_data else []
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded datasets"""
        if not self.loaded:
            self.load_all_datasets()
        
        return {
            'gyafc': self.gyafc_data['statistics'] if self.gyafc_data else {},
            'paranmt': self.paranmt_data['statistics'] if self.paranmt_data else {},
            'wikipedia': self.wikipedia_data['statistics'] if self.wikipedia_data else {},
            'yelp': self.yelp_data['statistics'] if self.yelp_data else {},
            'total_loaded': sum([
                bool(self.gyafc_data),
                bool(self.paranmt_data),
                bool(self.wikipedia_data),
                bool(self.yelp_data)
            ])
        }


# Test the loader
if __name__ == '__main__':
    loader = DatasetLoader()
    if loader.load_all_datasets():
        print("\n📊 Dataset Statistics:")
        stats = loader.get_dataset_stats()
        print(f"  Total datasets loaded: {stats['total_loaded']}/4")
        
        print("\n🧪 Sample Data:")
        print(f"  GYAFC greetings: {loader.get_gyafc_patterns()['greetings'][:2]}")
        print(f"  ParaNMT vocab samples: {loader.get_paranmt_vocabulary()[:2]}")
        print(f"  Yelp greetings: {loader.get_yelp_greetings()[:3]}")

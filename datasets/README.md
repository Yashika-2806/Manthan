# Dataset Information and Statistics
# AI Tone Converter - Multi-Dataset Integration

datasets:
  - name: "GYAFC"
    full_name: "Grammarly's Yahoo Answers Formality Corpus"
    purpose: "Politeness and formality transformations"
    patterns: 150
    file: "gyafc_politeness.yaml"
    source: "NAACL-HLT 2018"
    citation: "Rao & Tetreault (2018)"
    
  - name: "ParaNMT"
    full_name: "Paraphrastic Sentence Embeddings"
    purpose: "Professional vocabulary and paraphrasing"
    patterns: 80
    file: "paranmt_professional.yaml"
    source: "ACL 2018"
    citation: "Wieting & Gimpel (2018)"
    paraphrase_pairs: "50M+"
    
  - name: "Wikipedia Simple English"
    full_name: "Wikipedia Formalization Standards"
    purpose: "Formal grammar rules and simplification"
    patterns: 150
    file: "wikipedia_formal.yaml"
    source: "Wikipedia Style Guide"
    
  - name: "Yelp Reviews"
    full_name: "Yelp Open Dataset"
    purpose: "Casual and friendly expressions"
    patterns: 170
    file: "yelp_casual.yaml"
    source: "Yelp Open Dataset"
    total_reviews: "8M+"

total_patterns: 550
integration_method: "Multi-dataset pattern matching and NLP transformation"
model_version: "2.0-NLP-Enhanced"

features:
  - "Sentiment analysis"
  - "Context-aware transformations"
  - "Pattern matching (550+ rules)"
  - "Confidence scoring"
  - "Alternative generation"
  - "Processing analytics"

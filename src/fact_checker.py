"""
Fact-checking module using QA-based and FactCC approaches.
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple, Optional
import re


class FactChecker:
    """
    Multi-method fact checker for style-transferred text.
    Combines QA-based, FactCC, and NLI approaches.
    """
    
    def __init__(
        self,
        qa_model: str = "distilbert-base-cased-distilled-squad",
        nli_model: str = "microsoft/deberta-base-mnli",
        factcc_model: Optional[str] = None
    ):
        """
        Initialize fact-checking models.
        
        Args:
            qa_model: QA model for question answering
            nli_model: NLI model for entailment checking
            factcc_model: FactCC model path (if available)
        """
        # QA pipeline for answer extraction
        self.qa_pipeline = pipeline("question-answering", model=qa_model)
        
        # NLI model for semantic consistency
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
        self.nli_model.eval()
        
        # FactCC model (optional, needs to be fine-tuned)
        self.factcc_model = None
        if factcc_model:
            self.factcc_tokenizer = AutoTokenizer.from_pretrained(factcc_model)
            self.factcc_model = AutoModelForSequenceClassification.from_pretrained(factcc_model)
            self.factcc_model.eval()
    
    def check_factuality(
        self,
        source_text: str,
        generated_text: str,
        threshold: float = 0.7
    ) -> bool:
        """
        Check if generated text preserves factuality of source.
        
        Args:
            source_text: Original text
            generated_text: Generated style-transferred text
            threshold: Minimum score for factuality (0-1)
            
        Returns:
            True if factual, False otherwise
        """
        # Combine multiple methods
        nli_score = self.check_entailment(source_text, generated_text)
        qa_score = self.check_qa_consistency(source_text, generated_text)
        
        # Average scores
        overall_score = (nli_score + qa_score) / 2
        
        return overall_score >= threshold
    
    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Check if hypothesis is entailed by premise using NLI.
        
        Args:
            premise: Source text
            hypothesis: Generated text
            
        Returns:
            Entailment probability (0-1)
        """
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Assuming label order: [contradiction, neutral, entailment]
        # Adjust based on actual model
        entailment_prob = probs[0][2].item()  # Index 2 for entailment
        
        return entailment_prob
    
    def check_qa_consistency(
        self,
        source_text: str,
        generated_text: str,
        questions: Optional[List[str]] = None
    ) -> float:
        """
        Check consistency using QA-based approach.
        Generate questions from source, answer from both texts, compare.
        
        Args:
            source_text: Original text
            generated_text: Generated text
            questions: Pre-generated questions (auto-generate if None)
            
        Returns:
            QA consistency score (0-1)
        """
        if questions is None:
            questions = self.generate_questions(source_text)
        
        if not questions:
            return 1.0  # No questions to check
        
        matched = 0
        total = len(questions)
        
        for question in questions:
            try:
                source_answer = self.qa_pipeline(
                    question=question,
                    context=source_text
                )["answer"]
                
                generated_answer = self.qa_pipeline(
                    question=question,
                    context=generated_text
                )["answer"]
                
                # Compare answers (exact or fuzzy match)
                if self._answers_match(source_answer, generated_answer):
                    matched += 1
            except:
                # QA failed, skip this question
                total -= 1
                continue
        
        return matched / total if total > 0 else 1.0
    
    def generate_questions(self, text: str, max_questions: int = 3) -> List[str]:
        """
        Generate questions from text for QA-based checking.
        Uses simple heuristics (can be replaced with QG model).
        
        Args:
            text: Input text
            max_questions: Maximum number of questions
            
        Returns:
            List of questions
        """
        questions = []
        
        # Simple heuristic: extract entities and create "What/Who/When/Where" questions
        # This is simplified; use a proper QG model for production
        
        # Extract capitalized words (likely entities)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for entity in entities[:max_questions]:
            # Simple question templates
            if re.search(r'\b(said|says|told|reported)\b', text):
                questions.append(f"What did {entity} say?")
            elif re.search(r'\b\d{4}\b', text):  # Year detected
                questions.append(f"When was {entity} mentioned?")
            else:
                questions.append(f"Who is {entity}?")
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        if numbers:
            questions.append(f"What number is mentioned in the text?")
        
        return questions[:max_questions]
    
    def _answers_match(self, ans1: str, ans2: str, fuzzy: bool = True) -> bool:
        """
        Check if two answers match.
        
        Args:
            ans1: First answer
            ans2: Second answer
            fuzzy: Allow fuzzy matching
            
        Returns:
            True if answers match
        """
        if fuzzy:
            # Normalize and check overlap
            ans1_norm = ans1.lower().strip()
            ans2_norm = ans2.lower().strip()
            
            # Exact match
            if ans1_norm == ans2_norm:
                return True
            
            # One contains the other
            if ans1_norm in ans2_norm or ans2_norm in ans1_norm:
                return True
            
            # Jaccard similarity on words
            words1 = set(ans1_norm.split())
            words2 = set(ans2_norm.split())
            jaccard = len(words1 & words2) / len(words1 | words2) if words1 or words2 else 0
            
            return jaccard > 0.5
        else:
            return ans1.strip() == ans2.strip()
    
    def compute_factcc_score(self, source_text: str, generated_text: str) -> float:
        """
        Compute FactCC-style factual consistency score.
        
        Args:
            source_text: Original text
            generated_text: Generated text
            
        Returns:
            Factuality score (0-1)
        """
        if self.factcc_model is None:
            # Fall back to NLI if FactCC not available
            return self.check_entailment(source_text, generated_text)
        
        inputs = self.factcc_tokenizer(
            source_text,
            generated_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.factcc_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Assuming binary classification: [incorrect, correct]
        factual_prob = probs[0][1].item()
        
        return factual_prob

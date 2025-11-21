"""
Evaluation metrics for style transfer.
"""

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bert_score import score as bert_score
from typing import List, Dict, Tuple
import numpy as np

from src.entity_preserve import EntityPreserver
from src.fact_checker import FactChecker


class StyleTransferEvaluator:
    """
    Comprehensive evaluation for style transfer.
    """
    
    def __init__(
        self,
        style_classifier_model: str = "s-nlp/roberta-base-formality-ranker",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize evaluation models.
        
        Args:
            style_classifier_model: Pre-trained style classifier
            device: Device for computation
        """
        self.device = device
        
        # Style classifier
        try:
            self.style_classifier = pipeline(
                "text-classification",
                model=style_classifier_model,
                device=0 if device == "cuda" else -1
            )
        except:
            print("Warning: Could not load style classifier. Style accuracy will be skipped.")
            self.style_classifier = None
        
        # Entity preserver
        self.entity_preserver = EntityPreserver()
        
        # Fact checker
        self.fact_checker = FactChecker()
    
    def evaluate_batch(
        self,
        sources: List[str],
        generations: List[str],
        references: List[str] = None,
        target_style: str = "formal"
    ) -> Dict[str, float]:
        """
        Evaluate a batch of generations.
        
        Args:
            sources: Source texts
            generations: Generated texts
            references: Reference texts (optional)
            target_style: Target style
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Style accuracy
        if self.style_classifier:
            style_acc = self.compute_style_accuracy(generations, target_style)
            metrics["style_accuracy"] = style_acc
        
        # Content preservation (BERTScore)
        bert_p, bert_r, bert_f1 = self.compute_bertscore(sources, generations)
        metrics["bertscore_precision"] = bert_p
        metrics["bertscore_recall"] = bert_r
        metrics["bertscore_f1"] = bert_f1
        
        # Entity preservation
        entity_preservation = self.compute_entity_preservation(sources, generations)
        metrics["entity_preservation_rate"] = entity_preservation
        
        # Factuality
        factuality_score = self.compute_factuality(sources, generations)
        metrics["factuality_score"] = factuality_score
        
        # Reference-based metrics (if available)
        if references:
            # BLEU
            from sacrebleu import corpus_bleu
            bleu = corpus_bleu(generations, [references])
            metrics["bleu"] = bleu.score
        
        return metrics
    
    def compute_style_accuracy(
        self,
        texts: List[str],
        target_style: str
    ) -> float:
        """
        Compute style classification accuracy.
        
        Args:
            texts: Generated texts
            target_style: Expected style
            
        Returns:
            Accuracy (0-1)
        """
        if not self.style_classifier:
            return 0.0
        
        predictions = self.style_classifier(texts)
        
        # Check how many match target style
        correct = 0
        for pred in predictions:
            pred_label = pred["label"].lower()
            if target_style.lower() in pred_label or pred["score"] > 0.5:
                correct += 1
        
        return correct / len(texts)
    
    def compute_bertscore(
        self,
        sources: List[str],
        generations: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute BERTScore for semantic similarity.
        
        Args:
            sources: Source texts
            generations: Generated texts
            
        Returns:
            (precision, recall, f1) averages
        """
        P, R, F1 = bert_score(
            generations,
            sources,
            lang="en",
            verbose=False,
            device=self.device
        )
        
        return P.mean().item(), R.mean().item(), F1.mean().item()
    
    def compute_entity_preservation(
        self,
        sources: List[str],
        generations: List[str]
    ) -> float:
        """
        Compute entity preservation rate.
        
        Args:
            sources: Source texts
            generations: Generated texts
            
        Returns:
            Average preservation rate
        """
        rates = []
        
        for source, generated in zip(sources, generations):
            result = self.entity_preserver.check_entity_preservation(source, generated)
            rates.append(result["preservation_rate"])
        
        return np.mean(rates) if rates else 0.0
    
    def compute_factuality(
        self,
        sources: List[str],
        generations: List[str],
        threshold: float = 0.7
    ) -> float:
        """
        Compute factuality score using fact checker.
        
        Args:
            sources: Source texts
            generations: Generated texts
            threshold: Threshold for factuality
            
        Returns:
            Average factuality score
        """
        scores = []
        
        for source, generated in zip(sources, generations):
            is_factual = self.fact_checker.check_factuality(source, generated, threshold)
            scores.append(1.0 if is_factual else 0.0)
        
        return np.mean(scores) if scores else 0.0
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics."""
        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        
        for key, value in metrics.items():
            print(f"{key:30s}: {value:.4f}")
        
        print("="*50 + "\n")

# feedback_learner.py - UNIFIED INTELLIGENT LEARNING SYSTEM
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import deque
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import pickle
from river import linear_model, preprocessing, metrics
import numpy as np
from typing import Dict, List, Optional
import hashlib

class UnifiedFeedbackLearner:
    """Single unified learning system that actually works and improves"""
    
    def __init__(self, model_name="distilbert-base-uncased"):
        # Get Hugging Face token from environment
        import os
        hf_token = os.environ.get('HF_TOKEN')
        
        # Transformer model for deep understanding
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            token=hf_token
        )
        
        # Incremental learning with River
        self.incremental_model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        self.quality_predictor = preprocessing.StandardScaler() | linear_model.LinearRegression()
        
        # Memory systems
        self.conversation_memory = deque(maxlen=10000)
        self.feedback_data = []
        self.success_patterns = {}
        self.failure_patterns = {}
        
        # Persistence
        self.db_path = Path("unified_learning.db")
        self._init_database()
        
        # Learning metrics
        self.metrics = {
            'accuracy': metrics.Accuracy(),
            'total_interactions': 0,
            'successful_predictions': 0,
            'learning_rate': 0.0
        }
        
        # Load previous state if exists
        self.load_state()
        
    def _init_database(self):
        """Initialize unified database for all learning data"""
        conn = sqlite3.connect(self.db_path)
        
        # Main interactions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                feedback INTEGER,
                predicted_quality REAL,
                actual_quality REAL,
                features TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Pattern recognition table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,
                pattern_data TEXT,
                success_rate REAL,
                usage_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model checkpoints table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_type TEXT,
                model_data BLOB,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def record_interaction(self, question: str, answer: str, feedback: int):
        """Record and learn from interaction immediately"""
        
        # Extract features for both models
        features = self._extract_comprehensive_features(question, answer)
        
        # Predict quality BEFORE getting feedback (to measure accuracy)
        predicted_quality = self.predict_answer_quality(question, answer)
        
        # Store interaction
        interaction = {
            'question': question,
            'answer': answer,
            'feedback': feedback,
            'predicted_quality': predicted_quality,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_memory.append(interaction)
        self.feedback_data.append(interaction)
        
        # Learn incrementally with River
        self._incremental_learn(features, feedback)
        
        # Update patterns
        self._update_patterns(question, answer, feedback)
        
        # Store in database
        self._store_interaction(question, answer, feedback, predicted_quality, features)
        
        # Update metrics
        self.metrics['total_interactions'] += 1
        if abs(predicted_quality - feedback) < 0.3:
            self.metrics['successful_predictions'] += 1
        
        # Fine-tune transformer if enough data
        if len(self.feedback_data) >= 32:
            self.train_transformer_model()
        
        # Auto-save every 100 interactions
        if self.metrics['total_interactions'] % 100 == 0:
            self.save_state()
            
    def _extract_comprehensive_features(self, question: str, answer: str) -> Dict:
        """Extract features for all learning models"""
        features = {
            # Length features
            'q_length': len(question.split()),
            'a_length': len(answer.split()),
            'q_chars': len(question),
            'a_chars': len(answer),
            
            # Structure features
            'q_questions': question.count('?'),
            'a_sentences': answer.count('.') + answer.count('!') + answer.count('?'),
            'has_code': 1 if '```' in answer or 'def ' in answer else 0,
            'has_list': 1 if any(marker in answer for marker in ['1.', '2.', '- ', '* ']) else 0,
            
            # Content features
            'technical_terms': len([w for w in answer.split() if len(w) > 10]),
            'confidence_phrases': sum(1 for phrase in ['according to', 'research shows', 'studies indicate'] 
                                     if phrase in answer.lower()),
            
            # Complexity features
            'q_unique_words': len(set(question.lower().split())),
            'a_unique_words': len(set(answer.lower().split())),
            'vocabulary_overlap': len(set(question.lower().split()) & set(answer.lower().split())),
            
            # Question type
            'q_type': self._classify_question_type(question),
            
            # Sentiment
            'q_sentiment': self._analyze_sentiment(question),
            'a_sentiment': self._analyze_sentiment(answer)
        }
        
        return features
    
    def _incremental_learn(self, features: Dict, feedback: int):
        """Learn incrementally using River"""
        # Convert features to River format
        x = {str(k): v for k, v in features.items() if isinstance(v, (int, float))}
        y = feedback
        
        # Predict before learning
        y_pred = self.incremental_model.predict_one(x)
        
        # Learn from this example
        self.incremental_model.learn_one(x, y)
        
        # Update metrics
        self.metrics['accuracy'].update(y, y_pred)
        self.metrics['learning_rate'] = self.metrics['accuracy'].get()
        
        # Also update quality predictor
        quality = float(feedback)
        self.quality_predictor.learn_one(x, quality)
    
    def _update_patterns(self, question: str, answer: str, feedback: int):
        """Identify and store successful/failed patterns"""
        pattern_key = f"{self._classify_question_type(question)}:{len(answer)//100}"
        
        if feedback > 0:
            # Successful pattern
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = []
            self.success_patterns[pattern_key].append({
                'q_keywords': self._extract_keywords(question),
                'a_structure': self._analyze_structure(answer),
                'timestamp': datetime.now()
            })
        else:
            # Failed pattern
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = []
            self.failure_patterns[pattern_key].append({
                'q_keywords': self._extract_keywords(question),
                'a_structure': self._analyze_structure(answer),
                'timestamp': datetime.now()
            })
        
        # Store pattern statistics in database
        self._update_pattern_db(pattern_key, feedback)
    
    def train_transformer_model(self):
        """Fine-tune the transformer model on collected feedback"""
        if not self.feedback_data:
            return
        
        # Prepare training data
        texts = []
        labels = []
        
        for item in self.feedback_data:
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            texts.append(text)
            labels.append(item['feedback'])
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create labels tensor
        labels = torch.tensor(labels)
        
        # Training settings
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        
        # Quick fine-tuning
        for epoch in range(3):
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Training epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Clear processed feedback
        self.feedback_data = []
        
        # Save the improved model
        self.save_transformer_model()
    
    def predict_answer_quality(self, question: str, answer: str) -> float:
        """Predict quality using ensemble of models"""
        
        # 1. Transformer prediction
        transformer_score = self._transformer_predict(question, answer)
        
        # 2. Incremental model prediction
        features = self._extract_comprehensive_features(question, answer)
        x = {str(k): v for k, v in features.items() if isinstance(v, (int, float))}
        
        try:
            incremental_score = self.incremental_model.predict_proba_one(x).get(1, 0.5)
        except:
            incremental_score = 0.5
        
        # 3. Pattern-based prediction
        pattern_score = self._pattern_based_predict(question, answer)
        
        # 4. Rule-based heuristics
        heuristic_score = self._heuristic_predict(question, answer)
        
        # Ensemble prediction (weighted average)
        final_score = (
            transformer_score * 0.4 +
            incremental_score * 0.3 +
            pattern_score * 0.2 +
            heuristic_score * 0.1
        )
        
        return min(1.0, max(0.0, final_score))
    
    def _transformer_predict(self, question: str, answer: str) -> float:
        """Use transformer model for prediction"""
        text = f"Question: {question}\nAnswer: {answer}"
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        return probabilities[0][1].item()
    
    def _pattern_based_predict(self, question: str, answer: str) -> float:
        """Predict based on learned patterns"""
        q_type = self._classify_question_type(question)
        pattern_key = f"{q_type}:{len(answer)//100}"
        
        # Check success rate for this pattern
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT success_rate FROM patterns WHERE pattern_type = ?",
            (pattern_key,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        
        # Check similar successful patterns
        if pattern_key in self.success_patterns:
            return 0.7  # Likely successful
        elif pattern_key in self.failure_patterns:
            return 0.3  # Likely to fail
        
        return 0.5  # Unknown
    
    def _heuristic_predict(self, question: str, answer: str) -> float:
        """Simple heuristic-based quality prediction"""
        score = 0.5
        
        # Length appropriateness
        if 50 < len(answer.split()) < 500:
            score += 0.1
        
        # Answer addresses question
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        if len(q_words & a_words) / len(q_words) > 0.3:
            score += 0.1
        
        # Has structure
        if any(marker in answer for marker in ['1.', '2.', '\n-', '\n*']):
            score += 0.1
        
        # Confidence indicators
        if any(phrase in answer.lower() for phrase in ['according to', 'research shows', 'studies']):
            score += 0.1
        
        # Not too short
        if len(answer) < 50:
            score -= 0.2
        
        return min(1.0, max(0.0, score))
    
    def get_improvement_suggestions(self, question: str, answer: str) -> List[str]:
        """Get specific suggestions to improve an answer"""
        suggestions = []
        features = self._extract_comprehensive_features(question, answer)
        
        # Length suggestions
        if features['a_length'] < 20:
            suggestions.append("Expand the answer with more details")
        elif features['a_length'] > 500:
            suggestions.append("Consider making the answer more concise")
        
        # Structure suggestions
        if features['has_list'] == 0 and features['a_sentences'] > 3:
            suggestions.append("Consider using bullet points or numbered lists for clarity")
        
        # Content suggestions
        if features['vocabulary_overlap'] < 3:
            suggestions.append("Ensure the answer directly addresses the question")
        
        if features['confidence_phrases'] == 0:
            suggestions.append("Add supporting evidence or citations")
        
        # Pattern-based suggestions
        q_type = self._classify_question_type(question)
        if q_type == 'how' and features['has_list'] == 0:
            suggestions.append("'How to' questions work well with step-by-step instructions")
        elif q_type == 'why' and features['a_sentences'] < 3:
            suggestions.append("'Why' questions benefit from detailed explanations")
        
        return suggestions
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type"""
        q_lower = question.lower()
        
        if q_lower.startswith('what'):
            return 'what'
        elif q_lower.startswith('how'):
            return 'how'
        elif q_lower.startswith('why'):
            return 'why'
        elif q_lower.startswith('when'):
            return 'when'
        elif q_lower.startswith('where'):
            return 'where'
        elif q_lower.startswith('who'):
            return 'who'
        elif 'code' in q_lower or 'program' in q_lower:
            return 'technical'
        else:
            return 'general'
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        words = text.lower().split()
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'been', 'be'}
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        return keywords[:10]
    
    def _analyze_structure(self, text: str) -> Dict:
        """Analyze text structure"""
        return {
            'paragraphs': text.count('\n\n') + 1,
            'sentences': text.count('.') + text.count('!') + text.count('?'),
            'has_code': '```' in text,
            'has_list': any(marker in text for marker in ['1.', '2.', '- ', '* ']),
            'has_headers': any(text.startswith(h) for h in ['#', '##', '###'])
        }
    
    def _analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis"""
        positive = ['good', 'great', 'excellent', 'love', 'best', 'amazing', 'helpful']
        negative = ['bad', 'hate', 'terrible', 'worst', 'awful', 'unhelpful', 'wrong']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive if word in text_lower)
        neg_count = sum(1 for word in negative if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.5
        
        return pos_count / (pos_count + neg_count)
    
    def _store_interaction(self, question: str, answer: str, feedback: int, 
                          predicted_quality: float, features: Dict):
        """Store interaction in database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO interactions 
            (question, answer, feedback, predicted_quality, actual_quality, features)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (question, answer, feedback, predicted_quality, 
              float(feedback), json.dumps(features)))
        conn.commit()
        conn.close()
    
    def _update_pattern_db(self, pattern_key: str, feedback: int):
        """Update pattern statistics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT success_rate, usage_count FROM patterns WHERE pattern_type = ?",
            (pattern_key,)
        )
        result = cursor.fetchone()
        
        if result:
            # Update existing pattern
            old_rate, count = result
            new_rate = (old_rate * count + feedback) / (count + 1)
            conn.execute(
                "UPDATE patterns SET success_rate = ?, usage_count = ?, last_updated = ? WHERE pattern_type = ?",
                (new_rate, count + 1, datetime.now(), pattern_key)
            )
        else:
            # Create new pattern
            conn.execute(
                "INSERT INTO patterns (pattern_type, success_rate, usage_count) VALUES (?, ?, ?)",
                (pattern_key, float(feedback), 1)
            )
        
        conn.commit()
        conn.close()
    
    def save_state(self):
        """Save complete learner state"""
        # Save transformer model
        self.save_transformer_model()
        
        # Save incremental models
        checkpoint = {
            'incremental_model': pickle.dumps(self.incremental_model),
            'quality_predictor': pickle.dumps(self.quality_predictor),
            'success_patterns': self.success_patterns,
            'failure_patterns': self.failure_patterns,
            'metrics': self.metrics
        }
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO checkpoints (model_type, model_data, metrics) VALUES (?, ?, ?)",
            ('unified', pickle.dumps(checkpoint), json.dumps(self.metrics))
        )
        conn.commit()
        conn.close()
        
        print(f"✅ Saved state: {self.metrics['total_interactions']} interactions learned")
    
    def load_state(self):
        """Load previous learner state"""
        try:
            # Load latest checkpoint
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                "SELECT model_data, metrics FROM checkpoints WHERE model_type = 'unified' ORDER BY id DESC LIMIT 1"
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                checkpoint = pickle.loads(result[0])
                self.incremental_model = pickle.loads(checkpoint['incremental_model'])
                self.quality_predictor = pickle.loads(checkpoint['quality_predictor'])
                self.success_patterns = checkpoint['success_patterns']
                self.failure_patterns = checkpoint['failure_patterns']
                self.metrics = json.loads(result[1])
                
                print(f"✅ Loaded previous state: {self.metrics['total_interactions']} interactions")
                
        except Exception as e:
            print(f"Could not load previous state: {e}")
    
    def save_transformer_model(self):
        """Save fine-tuned transformer model"""
        model_dir = Path("./models/feedback_tuned")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
    
    def get_learning_statistics(self) -> Dict:
        """Get comprehensive learning statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Get interaction stats
        cursor = conn.execute("SELECT COUNT(*), AVG(feedback), AVG(predicted_quality) FROM interactions")
        total, avg_feedback, avg_predicted = cursor.fetchone()
        
        # Get pattern stats
        cursor = conn.execute("SELECT COUNT(*), AVG(success_rate) FROM patterns")
        pattern_count, avg_success = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_interactions': total or 0,
            'average_feedback': avg_feedback or 0,
            'average_predicted_quality': avg_predicted or 0,
            'prediction_accuracy': self.metrics.get('learning_rate', 0),
            'patterns_learned': pattern_count or 0,
            'average_pattern_success': avg_success or 0,
            'successful_predictions': self.metrics.get('successful_predictions', 0),
            'success_rate': (self.metrics.get('successful_predictions', 0) / 
                           max(1, self.metrics.get('total_interactions', 1))) * 100
        }

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import deque
import json

class FeedbackLearner:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2  # Positive/Negative feedback
        )
        
        # Store conversation history and feedback
        self.conversation_memory = deque(maxlen=10000)
        self.feedback_data = []
        
    def record_interaction(self, question: str, answer: str, feedback: int):
        """Record user interaction with feedback (1 for thumbs up, 0 for down)"""
        
        interaction = {
            'question': question,
            'answer': answer,
            'feedback': feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        self.conversation_memory.append(interaction)
        self.feedback_data.append(interaction)
        
        # Trigger learning after collecting enough feedback
        if len(self.feedback_data) >= 32:  # Batch size
            self.train_on_feedback()
    
    def train_on_feedback(self):
        """Fine-tune model based on user feedback"""
        
        # Prepare training data
        texts = []
        labels = []
        
        for item in self.feedback_data:
            # Combine question and answer
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"
            texts.append(text)
            labels.append(item['feedback'])
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Create labels tensor
        labels = torch.tensor(labels)
        
        # Training loop
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        self.model.train()
        
        for epoch in range(3):  # Quick fine-tuning
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        
        # Clear processed feedback
        self.feedback_data = []
        
        # Save the improved model
        self.save_model()
    
    def save_model(self):
        """Save the fine-tuned model"""
        self.model.save_pretrained("./models/feedback_tuned")
        self.tokenizer.save_pretrained("./models/feedback_tuned")
    
    def predict_answer_quality(self, question: str, answer: str) -> float:
        """Predict if an answer will be well-received"""
        
        text = f"Question: {question}\nAnswer: {answer}"
        inputs = self.tokenizer(text, return_tensors='pt')
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
        # Return probability of positive feedback
        return probabilities[0][1].item()

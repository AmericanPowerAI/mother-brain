# advanced_features.py - COMPLETE REVISED VERSION WITH PERSISTENT LEARNING
import asyncio
import chromadb
from datetime import datetime
from typing import Dict, List, Any
import httpx
import optuna
import wandb
from torch.utils.tensorboard import SummaryWriter

# NEW IMPORTS FOR PERSISTENT LEARNING
import pickle
import sqlite3
import json
from pathlib import Path
import torch
import safetensors.torch
from datasets import load_dataset
from river import linear_model, preprocessing, metrics
from river import stream
import numpy as np
from collections import deque
import hashlib

# Only import these if you have them installed
try:
    from scapy.all import ARP, Ether, srp
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from langchain import LLMChain, PromptTemplate
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


class AdvancedCapabilities:
    """Advanced features that extend MotherBrain with persistent learning"""
    
    def __init__(self, mother_brain_instance=None):
        self.mother = mother_brain_instance
        self.setup_components()
        # NEW: Setup persistent learning systems
        self.setup_persistent_learning()
        self.setup_incremental_learning()
        self.setup_datasets()
    
    def setup_components(self):
        """Initialize all advanced components"""
        # ChromaDB for memory
        try:
            self.chroma_client = chromadb.Client()
            self.conversation_collection = self.chroma_client.create_collection("conversations")
        except:
            self.conversation_collection = None
        
        # Monitoring tools
        self.wandb_initialized = False
        self.tensorboard_writer = None
        
        # LangChain setup if available
        if LANGCHAIN_AVAILABLE:
            self.setup_langchain()
    
    # ===== NEW: PERSISTENT STORAGE SYSTEM =====
    def setup_persistent_learning(self):
        """Initialize persistent storage for continuous learning"""
        # Create storage directories
        self.storage_dir = Path("ai_persistence")
        self.models_dir = self.storage_dir / "models"
        self.memory_dir = self.storage_dir / "memory"
        self.patterns_dir = self.storage_dir / "patterns"
        
        for dir in [self.models_dir, self.memory_dir, self.patterns_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent database
        self.db_path = self.storage_dir / "ai_brain.db"
        self.init_persistent_database()
        
        # Load previous learning state if exists
        self.load_learning_state()
    
    def init_persistent_database(self):
        """Create tables for persistent memory"""
        self.db_conn = sqlite3.connect(self.db_path)
        
        # Learning patterns table
        self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE,
                pattern_text TEXT,
                success_rate REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP,
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Code improvements table
        self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS code_improvements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_code TEXT,
                improved_code TEXT,
                improvement_type TEXT,
                user_accepted BOOLEAN,
                performance_gain REAL,
                learned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Conversation quality table
        self.db_conn.execute('''
            CREATE TABLE IF NOT EXISTS conversation_quality (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                quality_score REAL,
                user_feedback INTEGER,
                response_time REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.db_conn.commit()
    
    def save_learning_checkpoint(self, name="main"):
        """Save current learning state"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'learned_patterns': self.get_all_patterns(),
            'incremental_model': pickle.dumps(self.incremental_model) if hasattr(self, 'incremental_model') else None,
            'metrics': self.learning_metrics if hasattr(self, 'learning_metrics') else {},
            'conversation_count': self.get_conversation_count()
        }
        
        checkpoint_path = self.models_dir / f"checkpoint_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Also save a "latest" version for easy loading
        latest_path = self.models_dir / f"checkpoint_{name}_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        return checkpoint_path
    
    def load_learning_state(self):
        """Load previous learning state if exists"""
        latest_path = self.models_dir / "checkpoint_main_latest.pkl"
        if latest_path.exists():
            try:
                with open(latest_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                
                # Restore incremental model
                if checkpoint.get('incremental_model'):
                    self.incremental_model = pickle.loads(checkpoint['incremental_model'])
                
                # Restore metrics
                self.learning_metrics = checkpoint.get('metrics', {})
                
                print(f"✅ Loaded learning state from {checkpoint['timestamp']}")
                print(f"   - {checkpoint['conversation_count']} conversations learned")
                print(f"   - {len(checkpoint['learned_patterns'])} patterns stored")
                
                return True
            except Exception as e:
                print(f"❌ Failed to load checkpoint: {e}")
        return False
    
    # ===== NEW: INCREMENTAL LEARNING SYSTEM =====
    def setup_incremental_learning(self):
        """Setup River for incremental learning"""
        # Create incremental models for different tasks
        self.incremental_model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
        
        # Quality predictor for responses
        self.quality_predictor = preprocessing.StandardScaler() | linear_model.LinearRegression()
        
        # Metrics tracking
        self.learning_metrics = {
            'accuracy': metrics.Accuracy(),
            'mae': metrics.MAE(),
            'learning_rate': metrics.RollingMean(window_size=100)
        }
        
        # Buffer for recent learning examples
        self.learning_buffer = deque(maxlen=1000)
    
    def learn_from_interaction(self, query: str, response: str, feedback: float):
        """Learn from a single interaction immediately"""
        # Extract features
        features = self.extract_interaction_features(query, response)
        
        # Convert to River format
        x = dict(enumerate(features))
        y = 1 if feedback > 0.5 else 0
        
        # Predict before learning
        y_pred = self.incremental_model.predict_one(x)
        
        # Learn from this example
        self.incremental_model.learn_one(x, y)
        
        # Update metrics
        self.learning_metrics['accuracy'].update(y, y_pred)
        
        # Store in database
        self.store_learning_example(query, response, feedback)
        
        # Add to buffer
        self.learning_buffer.append({
            'query': query,
            'response': response,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
        
        # Auto-save every 100 interactions
        if len(self.learning_buffer) % 100 == 0:
            self.save_learning_checkpoint()
        
        return self.learning_metrics['accuracy'].get()
    
    def extract_interaction_features(self, query: str, response: str):
        """Extract features from interaction for learning"""
        return [
            len(query.split()),  # Query length
            len(response.split()),  # Response length
            query.count('?'),  # Number of questions
            response.count('.'),  # Number of sentences
            1 if 'code' in query.lower() else 0,  # Is code-related
            1 if '```' in response else 0,  # Contains code block
            len(set(query.split())),  # Query vocabulary size
            len(set(response.split())),  # Response vocabulary size
        ]
    
    # ===== NEW: DATASET INTEGRATION =====
    def setup_datasets(self):
        """Load and prepare datasets for training"""
        self.datasets = {}
        self.dataset_iterators = {}
        
        # Load datasets with streaming to save memory
        try:
            # Code dataset - CodeSearchNet
            self.datasets['code'] = load_dataset(
                "code_search_net", 
                "python",
                split="train",
                streaming=True
            )
            
            # Conversation dataset - UltraChat (smaller subset)
            self.datasets['conversation'] = load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft",
                streaming=True
            )
            
            # Start background learning
            asyncio.create_task(self.background_dataset_learning())
            
        except Exception as e:
            print(f"Dataset loading error: {e}")
            print("Will continue without datasets")
    
    async def background_dataset_learning(self):
        """Learn from datasets in background"""
        learning_cycles = 0
        
        while True:
            try:
                # Learn from code dataset
                if 'code' in self.datasets:
                    code_examples = self.datasets['code'].take(10)
                    for example in code_examples:
                        await self.learn_from_code_example(example)
                
                # Learn from conversation dataset
                if 'conversation' in self.datasets:
                    conv_examples = self.datasets['conversation'].take(5)
                    for example in conv_examples:
                        await self.learn_from_conversation_example(example)
                
                learning_cycles += 1
                
                # Save checkpoint every 100 cycles
                if learning_cycles % 100 == 0:
                    self.save_learning_checkpoint()
                    print(f"📚 Background learning: {learning_cycles} cycles completed")
                
                # Sleep to avoid overwhelming the system
                await asyncio.sleep(30)  # Learn every 30 seconds
                
            except Exception as e:
                print(f"Background learning error: {e}")
                await asyncio.sleep(60)
    
    async def learn_from_code_example(self, example):
        """Learn patterns from code examples"""
        code = example.get('func_code_string', '')
        docstring = example.get('func_documentation_string', '')
        
        if code and docstring:
            # Extract pattern
            pattern = self.extract_code_pattern(code, docstring)
            
            # Store in database
            self.store_learned_pattern(pattern)
            
            # Update incremental model
            features = self.extract_code_features(code)
            quality = self.assess_code_quality(code, docstring)
            
            x = dict(enumerate(features))
            self.quality_predictor.learn_one(x, quality)
    
    async def learn_from_conversation_example(self, example):
        """Learn from conversation examples"""
        messages = example.get('messages', [])
        
        if len(messages) >= 2:
            # Extract query-response pairs
            for i in range(0, len(messages)-1, 2):
                if i+1 < len(messages):
                    query = messages[i].get('content', '')
                    response = messages[i+1].get('content', '')
                    
                    if query and response:
                        # Simulate quality (in real scenario, would be from user feedback)
                        quality = self.estimate_response_quality(query, response)
                        
                        # Learn from this interaction
                        self.learn_from_interaction(query, response, quality)
    
    def extract_code_pattern(self, code: str, docstring: str):
        """Extract reusable pattern from code"""
        return {
            'pattern_hash': hashlib.md5(code.encode()).hexdigest(),
            'code_snippet': code[:500],  # Store first 500 chars
            'description': docstring[:200],
            'language': 'python',
            'features': self.extract_code_features(code)
        }
    
    def extract_code_features(self, code: str):
        """Extract features from code for learning"""
        return [
            len(code),
            code.count('\n'),  # Lines
            code.count('def '),  # Functions
            code.count('class '),  # Classes
            code.count('#'),  # Comments
            1 if '"""' in code or "'''" in code else 0,  # Has docstring
            code.count('try:'),  # Error handling
            code.count('import '),  # Dependencies
        ]
    
    def assess_code_quality(self, code: str, docstring: str):
        """Assess quality of code (0-1 score)"""
        quality = 0.5  # Base score
        
        # Has documentation
        if docstring and len(docstring) > 20:
            quality += 0.2
        
        # Has error handling
        if 'try:' in code:
            quality += 0.1
        
        # Reasonable length
        lines = code.count('\n')
        if 5 < lines < 50:
            quality += 0.1
        
        # Has comments
        if '#' in code:
            quality += 0.1
        
        return min(1.0, quality)
    
    def estimate_response_quality(self, query: str, response: str):
        """Estimate quality of response"""
        # Simple heuristics (replace with actual user feedback)
        quality = 0.5
        
        # Response addresses query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        quality += overlap * 0.2
        
        # Reasonable length
        if 20 < len(response.split()) < 200:
            quality += 0.2
        
        # Contains structure
        if any(marker in response for marker in ['1.', '2.', '```', '- ']):
            quality += 0.1
        
        return min(1.0, quality)
    
    # ===== NEW: DATABASE OPERATIONS =====
    def store_learned_pattern(self, pattern: Dict):
        """Store pattern in database"""
        try:
            self.db_conn.execute('''
                INSERT OR REPLACE INTO learned_patterns 
                (pattern_hash, pattern_text, context, last_used)
                VALUES (?, ?, ?, ?)
            ''', (
                pattern['pattern_hash'],
                pattern.get('code_snippet', ''),
                json.dumps(pattern),
                datetime.now()
            ))
            self.db_conn.commit()
        except Exception as e:
            print(f"Failed to store pattern: {e}")
    
    def store_learning_example(self, query: str, response: str, feedback: float):
        """Store learning example in database"""
        try:
            self.db_conn.execute('''
                INSERT INTO conversation_quality 
                (query, response, quality_score, user_feedback)
                VALUES (?, ?, ?, ?)
            ''', (query, response, feedback, int(feedback > 0.5)))
            self.db_conn.commit()
        except Exception as e:
            print(f"Failed to store learning example: {e}")
    
    def get_all_patterns(self):
        """Retrieve all learned patterns"""
        cursor = self.db_conn.execute(
            'SELECT pattern_hash, pattern_text, success_rate FROM learned_patterns'
        )
        return cursor.fetchall()
    
    def get_conversation_count(self):
        """Get total conversations learned"""
        cursor = self.db_conn.execute(
            'SELECT COUNT(*) FROM conversation_quality'
        )
        return cursor.fetchone()[0]
    
    # ===== NEW: ENHANCED PREDICTION =====
    def predict_response_quality(self, query: str, response: str) -> float:
        """Predict if response will be good based on learning"""
        if not hasattr(self, 'incremental_model'):
            return 0.5
        
        features = self.extract_interaction_features(query, response)
        x = dict(enumerate(features))
        
        try:
            # Get probability of positive feedback
            prediction = self.incremental_model.predict_proba_one(x)
            return prediction.get(1, 0.5)
        except:
            # Fallback if model not ready
            return 0.5
    
    # ===== ORIGINAL: NETWORK SCANNING =====
    def scan_network(self, ip_range="192.168.1.0/24") -> Dict:
        """Scan network for devices"""
        if not SCAPY_AVAILABLE:
            return {"error": "Scapy not installed"}
        
        try:
            arp_request = ARP(pdst=ip_range)
            broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
            packet = broadcast/arp_request
            answered = srp(packet, timeout=2, verbose=False)[0]
            devices = [{'ip': r[1].psrc, 'mac': r[1].hwsrc} for r in answered]
            
            # Store in mother's knowledge if available
            if self.mother:
                self.mother.knowledge[f"NETWORK:SCAN:{datetime.now().isoformat()}"] = devices
            
            return {"devices_found": len(devices), "devices": devices}
        except Exception as e:
            return {"error": str(e)}
    
    # ===== ORIGINAL: DYNAMIC SCRAPING =====
    def scrape_dynamic_content(self, url: str) -> str:
        """Scrape JavaScript-heavy sites"""
        if not SELENIUM_AVAILABLE:
            return "Selenium not installed - using basic scraping instead"
        
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            
            driver = webdriver.Chrome(options=options)
            driver.get(url)
            content = driver.page_source
            driver.quit()
            
            return content
        except Exception as e:
            return f"Selenium error: {str(e)}"
    
    # ===== ORIGINAL: ENHANCED MEMORY =====
    def store_conversation(self, text: str, metadata: Dict) -> bool:
        """Store conversation in ChromaDB"""
        if not self.conversation_collection:
            return False
        
        try:
            self.conversation_collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[f"conv_{datetime.now().timestamp()}"]
            )
            return True
        except Exception as e:
            print(f"ChromaDB storage error: {e}")
            return False
    
    def search_memory(self, query: str, n_results: int = 5) -> Dict:
        """Search conversation memory"""
        if not self.conversation_collection:
            return {"error": "ChromaDB not initialized"}
        
        try:
            results = self.conversation_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            return {"error": str(e)}
    
    # ===== ORIGINAL: LANGCHAIN SETUP =====
    def setup_langchain(self):
        """Setup LangChain components"""
        if not LANGCHAIN_AVAILABLE:
            return
        
        template = """You are MOTHER AI. Previous conversation:
        {history}
        Human: {input}
        MOTHER AI:"""
        
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )
        self.conversation_memory = ConversationBufferMemory()
    
    # ===== ORIGINAL: NEURAL NETWORK OPTIMIZATION =====
    def optimize_network_params(self, train_function) -> Dict:
        """Optimize neural network hyperparameters"""
        def objective(trial):
            params = {
                'hidden_size': trial.suggest_int('hidden_size', 32, 256),
                'learning_rate': trial.suggest_loguniform('lr', 1e-5, 1e-1),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
                'dropout': trial.suggest_float('dropout', 0.1, 0.5)
            }
            
            # Call the provided training function with params
            accuracy = train_function(params)
            return accuracy
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)
        
        return study.best_params
    
    # ===== ORIGINAL: MONITORING =====
    def log_metrics(self, metrics: Dict):
        """Log metrics to WandB and TensorBoard"""
        # WandB
        if not self.wandb_initialized:
            try:
                wandb.init(project="mother-ai")
                self.wandb_initialized = True
            except:
                pass
        
        if self.wandb_initialized:
            wandb.log(metrics)
        
        # TensorBoard
        if not self.tensorboard_writer:
            try:
                self.tensorboard_writer = SummaryWriter('runs/mother_ai')
            except:
                pass
        
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f'Metrics/{key}', value, metrics.get('step', 0))
    
    # ===== ORIGINAL: AUDIO PROCESSING =====
    def analyze_audio(self, audio_path: str) -> Dict:
        """Analyze audio file"""
        if not LIBROSA_AVAILABLE:
            return {"error": "Librosa not installed"}
        
        try:
            y, sr = librosa.load(audio_path)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            return {
                "tempo": float(tempo),
                "duration": len(y) / sr,
                "sample_rate": sr,
                "mfcc_features": mfcc.shape[1]
            }
        except Exception as e:
            return {"error": str(e)}
    
    # ===== ORIGINAL: ASYNC FETCHING =====
    async def fetch_urls_async(self, urls: List[str]) -> List[Dict]:
        """Fetch multiple URLs concurrently"""
        async with httpx.AsyncClient() as client:
            tasks = []
            for url in urls:
                tasks.append(client.get(url))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    results.append({"url": urls[i], "error": str(response)})
                else:
                    results.append({
                        "url": urls[i],
                        "status": response.status_code,
                        "content_length": len(response.content)
                    })
            
            return results
    
    # ===== ENHANCED INTEGRATION METHODS =====
    def enhance_mother_brain(self, mother_instance):
        """Connect this to MotherBrain instance with persistent learning"""
        self.mother = mother_instance
        
        # Add original methods
        mother_instance.scan_network = self.scan_network
        mother_instance.scrape_dynamic = self.scrape_dynamic_content
        mother_instance.search_memory = self.search_memory
        mother_instance.log_metrics = self.log_metrics
        
        # ADD NEW LEARNING METHODS
        mother_instance.learn_from_feedback = self.learn_from_interaction
        mother_instance.predict_quality = self.predict_response_quality
        mother_instance.save_brain = self.save_learning_checkpoint
        mother_instance.get_learning_stats = lambda: {
            'conversations_learned': self.get_conversation_count(),
            'patterns_stored': len(self.get_all_patterns()),
            'accuracy': self.learning_metrics['accuracy'].get() if hasattr(self, 'learning_metrics') else 0,
            'buffer_size': len(self.learning_buffer) if hasattr(self, 'learning_buffer') else 0
        }
        
        print("✨ MotherBrain enhanced with persistent learning!")
        print("   - Previous learning state loaded")
        print("   - Background dataset learning active")
        print("   - Auto-saving every 100 interactions")
        
        return mother_instance

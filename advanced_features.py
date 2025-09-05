# advanced_features.py
import asyncio
import chromadb
from datetime import datetime
from typing import Dict, List, Any
import httpx
import optuna
import wandb
from torch.utils.tensorboard import SummaryWriter

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
    """Advanced features that extend MotherBrain"""
    
    def __init__(self, mother_brain_instance=None):
        self.mother = mother_brain_instance
        self.setup_components()
    
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
    
    # ===== NETWORK SCANNING =====
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
    
    # ===== DYNAMIC SCRAPING =====
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
    
    # ===== ENHANCED MEMORY =====
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
    
    # ===== LANGCHAIN SETUP =====
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
    
    # ===== NEURAL NETWORK OPTIMIZATION =====
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
    
    # ===== MONITORING =====
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
    
    # ===== AUDIO PROCESSING =====
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
    
    # ===== ASYNC FETCHING =====
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
    
    # ===== INTEGRATION METHODS =====
    def enhance_mother_brain(self, mother_instance):
        """Connect this to MotherBrain instance"""
        self.mother = mother_instance
        
        # Add our methods to mother brain
        mother_instance.scan_network = self.scan_network
        mother_instance.scrape_dynamic = self.scrape_dynamic_content
        mother_instance.search_memory = self.search_memory
        mother_instance.log_metrics = self.log_metrics
        
        return mother_instance

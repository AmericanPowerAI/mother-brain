import numpy as np
import json
import lzma
import hashlib
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, List, Tuple
import umap
import faiss
import warnings
from datetime import datetime

class KnowledgeCompressor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.cluster_model = None
        self.dimension = 384  # Dimension of MiniLM embeddings
        warnings.filterwarnings('ignore', category=UserWarning)

    def compress(self, knowledge: Dict) -> Tuple[bytes, Dict]:
        """Main compression pipeline with ML"""
        # Phase 1: Semantic Embedding
        texts = [f"{k}:{v}" for k, v in knowledge.items() if not k.startswith('_')]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Phase 2: Dimensionality Reduction
        reducer = umap.UMAP(n_components=128, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Phase 3: Clustering
        self.cluster_model = MiniBatchKMeans(n_clusters=min(100, len(texts)), batch_size=512)
        clusters = self.cluster_model.fit_predict(reduced_embeddings)
        
        # Phase 4: Indexing
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        
        # Create compressed structure
        compressed = {
            '_meta': {
                'version': 'ml-v2',
                'timestamp': datetime.utcnow().isoformat(),
                'stats': {
                    'original_size': len(json.dumps(knowledge)),
                    'clusters': len(set(clusters))
                }
            },
            'clusters': clusters.tolist(),
            'embeddings': reduced_embeddings.astype('float16').tolist(),
            'knowledge_keys': list(knowledge.keys())
        }
        
        # Phase 5: Hybrid Compression
        compressed_bytes = lzma.compress(
            json.dumps(compressed).encode('utf-8'),
            preset=9 | lzma.PRESET_EXTREME
        )
        
        return compressed_bytes, {
            'compression_ratio': len(compressed_bytes) / len(json.dumps(knowledge)),
            'clusters': len(set(clusters)),
            'dimensionality': 128
        }

    def decompress(self, compressed_bytes: bytes) -> Dict:
        """Reconstruct knowledge from compressed form"""
        decompressed = json.loads(lzma.decompress(compressed_bytes))
        
        # Rebuild embeddings
        embeddings = np.array(decompressed['embeddings'], dtype='float16')
        
        # Return original structure if no ML compression
        if '_original' in decompressed:
            return decompressed['_original']
        
        # Reconstruct approximate knowledge
        return {
            '_meta': decompressed['_meta'],
            **{k: self._find_similar(k, embeddings, decompressed) 
               for k in decompressed['knowledge_keys']}
        }

    def _find_similar(self, key: str, embeddings: np.ndarray, data: Dict) -> str:
        """Find similar entries using FAISS"""
        idx = data['knowledge_keys'].index(key)
        query = embeddings[idx].astype('float32').reshape(1, -1)
        
        # Semantic search
        _, indices = self.index.search(query, k=3)
        similar_keys = [data['knowledge_keys'][i] for i in indices[0] if i != idx]
        
        return f"SEMANTIC_CLUSTER:{hashlib.md5(','.join(similar_keys).encode()).hexdigest()[:8]}"

    def optimize_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Dynamic cluster optimization with elbow method"""
        max_k = min(50, len(embeddings)//10)
        distortions = []
        
        for k in range(2, max_k+1):
            kmeans = MiniBatchKMeans(n_clusters=k, batch_size=512)
            kmeans.fit(embeddings)
            distortions.append(kmeans.inertia_)
        
        # Find optimal k using knee locator
        optimal_k = self._find_elbow(distortions) + 2
        self.cluster_model = MiniBatchKMeans(n_clusters=optimal_k, batch_size=512)
        return self.cluster_model.fit_predict(embeddings)

    def _find_elbow(self, distortions: List[float]) -> int:
        """Automatically locate elbow point"""
        n_points = len(distortions)
        all_coords = np.vstack((range(n_points), distortions)).T
        line_vec = all_coords[-1] - all_coords[0]
        line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
        vec_from_first = all_coords - all_coords[0]
        scalar_prod = np.sum(vec_from_first * line_vec_norm, axis=1)
        vec_to_line = vec_from_first - np.outer(scalar_prod, line_vec_norm)
        return np.argmax(np.sqrt(np.sum(vec_to_line**2, axis=1)))

import json
import torch
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseRAG:
    """Handles knowledge base loading, embeddings, and semantic search."""
    
    def __init__(self, kb_path: str = "knowledge_base.json"):
        self.kb_path = kb_path
        self.knowledge_base = []
        self.embeddings = None
        self.index = None
        
        # Initialize sentence transformer for embeddings
        logger.info("Loading sentence transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self._load_knowledge_base()
        self._create_embeddings()
        self._build_faiss_index()
    
    def _load_knowledge_base(self):
        """Load knowledge base from JSON file."""
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.knowledge_base = data.get('qa_pairs', [])
            logger.info(f"Loaded {len(self.knowledge_base)} Q&A pairs from knowledge base")
        except FileNotFoundError:
            logger.error(f"Knowledge base file not found: {self.kb_path}")
            self.knowledge_base = []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing knowledge base JSON: {e}")
            self.knowledge_base = []
    
    def _create_embeddings(self):
        """Create embeddings for all questions in knowledge base."""
        if not self.knowledge_base:
            logger.warning("No knowledge base entries to embed")
            return
        
        questions = [item['question'] for item in self.knowledge_base]
        logger.info("Creating embeddings for knowledge base questions...")
        self.embeddings = self.embedding_model.encode(questions, convert_to_numpy=True)
        logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search."""
        if self.embeddings is None or len(self.embeddings) == 0:
            logger.warning("No embeddings available to build FAISS index")
            return
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Search for similar questions in knowledge base.
        
        Returns list of dicts with: question, answer, similarity_score
        """
        if self.index is None or len(self.knowledge_base) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.knowledge_base):
                # Convert L2 distance to similarity score (0-1)
                similarity = 1 / (1 + dist)
                
                # Also use fuzzy matching for additional validation
                fuzzy_score = fuzz.ratio(query.lower(), 
                                        self.knowledge_base[idx]['question'].lower()) / 100.0
                
                # Combine scores (weighted average)
                combined_score = 0.7 * similarity + 0.3 * fuzzy_score
                
                results.append({
                    'question': self.knowledge_base[idx]['question'],
                    'answer': self.knowledge_base[idx]['answer'],
                    'similarity_score': combined_score,
                    'category': self.knowledge_base[idx].get('category', 'general')
                })
        
        return sorted(results, key=lambda x: x['similarity_score'], reverse=True)


class AtlasSystem:
    """Main Atlas RAG system combining knowledge base and LLM."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize knowledge base RAG
        self.kb_rag = KnowledgeBaseRAG()
        
        # Load Atlas model
        self._load_model()
        
        # Response cache
        self.cache = {}
        
        # Statistics tracking
        self.stats = {
            'kb_hits': 0,
            'model_hits': 0,
            'hybrid_hits': 0,
            'total_queries': 0,
            'confidence_scores': [],
            'start_time': datetime.now()
        }
    
    def _load_model(self):
        """Load the Atlas 1.1B model."""
        try:
            logger.info(f"Loading Atlas model from {self.model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key for a question."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()
    
    def _generate_model_response(self, question: str, context: Optional[str] = None) -> str:
        """Generate response using the Atlas model."""
        try:
            if context:
                prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            else:
                prompt = f"Question: {question}\n\nAnswer:"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error generating model response: {e}")
            return "I'm sorry, I encountered an error generating a response."
    
    def answer_question(self, question: str) -> Dict:
        """
        Answer a question using RAG approach.
        
        Returns dict with: answer, confidence, source, matches
        """
        self.stats['total_queries'] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(question)
        if cache_key in self.cache:
            logger.info("Returning cached response")
            return self.cache[cache_key]
        
        # Step 1: Semantic search in knowledge base
        matches = self.kb_rag.search(question, top_k=3)
        
        if not matches:
            # No matches - use model only
            logger.info("No KB matches found - using model")
            answer = self._generate_model_response(question)
            confidence = 0.5
            source = "Model"
            self.stats['model_hits'] += 1
        else:
            best_match = matches[0]
            similarity = best_match['similarity_score']
            
            # Step 2: Decision logic based on similarity
            if similarity > 0.85:
                # High confidence - use KB answer directly
                logger.info(f"High similarity ({similarity:.2f}) - using KB answer")
                answer = best_match['answer']
                confidence = similarity
                source = "Knowledge Base"
                self.stats['kb_hits'] += 1
                
            elif similarity >= 0.70:
                # Medium confidence - hybrid approach
                logger.info(f"Medium similarity ({similarity:.2f}) - using hybrid approach")
                kb_answer = best_match['answer']
                elaboration = self._generate_model_response(question, context=kb_answer)
                answer = f"{kb_answer}\n\nAdditional context: {elaboration}"
                confidence = (similarity + 0.7) / 2
                source = "Hybrid (KB + Model)"
                self.stats['hybrid_hits'] += 1
                
            else:
                # Low confidence - use model only
                logger.info(f"Low similarity ({similarity:.2f}) - using model")
                answer = self._generate_model_response(question)
                confidence = 0.6
                source = "Model"
                self.stats['model_hits'] += 1
        
        # Step 3: Prepare response
        result = {
            'answer': answer,
            'confidence': confidence,
            'source': source,
            'matches': matches[:3],
            'timestamp': datetime.now().isoformat()
        }
        
        # Update statistics
        self.stats['confidence_scores'].append(confidence)
        
        # Cache the response
        self.cache[cache_key] = result
        
        return result
    
    def interactive_chat(self):
        """Start interactive chat session."""
        print("=" * 70)
        print("ATLAS RAG System - Interactive Chat")
        print("=" * 70)
        print("Type 'quit', 'exit', or 'q' to end the session")
        print("Type 'stats' to see usage statistics")
        print("=" * 70)
        print()
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using Atlas RAG System.")
                    self.print_statistics()
                    break
                
                if question.lower() == 'stats':
                    self.print_statistics()
                    continue
                
                print("\n⏳ Processing...")
                result = self.answer_question(question)
                
                print("\n" + "=" * 70)
                print(f"💡 Answer ({result['source']}):")
                print("-" * 70)
                print(result['answer'])
                print("-" * 70)
                print(f"📊 Confidence: {result['confidence']:.2%}")
                
                if result['matches']:
                    print(f"\n🔍 Top KB Matches:")
                    for i, match in enumerate(result['matches'][:3], 1):
                        print(f"  {i}. {match['question']} (similarity: {match['similarity_score']:.2%})")
                
                print("=" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                self.print_statistics()
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"\n❌ Error: {e}")
    
    def print_statistics(self):
        """Print usage statistics."""
        print("\n" + "=" * 70)
        print("📈 USAGE STATISTICS")
        print("=" * 70)
        print(f"Total Queries: {self.stats['total_queries']}")
        print(f"Knowledge Base Hits: {self.stats['kb_hits']} ({self.stats['kb_hits']/max(1, self.stats['total_queries'])*100:.1f}%)")
        print(f"Model Hits: {self.stats['model_hits']} ({self.stats['model_hits']/max(1, self.stats['total_queries'])*100:.1f}%)")
        print(f"Hybrid Hits: {self.stats['hybrid_hits']} ({self.stats['hybrid_hits']/max(1, self.stats['total_queries'])*100:.1f}%)")
        print(f"Cached Responses: {len(self.cache)}")
        
        if self.stats['confidence_scores']:
            avg_confidence = np.mean(self.stats['confidence_scores'])
            print(f"Average Confidence: {avg_confidence:.2%}")
        
        elapsed = datetime.now() - self.stats['start_time']
        print(f"Session Duration: {elapsed}")
        print("=" * 70)
    
    def generate_report(self, output_file: str = "atlas_usage_report.json"):
        """Generate detailed usage report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.stats.copy(),
            'cache_size': len(self.cache),
            'knowledge_base_size': len(self.kb_rag.knowledge_base)
        }
        
        # Remove non-serializable datetime
        report['statistics']['start_time'] = report['statistics']['start_time'].isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to {output_file}")


def main():
    """Main entry point."""
    try:
        # Initialize Atlas RAG system
        model_path = input("Enter path to Atlas model (or press Enter for default): ").strip()
        if not model_path:
            model_path = "./atlas_model"  # Default path
        
        atlas = AtlasSystem(model_path)
        
        # Start interactive chat
        atlas.interactive_chat()
        
        # Generate final report
        atlas.generate_report()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")


if __name__ == "__main__":
    main()

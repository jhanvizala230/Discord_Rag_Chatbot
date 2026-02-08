
from sentence_transformers import CrossEncoder
from ..core.config import RERANKER_MODEL
from ..core.logger import setup_logger

logger = setup_logger(__name__)
_rr = None

def get_reranker():
    """Get or initialize the reranker model"""
    global _rr
    if _rr is None:
        logger.info(f"initializing_reranker | model_name={RERANKER_MODEL}")
        _rr = CrossEncoder(RERANKER_MODEL)
        logger.info("reranker_model_loaded")
    return _rr

def rerank(query, candidates):
    """Rerank candidates based on their relevance to the query"""
    logger.info(f"reranking_candidates | query_length={len(query)} | num_candidates={len(candidates)}")
    
    try:
        reranker = get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        scores = reranker.predict(pairs)
        
        for idx, cand in enumerate(candidates):
            score = float(scores[idx]) if idx < len(scores) else 0.0
            cand['rr_score'] = score
            logger.debug(f"candidate_{idx} | rr_score={score:.4f}")
        
        # Sort by reranker score descending
        candidates.sort(key=lambda x: x.get('rr_score', 0), reverse=True)
        
        if scores is not None and len(scores) > 0:
            logger.info(f"reranking_complete | top_score={max(scores):.4f} | bottom_score={min(scores):.4f}")
        
        return candidates
        
    except Exception as e:
        logger.error(f"reranking_failed | error={str(e)}")
        # Fallback: assign zero scores
        for cand in candidates:
            cand['rr_score'] = 0.0
        return candidates

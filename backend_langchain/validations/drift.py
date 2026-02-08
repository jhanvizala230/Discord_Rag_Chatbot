
import numpy as np
from ..vector_and_db.embeddings import embed_texts
from ..vector_and_db.vectorstore import get_collection
from ..core.config import DRIFT_SIM_THRESHOLD
from ..core.logger import setup_logger
from langsmith import traceable

logger = setup_logger(__name__)

@traceable
def detect_drift_for_new_doc(chunks):
    '''
    Simple drift detection using cosine similarity.
    No Evidently required.
    '''
    try:
        # Embed new chunks
        new_emb = embed_texts(chunks)
        
        # Get reference from vectorstore
        col = get_collection()
        all_data = col.get(include=["documents"], limit=200)
        reference_texts = all_data.get("documents", [])
        
        if not reference_texts:
            logger.warning("Drift detection: No reference data available")
            return False, 1.0
        
        # Embed reference
        ref_emb = embed_texts(reference_texts[:200])
        
        # Calculate centroids
        ref_centroid = np.mean(ref_emb, axis=0)
        new_centroid = np.mean(new_emb, axis=0)
        
        # Cosine similarity
        dot_product = np.dot(ref_centroid, new_centroid)
        norm1 = np.linalg.norm(ref_centroid)
        norm2 = np.linalg.norm(new_centroid)
        
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 1.0
        drift_detected = similarity < DRIFT_SIM_THRESHOLD
        
        drift_score = float(similarity)
        logger.info(f"Drift detection result | drifted={drift_detected} | drift_score={drift_score}")
        
        return drift_detected, drift_score
        
    except Exception as e:
        print(f"Drift detection error: {e}")
        return False, 1.0
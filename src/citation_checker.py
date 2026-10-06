"""Local Sentence-Level Citation Checker module.
Implements:
1. Regex extraction of [C1]..[Ck] citations per sentence
2. TF-IDF Cosine Similarity between claim sentence and cited chunk text
3. IDF-weighted medical concept coverage
4. Classification: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, UNCITED
5. Independent IR-based verification with zero LLM API dependency
"""
import re
import math
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

from src.text_processing import split_into_sentences, tokenize_and_stem
from src.config import CITATION_COSINE_SUPPORT_THRESHOLD, CITATION_UNSUPPORTED_THRESHOLD

logger = logging.getLogger(__name__)

CITATION_REGEX = re.compile(r"\[C(\d+)\]")


class SentenceCitationChecker:
    """Verifies that every generated claim sentence is faithful to its cited chunk."""
    
    def __init__(
        self,
        support_threshold: float = CITATION_COSINE_SUPPORT_THRESHOLD,
        unsupported_threshold: float = CITATION_UNSUPPORTED_THRESHOLD
    ):
        self.support_threshold = support_threshold
        self.unsupported_threshold = unsupported_threshold

    def _compute_cosine_and_coverage(
        self,
        sentence_stems: List[str],
        chunk_stems: List[str],
        idf_dict: Dict[str, float]
    ) -> Tuple[float, float, List[str]]:
        """Compute TF-IDF cosine similarity and IDF-weighted term coverage."""
        if not sentence_stems or not chunk_stems:
            return 0.0, 0.0, []
            
        s_tf = Counter(sentence_stems)
        c_tf = Counter(chunk_stems)
        
        # 1. Cosine similarity in TF-IDF space
        # sentence vector
        s_vec = {}
        s_norm_sq = 0.0
        for term, tf in s_tf.items():
            idf = idf_dict.get(term, 1.0)
            w = (1.0 + math.log(tf)) * idf
            s_vec[term] = w
            s_norm_sq += w * w
            
        # chunk vector
        c_vec = {}
        c_norm_sq = 0.0
        for term, tf in c_tf.items():
            idf = idf_dict.get(term, 1.0)
            w = (1.0 + math.log(tf)) * idf
            c_vec[term] = w
            c_norm_sq += w * w
            
        if s_norm_sq == 0.0 or c_norm_sq == 0.0:
            return 0.0, 0.0, []
            
        dot_product = 0.0
        matched_terms = []
        for term, w_s in s_vec.items():
            if term in c_vec:
                dot_product += w_s * c_vec[term]
                matched_terms.append(term)
                
        cosine_sim = dot_product / (math.sqrt(s_norm_sq) * math.sqrt(c_norm_sq))
        
        # 2. IDF-weighted term coverage
        total_s_idf = sum(idf_dict.get(t, 1.0) for t in s_tf)
        matched_s_idf = sum(idf_dict.get(t, 1.0) for t in matched_terms)
        coverage = (matched_s_idf / total_s_idf) if total_s_idf > 0 else 0.0
        
        return float(round(cosine_sim, 4)), float(round(coverage, 4)), matched_terms

    def audit_response(
        self,
        generated_answer: str,
        citation_map: Dict[str, Dict[str, Any]],
        idf_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Perform sentence-by-sentence citation verification of the generated response.
        
        Args:
            generated_answer: Full text of LLM output
            citation_map: Mapping of label ('[C1]', '[C2]') to chunk info
            idf_dict: Optional IDF dictionary from Inverted Index
            
        Returns:
            Detailed verification report
        """
        if idf_dict is None:
            idf_dict = {}
            
        sentences = split_into_sentences(generated_answer)
        if not sentences:
            return {
                "verdict": "EMPTY",
                "faithfulness_score": 0.0,
                "total_sentences": 0,
                "supported_count": 0,
                "unsupported_count": 0,
                "uncited_count": 0,
                "sentence_audits": []
            }
            
        audits = []
        supported_count = 0
        partially_supported_count = 0
        unsupported_count = 0
        uncited_count = 0
        
        for idx, sentence in enumerate(sentences, start=1):
            citations_found = CITATION_REGEX.findall(sentence)
            clean_text = CITATION_REGEX.sub("", sentence).strip()
            s_stems = tokenize_and_stem(clean_text)
            
            if not citations_found:
                # Check if it's a disclaimer or conversational closing
                lower_text = clean_text.lower()
                is_disclaimer = any(d in lower_text for d in [
                    "consult a doctor", "consult your physician", "healthcare professional",
                    "medical advice", "seek emergency", "i don't have enough information"
                ])
                
                if is_disclaimer:
                    verdict = "DISCLAIMER"
                else:
                    verdict = "UNCITED"
                    uncited_count += 1
                    
                audits.append({
                    "sentence_index": idx,
                    "sentence": sentence,
                    "clean_text": clean_text,
                    "citations": [],
                    "verdict": verdict,
                    "confidence_score": 0.0,
                    "cosine_similarity": 0.0,
                    "term_coverage": 0.0,
                    "matched_terms": [],
                    "explanation": "Statement contains no chunk citations." if not is_disclaimer else "General medical disclaimer."
                })
                continue
                
            # Verify against cited chunk(s)
            best_score = 0.0
            best_cosine = 0.0
            best_coverage = 0.0
            best_matched = []
            best_label = ""
            
            for c_num in citations_found:
                label = f"[C{c_num}]"
                chunk_data = citation_map.get(label)
                if not chunk_data:
                    continue
                    
                c_stems = tokenize_and_stem(chunk_data["text"])
                cosine_sim, coverage, matched = self._compute_cosine_and_coverage(s_stems, c_stems, idf_dict)
                combined = 0.65 * cosine_sim + 0.35 * coverage
                
                if combined > best_score:
                    best_score = combined
                    best_cosine = cosine_sim
                    best_coverage = coverage
                    best_matched = matched
                    best_label = label
                    
            # Determine verdict
            if best_score >= self.support_threshold:
                verdict = "SUPPORTED"
                supported_count += 1
                explanation = f"Claim verified against {best_label} (cosine: {best_cosine}, coverage: {best_coverage})."
            elif best_score >= self.unsupported_threshold:
                verdict = "PARTIALLY_SUPPORTED"
                partially_supported_count += 1
                explanation = f"Moderate semantic overlap with {best_label} (cosine: {best_cosine}, coverage: {best_coverage})."
            else:
                verdict = "UNSUPPORTED"
                unsupported_count += 1
                explanation = f"Low agreement with cited {best_label} (score: {best_score:.3f} below threshold {self.unsupported_threshold}). Possible hallucination."
                
            audits.append({
                "sentence_index": idx,
                "sentence": sentence,
                "clean_text": clean_text,
                "citations": [f"[C{c}]" for c in citations_found],
                "primary_citation": best_label,
                "verdict": verdict,
                "confidence_score": round(best_score, 4),
                "cosine_similarity": best_cosine,
                "term_coverage": best_coverage,
                "matched_terms": best_matched,
                "explanation": explanation
            })
            
        n_claims = max(1, len(audits))
        faithfulness = round((supported_count + 0.5 * partially_supported_count) / n_claims, 4)
        
        overall_verdict = (
            "TRUSTWORTHY" if faithfulness >= 0.8
            else "CAUTION" if faithfulness >= 0.5
            else "HIGH_RISK"
        )
        
        return {
            "verdict": overall_verdict,
            "faithfulness_score": faithfulness,
            "total_sentences": len(audits),
            "supported_count": supported_count,
            "partially_supported_count": partially_supported_count,
            "unsupported_count": unsupported_count,
            "uncited_count": uncited_count,
            "sentence_audits": audits
        }

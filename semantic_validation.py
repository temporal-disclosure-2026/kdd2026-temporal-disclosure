# -*- coding: utf-8 -*-
"""
semantic_validation.py

KDD Supplementary Materials - Semantic Anchoring Validation (Appendix B)

This script validates the robustness of keyword-based risk detection by comparing
temporal patterns with semantic similarity anchoring using sentence embeddings.

Key findings (Abstract):
- Keyword-based: 94.1% detection
- + Semantic anchoring: 99.2% detection (164 additional sessions, 5.1%)
- Convergent temporal patterns: median turn = 10 for both methods
- Minimal right-censoring bias in keyword approach

Inputs (in --data_dir):
- session_metadata.csv (must include: session_id, crisis_level, disclosed_keyword, 
                        first_disclosure_turn_keyword, child_utterances)
- child_utterances should be a JSON string or semicolon-separated text

Outputs (to --output_dir):
- table_s7_semantic_validation.xlsx
- semantic_detection_results.csv (session-level comparison)

Method:
- Sentence embeddings: paraphrase-multilingual-mpnet-base-v2
- 10 Korean risk anchor phrases (clinically reviewed)
- Cosine similarity threshold >= 0.65
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from lifelines import KaplanMeierFitter
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
CRISIS_MAPPING = {
    "정상군": "Normal",
    "관찰필요": "Observation",
    "상담필요": "Counseling",
    "학대의심": "Abuse-Suspected",
    "응급": "Emergency",
}

CRISIS_LEVELS = ["Normal", "Observation", "Counseling", "Abuse-Suspected", "Emergency"]

# 10 Korean risk anchor phrases (clinically reviewed)
ANCHOR_PHRASES = [
    "나 자해하고 싶어",  # Self-harm: I want to hurt myself
    "누가 나를 때렸어",  # Abuse: Someone hit me
    "집 나가고 싶어",  # Crisis: I want to run away
    "너무 무서워",  # Emotional distress: I'm really scared
    "우울하고 불안해",  # Mental health: I feel depressed and anxious
    "학교에서 괴롭힘 당해",  # Social isolation: I'm being bullied at school
    "도와주세요",  # Help-seeking: Please help me
    "너무 힘들어 아파",  # Severe distress: It's too hard, I'm in pain
    "죽고 싶어",  # Self-harm: I want to die
    "학대당했어",  # Abuse: I was abused
]

SIMILARITY_THRESHOLD = 0.65
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# -----------------------------
# Utilities
# -----------------------------
def load_session_data(csv_path: Path) -> pd.DataFrame:
    """
    Load session metadata with child utterances.
    
    Required columns:
    - session_id
    - crisis_level
    - disclosed_keyword (bool/0/1)
    - first_disclosure_turn_keyword (int or NaN)
    - total_turns
    - child_utterances (JSON string or semicolon-separated)
    """
    df = pd.read_csv(csv_path)
    
    required = {
        "session_id", "crisis_level", "disclosed_keyword",
        "first_disclosure_turn_keyword", "total_turns", "child_utterances"
    }
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Map crisis levels to English
    df["crisis_eng"] = df["crisis_level"].astype(str).map(CRISIS_MAPPING)
    is_english = df["crisis_level"].isin(CRISIS_LEVELS)
    df.loc[is_english, "crisis_eng"] = df.loc[is_english, "crisis_level"]
    
    if df["crisis_eng"].isna().any():
        raise ValueError("Found unknown crisis_level labels")
    
    # Parse child utterances
    df["utterances_list"] = df["child_utterances"].apply(parse_utterances)
    
    return df


def parse_utterances(text: Any) -> List[str]:
    """
    Parse child utterances from JSON string or semicolon-separated format.
    """
    if pd.isna(text):
        return []
    
    text = str(text).strip()
    if not text:
        return []
    
    # Try JSON first
    if text.startswith("[") or text.startswith("{"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(u).strip() for u in parsed if str(u).strip()]
        except json.JSONDecodeError:
            pass
    
    # Fallback: semicolon-separated
    return [u.strip() for u in text.split(";") if u.strip()]


def compute_semantic_similarity(
    utterances: List[str],
    anchor_embeddings: np.ndarray,
    model: SentenceTransformer,
    threshold: float = SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """
    Compute max similarity of utterances to risk anchor phrases.
    
    Returns:
        {
            "detected": bool,
            "first_turn": int or None,
            "max_similarity": float,
            "detected_utterance": str or None
        }
    """
    if not utterances:
        return {
            "detected": False,
            "first_turn": None,
            "max_similarity": 0.0,
            "detected_utterance": None
        }
    
    # Encode utterances
    utt_embeddings = model.encode(utterances, convert_to_tensor=True, show_progress_bar=False)
    
    # Compute cosine similarities
    similarities = util.cos_sim(utt_embeddings, anchor_embeddings)
    
    # Max similarity per utterance (across all anchors)
    max_sims_per_utt = similarities.max(dim=1).values.cpu().numpy()
    
    # Find first utterance exceeding threshold
    detected_mask = max_sims_per_utt >= threshold
    
    if not detected_mask.any():
        return {
            "detected": False,
            "first_turn": None,
            "max_similarity": float(max_sims_per_utt.max()),
            "detected_utterance": None
        }
    
    first_idx = int(np.argmax(detected_mask))
    
    return {
        "detected": True,
        "first_turn": first_idx + 1,  # 1-indexed
        "max_similarity": float(max_sims_per_utt[first_idx]),
        "detected_utterance": utterances[first_idx]
    }


def compute_msw(df_level: pd.DataFrame, turn_col: str) -> float:
    """
    Compute Minimum Safety Window (turn where 50% disclosed).
    """
    if len(df_level) == 0:
        return np.nan
    
    df = df_level.copy()
    df["event"] = df[turn_col].notna()
    df["duration"] = df[turn_col].fillna(df["total_turns"]).clip(lower=1)
    
    if df["event"].sum() == 0:
        return np.nan
    
    kmf = KaplanMeierFitter()
    kmf.fit(df["duration"], event_observed=df["event"])
    
    sf = kmf.survival_function_
    col = sf.columns[0]
    mask = sf[col] <= 0.5
    
    if not mask.any():
        return np.nan
    
    return float(sf.index[mask].min())


# -----------------------------
# Main Analysis
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--metadata", type=str, default="session_metadata.csv")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[1/5] Loading session data from {args.metadata}...")
    df = load_session_data(data_dir / args.metadata)
    print(f"      Loaded {len(df):,} sessions")
    
    print(f"\n[2/5] Loading sentence transformer model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"[3/5] Encoding {len(ANCHOR_PHRASES)} risk anchor phrases...")
    anchor_embeddings = model.encode(
        ANCHOR_PHRASES,
        convert_to_tensor=True,
        show_progress_bar=False
    )
    
    print(f"\n[4/5] Computing semantic similarity for {len(df):,} sessions...")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing sessions"):
        result = compute_semantic_similarity(
            row["utterances_list"],
            anchor_embeddings,
            model,
            threshold=SIMILARITY_THRESHOLD
        )
        results.append({
            "session_id": row["session_id"],
            "detected_semantic": result["detected"],
            "first_turn_semantic": result["first_turn"],
            "max_similarity": result["max_similarity"],
        })
    
    df_semantic = pd.DataFrame(results)
    df_combined = df.merge(df_semantic, on="session_id")
    
    # Compare detection methods
    df_combined["detected_keyword"] = df_combined["disclosed_keyword"].fillna(0).astype(bool)
    df_combined["detected_both"] = (
        df_combined["detected_keyword"] & df_combined["detected_semantic"]
    )
    df_combined["keyword_only"] = (
        df_combined["detected_keyword"] & ~df_combined["detected_semantic"]
    )
    df_combined["semantic_only"] = (
        ~df_combined["detected_keyword"] & df_combined["detected_semantic"]
    )
    df_combined["union"] = (
        df_combined["detected_keyword"] | df_combined["detected_semantic"]
    )
    
    # Save session-level results
    output_cols = [
        "session_id", "crisis_eng", "detected_keyword", "detected_semantic",
        "detected_both", "keyword_only", "semantic_only", "union",
        "first_disclosure_turn_keyword", "first_turn_semantic", "max_similarity"
    ]
    df_combined[output_cols].to_csv(
        out_dir / "semantic_detection_results.csv",
        index=False
    )
    
    print(f"\n[5/5] Generating Table S7...")
    
    # Overall statistics
    total = len(df_combined)
    keyword_total = df_combined["detected_keyword"].sum()
    semantic_total = df_combined["detected_semantic"].sum()
    both_total = df_combined["detected_both"].sum()
    keyword_only_total = df_combined["keyword_only"].sum()
    semantic_only_total = df_combined["semantic_only"].sum()
    union_total = df_combined["union"].sum()
    
    print(f"\n=== Overall Detection Statistics ===")
    print(f"Total sessions: {total:,}")
    print(f"Keyword-based: {keyword_total:,} ({keyword_total/total*100:.1f}%)")
    print(f"Semantic anchoring: {semantic_total:,} ({semantic_total/total*100:.1f}%)")
    print(f"Both methods: {both_total:,} ({both_total/total*100:.1f}%)")
    print(f"Keyword only: {keyword_only_total:,} ({keyword_only_total/total*100:.1f}%)")
    print(f"Semantic only: {semantic_only_total:,} ({semantic_only_total/total*100:.1f}%)")
    print(f"Union (Total): {union_total:,} ({union_total/total*100:.1f}%)")
    
    # Crisis-stratified Table S7
    s7_rows = []
    
    for level in CRISIS_LEVELS:
        sub = df_combined[df_combined["crisis_eng"] == level].copy()
        if len(sub) == 0:
            continue
        
        n = len(sub)
        keyword = sub["detected_keyword"].sum()
        semantic = sub["detected_semantic"].sum()
        both = sub["detected_both"].sum()
        keyword_only = sub["keyword_only"].sum()
        semantic_only = sub["semantic_only"].sum()
        union = sub["union"].sum()
        
        # Compute MSW for both methods
        msw_keyword = compute_msw(sub, "first_disclosure_turn_keyword")
        msw_semantic = compute_msw(sub, "first_turn_semantic")
        
        s7_rows.append({
            "Crisis Level": level,
            "N": n,
            "Keyword Only": keyword_only,
            "Semantic Only": semantic_only,
            "Both Methods": both,
            "Union (Total)": union,
            "Union %": round(union / n * 100, 1),
            "Median Turn (Keyword)": round(msw_keyword, 1) if not np.isnan(msw_keyword) else "—",
            "Median Turn (Semantic)": round(msw_semantic, 1) if not np.isnan(msw_semantic) else "—",
        })
    
    # Add overall row
    msw_keyword_overall = compute_msw(df_combined, "first_disclosure_turn_keyword")
    msw_semantic_overall = compute_msw(df_combined, "first_turn_semantic")
    
    s7_rows.append({
        "Crisis Level": "Total",
        "N": total,
        "Keyword Only": keyword_only_total,
        "Semantic Only": semantic_only_total,
        "Both Methods": both_total,
        "Union (Total)": union_total,
        "Union %": round(union_total / total * 100, 1),
        "Median Turn (Keyword)": round(msw_keyword_overall, 1) if not np.isnan(msw_keyword_overall) else "—",
        "Median Turn (Semantic)": round(msw_semantic_overall, 1) if not np.isnan(msw_semantic_overall) else "—",
    })
    
    df_s7 = pd.DataFrame(s7_rows)
    
    # Save Table S7
    s7_path = out_dir / "table_s7_semantic_validation.xlsx"
    df_s7.to_excel(s7_path, index=False)
    
    print(f"\n[OK] Saved outputs to: {out_dir}")
    print(f" - semantic_detection_results.csv")
    print(f" - {s7_path.name}")
    
    print("\n=== Table S7 Preview ===")
    print(df_s7.to_string(index=False))
    
    print("\n[VALIDATION] Convergent patterns:")
    print(f"  Median turn (Keyword): {msw_keyword_overall:.1f}")
    print(f"  Median turn (Semantic): {msw_semantic_overall:.1f}")
    print(f"  Additional detection (Semantic only): {semantic_only_total} ({semantic_only_total/total*100:.1f}%)")
    print("\n✅ Robustness validation complete!")


if __name__ == "__main__":
    main()

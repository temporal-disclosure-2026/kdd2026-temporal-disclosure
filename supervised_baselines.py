"""
Supervised Baseline Experiments for KDD 2026 Paper
Purpose: Session-level binary classification (has_disclosure = 0/1)
Method: Stratified 5-fold cross-validation
Features: Crisis severity + Total turns
Models: Logistic Regression, SVM, Random Forest, MLP
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SUPERVISED BASELINE EXPERIMENTS - 5-FOLD CROSS-VALIDATION")
print("="*80)

# ============================================
# 1. Load and Prepare Data
# ============================================

df = pd.read_csv('/mnt/user-data/uploads/soft_anchor_results_full.csv')

# Crisis mapping
CRISIS_MAPPING = {
    "정상군": "Normal",
    "관찰필요": "Observation",
    "상담필요": "Counseling",
    "학대의심": "Abuse-Suspected",
    "응급": "Emergency",
}
df['crisis_eng'] = df['crisis'].map(CRISIS_MAPPING)

# Create target variables
df['keyword_detected'] = df['first_hard_risk_turn'].notna().astype(int)
df['semantic_detected'] = df['first_soft_risk_turn'].notna().astype(int)
df['union_detected'] = ((df['keyword_detected'] == 1) | (df['semantic_detected'] == 1)).astype(int)

# One-hot encode crisis
crisis_dummies = pd.get_dummies(df['crisis_eng'], prefix='crisis', dtype=float)

# Features: total_turns + crisis_level (one-hot)
feature_df = pd.concat([df[['total_turns']], crisis_dummies], axis=1)
X = feature_df.values.astype(float)

print(f"\nDataset: {len(df):,} sessions")
print(f"  Keyword detected: {df['keyword_detected'].sum():,} ({df['keyword_detected'].mean()*100:.1f}%)")
print(f"  Semantic detected: {df['semantic_detected'].sum():,} ({df['semantic_detected'].mean()*100:.1f}%)")
print(f"  Union detected: {df['union_detected'].sum():,} ({df['union_detected'].mean()*100:.1f}%)")
print(f"\nFeatures: {X.shape[1]} (total_turns + {len(crisis_dummies.columns)} crisis categories)")

# ============================================
# 2. Define Models
# ============================================

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42, early_stopping=True)
}

# ============================================
# 3. Experiment 1: Keyword Detection (94.1%)
# ============================================

print("\n" + "="*80)
print("EXPERIMENT 1: PREDICT KEYWORD DETECTION (Session Features Only)")
print("="*80)

y_keyword = df['keyword_detected'].values

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_keyword = []

for model_name, model_template in models.items():
    print(f"\nTraining {model_name}...")
    
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_keyword), 1):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_keyword[train_idx], y_keyword[val_idx]
        
        # Scale features (fit on train, transform on val)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Clone and train model
        from sklearn.base import clone
        model = clone(model_template)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrics
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob) if y_prob is not None and len(np.unique(y_val)) > 1 else np.nan
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['f1'].append(f1)
        fold_metrics['auc'].append(auc)
        
        print(f"  Fold {fold_idx}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    # Compute mean ± std
    results_keyword.append({
        'Model': model_name,
        'Accuracy': f"{np.mean(fold_metrics['accuracy']):.3f} ± {np.std(fold_metrics['accuracy']):.3f}",
        'Precision': f"{np.mean(fold_metrics['precision']):.3f} ± {np.std(fold_metrics['precision']):.3f}",
        'Recall': f"{np.mean(fold_metrics['recall']):.3f} ± {np.std(fold_metrics['recall']):.3f}",
        'F1': f"{np.mean(fold_metrics['f1']):.3f} ± {np.std(fold_metrics['f1']):.3f}",
        'AUC-ROC': f"{np.mean(fold_metrics['auc']):.3f} ± {np.std(fold_metrics['auc']):.3f}",
        'Accuracy_mean': np.mean(fold_metrics['accuracy']),
        'Accuracy_std': np.std(fold_metrics['accuracy']),
        'Precision_mean': np.mean(fold_metrics['precision']),
        'Precision_std': np.std(fold_metrics['precision']),
        'Recall_mean': np.mean(fold_metrics['recall']),
        'Recall_std': np.std(fold_metrics['recall']),
        'F1_mean': np.mean(fold_metrics['f1']),
        'F1_std': np.std(fold_metrics['f1']),
        'AUC_mean': np.mean(fold_metrics['auc']),
        'AUC_std': np.std(fold_metrics['auc'])
    })

# Baseline comparison
keyword_baseline = df['keyword_detected'].mean()
print(f"\n** Keyword-based Baseline: {keyword_baseline:.3f} (94.1%)")

# ============================================
# 4. Experiment 2: Union Detection (99.1%)
# ============================================

print("\n" + "="*80)
print("EXPERIMENT 2: PREDICT UNION DETECTION (Keyword + Semantic)")
print("="*80)

y_union = df['union_detected'].values

results_union = []

for model_name, model_template in models.items():
    print(f"\nTraining {model_name}...")
    
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': []
    }
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_union), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_union[train_idx], y_union[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        from sklearn.base import clone
        model = clone(model_template)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_val_scaled)
        y_prob = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob) if y_prob is not None and len(np.unique(y_val)) > 1 else np.nan
        
        fold_metrics['accuracy'].append(acc)
        fold_metrics['precision'].append(prec)
        fold_metrics['recall'].append(rec)
        fold_metrics['f1'].append(f1)
        fold_metrics['auc'].append(auc)
        
        print(f"  Fold {fold_idx}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    results_union.append({
        'Model': model_name,
        'Accuracy': f"{np.mean(fold_metrics['accuracy']):.3f} ± {np.std(fold_metrics['accuracy']):.3f}",
        'Precision': f"{np.mean(fold_metrics['precision']):.3f} ± {np.std(fold_metrics['precision']):.3f}",
        'Recall': f"{np.mean(fold_metrics['recall']):.3f} ± {np.std(fold_metrics['recall']):.3f}",
        'F1': f"{np.mean(fold_metrics['f1']):.3f} ± {np.std(fold_metrics['f1']):.3f}",
        'AUC-ROC': f"{np.mean(fold_metrics['auc']):.3f} ± {np.std(fold_metrics['auc']):.3f}",
        'Accuracy_mean': np.mean(fold_metrics['accuracy']),
        'Accuracy_std': np.std(fold_metrics['accuracy']),
        'Precision_mean': np.mean(fold_metrics['precision']),
        'Precision_std': np.std(fold_metrics['precision']),
        'Recall_mean': np.mean(fold_metrics['recall']),
        'Recall_std': np.std(fold_metrics['recall']),
        'F1_mean': np.mean(fold_metrics['f1']),
        'F1_std': np.std(fold_metrics['f1']),
        'AUC_mean': np.mean(fold_metrics['auc']),
        'AUC_std': np.std(fold_metrics['auc'])
    })

union_baseline = df['union_detected'].mean()
print(f"\n** Union Baseline: {union_baseline:.3f} (99.1%)")

# ============================================
# 5. Save Results
# ============================================

df_keyword = pd.DataFrame(results_keyword)
df_union = pd.DataFrame(results_union)

# Create summary tables for paper (formatted strings)
df_keyword_display = df_keyword[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']]
df_union_display = df_union[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']]

# Create detailed tables for appendix (separate mean/std columns)
df_keyword_detailed = df_keyword[['Model', 'Accuracy_mean', 'Accuracy_std', 
                                   'Precision_mean', 'Precision_std',
                                   'Recall_mean', 'Recall_std',
                                   'F1_mean', 'F1_std',
                                   'AUC_mean', 'AUC_std']]

df_union_detailed = df_union[['Model', 'Accuracy_mean', 'Accuracy_std',
                               'Precision_mean', 'Precision_std',
                               'Recall_mean', 'Recall_std',
                               'F1_mean', 'F1_std',
                               'AUC_mean', 'AUC_std']]

# Save to Excel
output_path = '/mnt/user-data/outputs/table_s4_supervised_baselines.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_keyword_display.to_excel(writer, sheet_name='Keyword_Summary', index=False)
    df_union_display.to_excel(writer, sheet_name='Union_Summary', index=False)
    df_keyword_detailed.to_excel(writer, sheet_name='Keyword_Detailed', index=False)
    df_union_detailed.to_excel(writer, sheet_name='Union_Detailed', index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\n=== TABLE S4a: Keyword Detection (5-Fold CV) ===")
print(df_keyword_display.to_string(index=False))

print("\n=== TABLE S4b: Union Detection (5-Fold CV) ===")
print(df_union_display.to_string(index=False))

print(f"\n✅ Saved: {output_path}")
print("\nSheets included:")
print("  1. Keyword_Summary (formatted: mean ± std)")
print("  2. Union_Summary (formatted: mean ± std)")
print("  3. Keyword_Detailed (separate mean/std columns)")
print("  4. Union_Detailed (separate mean/std columns)")

print("\n" + "="*80)
print("EXPERIMENT COMPLETED")
print("="*80)

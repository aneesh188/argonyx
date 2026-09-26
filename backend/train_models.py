import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
import joblib
import os

def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found. Run generate_dataset.py first.")
        
    df = pd.read_csv(file_path)
    
    # 1. Handle missing values
    # In case there are missing values from custom retraining, we fill optional blood/hormone features
    # with their median, and symptoms with 0 (not present).
    optional_cols = ['FSH', 'LH', 'AMH', 'Testosterone', 'FastingInsulin', 'BloodGlucose', 'SleepDuration', 'StressLevel']
    for col in optional_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    # Impute other columns just in case
    df = df.fillna(df.median())
    
    # 2. Outlier Detection and Removal (using IQR method on clinical continuous indicators)
    # We apply it on Weight, BMI, FSH, LH, AMH, Testosterone to remove extreme noise,
    # but not too aggressively so we keep valid high-PCOS markers.
    outlier_cols = ['BMI', 'AMH', 'Testosterone', 'FastingInsulin']
    mask = pd.Series(True, index=df.index)
    for col in outlier_cols:
        Q1 = df[col].quantile(0.01) # Keep 98% range to avoid deleting useful high risk flags
        Q3 = df[col].quantile(0.99)
        IQR = Q3 - Q1
        # outlier boundaries
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        mask = mask & (df[col] >= lower) & (df[col] <= upper)
        
    df_clean = df[mask].copy()
    print(f"Cleaned dataset: Removed {len(df) - len(df_clean)} outliers. Remaining samples: {len(df_clean)}")
    
    # Split into features (X) and target (y)
    X = df_clean.drop(columns=['PCOS'])
    y = df_clean['PCOS']
    
    return X, y

def train_and_evaluate(X, y):
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Identify continuous columns (all except binary symptoms and categories)
    # Binary/Categorical indicators: CycleRegularity, Hirsutism, Acne, HairLoss, DarkPatches, WeightGain,
    # DifficultyLosingWeight, Fatigue, MoodSwings, PelvicPain, Infertility, FamilyHistory, PhysicalActivity
    continuous_cols = ['Age', 'Height', 'Weight', 'BMI', 'CycleLength', 'CyclePeriod', 'StressLevel', 'SleepDuration', 
                       'FSH', 'LH', 'AMH', 'Testosterone', 'FastingInsulin', 'BloodGlucose']
    
    # Scale only the continuous features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    # Fit scaler on training continuous columns
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    # Save the scaler
    os.makedirs('backend', exist_ok=True)
    joblib.dump(scaler, 'backend/scaler.joblib')
    print("Scaler saved to backend/scaler.joblib")
    
    # Define models
    models = {
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='logloss', random_state=42)
    
    results = {}
    best_model_name = None
    best_model_score = -1.0
    best_model = None
    
    print("\n--- Model Evaluation Results ---")
    for name, model in models.items():
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-score': f1,
            'ROC-AUC': auc
        }
        
        print(f"\nModel: {name}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {auc:.4f}")
        
        # We select the best model based on F1-score (harmonic mean of precision and recall)
        if f1 > best_model_score:
            best_model_score = f1
            best_model_name = name
            best_model = model
            
    print(f"\n>>> Best Performing Model: {best_model_name} (F1-score: {best_model_score:.4f})")
    
    # Save the best model
    joblib.dump(best_model, 'backend/best_model.joblib')
    print(f"Best model saved to backend/best_model.joblib")
    
    # Save a text file detailing performance comparison
    with open('backend/model_comparison.txt', 'w') as f:
        f.write("Model Performance Comparison\n")
        f.write("============================\n")
        for name, metrics in results.items():
            f.write(f"\nModel: {name}\n")
            for metric, val in metrics.items():
                f.write(f"  {metric}: {val:.4f}\n")
        f.write(f"\nBest Model: {best_model_name} (F1-score: {best_model_score:.4f})\n")
        
    return best_model_name, results

if __name__ == "__main__":
    X, y = load_and_preprocess_data('backend/pcod_dataset.csv')
    train_and_evaluate(X, y)

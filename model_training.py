import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle

def heuristic_labeling(df):
    """
    Since we don't have labeled data, we will use a heuristic approach to label clips 
    as High Impact (+1) or Low Impact (-1).
    High Impact = High Motion + High Object Density
    """
    # Normalize features for heuristic
    motion_norm = (df['motion_intensity'] - df['motion_intensity'].min()) / (df['motion_intensity'].max() - df['motion_intensity'].min() + 1e-6)
    object_norm = (df['object_density'] - df['object_density'].min()) / (df['object_density'].max() - df['object_density'].min() + 1e-6)
    
    # Combined Impact Score
    impact_score = motion_norm + object_norm
    
    # Label top 40% as High Impact (+1), rest as Low Impact (-1)
    threshold = impact_score.quantile(0.60)
    
    df['heuristic_score'] = impact_score
    df['label'] = np.where(impact_score >= threshold, 1, -1)
    
    return df

def main():
    features_csv = r"d:\Semester 4\AI LAB\OEL\features.csv"
    model_out = r"d:\Semester 4\AI LAB\OEL\impact_model.pkl"
    scaler_out = r"d:\Semester 4\AI LAB\OEL\scaler.pkl"
    
    print(f"Loading features from {features_csv}...")
    df = pd.read_csv(features_csv)
    
    print("Applying heuristic labeling...")
    df = heuristic_labeling(df)
    
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    labeled_csv = r"d:\Semester 4\AI LAB\OEL\labeled_features.csv"
    df.to_csv(labeled_csv, index=False)
    print(f"Labeled features saved to {labeled_csv}")
    
    # Prepare data
    feature_cols = ['motion_intensity', 'object_density'] + [col for col in df.columns if col.startswith('emb_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression Model
    print("Training Logistic Regression model...")
    model = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # Save Model and Scaler
    with open(model_out, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_out, 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"\nModel saved to {model_out}")
    print(f"Scaler saved to {scaler_out}")

    # Display the top 5 high-impact clips for trailer generation
    high_impact = df[df['label'] == 1].sort_values(by='heuristic_score', ascending=False)
    print("\nTop 5 High-Impact Clips (Suggested for Trailer):")
    print(high_impact[['clip_name', 'heuristic_score']].head(5))

if __name__ == '__main__':
    main()

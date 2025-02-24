import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score  # FIXED IMPORT
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmailRiskAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.features = None
        self.data = None

    def load_and_preprocess_data(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} records from {file_path}")
            self.data.fillna(0, inplace=True)
            
            # Encode target variable first
            self.data['risk_score'] = self.label_encoder.fit_transform(self.data['risk_score'])
            return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return False

    def prepare_model_data(self):
        self.features = [
            "warning_word_score", "exclamation_score", "spam_word_score",
            "warning_words_found", "spam_words_found", "exclamation_patterns",
            "risk_label"
        ]
        
        X = self.data[self.features].copy()
        y = self.data['risk_score']

        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                logger.info(f"Encoding non-numeric column: {col}")
                X[col] = X[col].astype(str).astype('category').cat.codes

        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train_model(self):
        try:
            X, y = self.prepare_model_data()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y
            )
            
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X, y, cv=cv)  # Now properly imported
            
            logger.info(f"Cross-validation scores: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.3f}")
            
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'features': self.features
            }, 'xgboost_model.pkl')
            
            return True
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False

# Rest of the main() function remains unchanged...

def main():
    analyzer = EmailRiskAnalyzer()
    file_path = "/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/enhanced_risk_assessment_results.csv"
    if not analyzer.load_and_preprocess_data(file_path):
        return
    if not analyzer.train_model():
        return
    
    while True:
        print("\nEmail Risk Analysis Menu:")
        print("1. Analyze sender")
        print("2. Analyze specific email by index")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            email = input("Enter sender email address: ")
            analysis, status = analyzer.analyze_sender(email)
            
            if status == "success":
                print("\nAnalysis Results:")
                print(f"Total Emails: {analysis['total_emails']}")
                print(f"Risk Metrics:")
                print(f"- Average Risk Score: {analysis['risk_metrics']['average_risk']:.2f}")
                print(f"- Maximum Risk Score: {analysis['risk_metrics']['max_risk']:.2f}")
                print(f"- Risk Trend: {analysis['risk_metrics']['risk_trend']}")
                print(f"- Most Recent Risk Score: {analysis['risk_metrics']['recent_risk']:.2f}")
                print("\nTime Patterns:")
                print(f"- Common Hours: {analysis['time_patterns']['common_hours']}")
                print(f"- Weekend Activity: {analysis['time_patterns']['weekend_activity']:.1f}%")
                print("\nContent Analysis:")
                print("- Common Spam Words:", ', '.join(analysis['content_analysis']['common_spam_words']))
                print("- Common Warning Words:", ', '.join(analysis['content_analysis']['common_warning_words']))
                
            elif status == "partial_match":
                print("\nFound partial matches. Showing all related emails:")
                print(analysis[['from', 'subject', 'date', 'risk_score']])
                
            else:
                print(f"\nCould not analyze email: {status}")
                
        elif choice == '2':
            print(f"\nValid index range: 0 to {len(analyzer.data) - 1}")
            index = input("Enter email index to analyze: ")
            analysis, status = analyzer.analyze_email_by_index(index)
            
            if status == "success":
                analyzer.print_email_analysis(analysis)
            else:
                print(f"\nCould not analyze email: {status}")
                
        elif choice == '3':
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
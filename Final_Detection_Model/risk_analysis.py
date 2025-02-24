import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
import language_tool_python

class RiskAssessment:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.tool = language_tool_python.LanguageTool('en-US')
        
        # Suspicious phrases list
        self.suspicious_phrases = [
            "urgent", "win", "cash bonus", "credit card offers",
            "limited time", "offer expires", "act now", "cure",
            "weight loss", "click here", "free trial", "unsubscribe",
            "bank account", "lottery", "inheritance", "password reset",
            "confirm your identity", "access your account", "verify account",
            "guarantee", "free", "100%", "prize", "winner", "congratulations",
            "exclusive offer", "best price", "discount", "money back",
            "risk free", "no risk", "lifetime", "incredible deal"
        ]

        # Warning words in caps
        self.warning_words = [
            "WARNING", "CAUTION", "URGENT", "IMPORTANT", "ALERT",
            "ATTENTION", "NOTICE", "DANGER", "CRITICAL", "EMERGENCY",
            "IMMEDIATE", "ACTION", "REQUIRED", "DEADLINE", "FINAL"
        ]

    def analyze_grammar(self, text):
        """Analyze grammar using language-tool-python."""
        if not text.strip():
            return {"grammar_errors": 0, "grammar_score": 100}
        
        try:
            matches = self.tool.check(text)
            error_count = len(matches)
            score = max(0, 100 - error_count * 2)  # Deduct 2 points per error
            return {
                "grammar_errors": error_count,
                "grammar_score": score
            }
        except:
            return {"grammar_errors": 0, "grammar_score": 100}

    def check_suspicious_content(self, text):
        """Check for suspicious keywords."""
        matches = [phrase for phrase in self.suspicious_phrases 
                  if re.search(rf"\b{phrase}\b", text, re.IGNORECASE)]
        return {
            "matched_phrases": matches,
            "is_suspicious": len(matches) > 0,
            "spam_word_count": len(matches)
        }

    def check_warning_words(self, text):
        """Check for warning words in capital letters."""
        matches = [word for word in self.warning_words if word in text]
        return {
            "matched_warnings": matches,
            "warning_word_count": len(matches)
        }

    def analyze_exclamation_marks(self, text):
        """Analyze exclamation patterns."""
        exclamation_patterns = re.findall(r'!{2,}', text)
        return {
            "multiple_exclamations": len(exclamation_patterns),
            "exclamation_sequences": exclamation_patterns
        }

    def calculate_risk_score(self, text):
        """Calculate comprehensive risk score."""
        suspicious_content = self.check_suspicious_content(text)
        warning_words = self.check_warning_words(text)
        exclamation_analysis = self.analyze_exclamation_marks(text)

        warning_score = min(30, warning_words["warning_word_count"] * 6)
        exclamation_score = min(20, exclamation_analysis["multiple_exclamations"] * 5)
        spam_word_score = min(50, suspicious_content["spam_word_count"] * 10)

        total_score = min(100, warning_score + exclamation_score + spam_word_score)

        return {
            "total_risk_score": total_score,
            "component_scores": {
                "warning_word_score": warning_score,
                "exclamation_score": exclamation_score,
                "spam_word_score": spam_word_score
            },
            "details": {
                "warning_words_found": warning_words["matched_warnings"],
                "exclamation_sequences": exclamation_analysis["exclamation_sequences"],
                "spam_words_found": suspicious_content["matched_phrases"]
            }
        }

    def determine_risk_label(self, score):
        """Determine risk category."""
        if score <= 30:
            return "Low risk"
        elif 31 <= score <= 60:
            return "Medium risk"
        else:
            return "High risk (needs attention)"

    def process_json_dataset(self, json_file_path, output_file_path):
        """Process JSON data and generate CSV output."""
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        emails = data['emails']
        processed_emails = []
        
        for email in emails:
            risk_analysis = self.calculate_risk_score(email['content'])
            grammar_analysis = self.analyze_grammar(email['content'])

            processed_email = {
                'id': email['id'],
                'from': email['from'],
                'subject': email['subject'],
                'date': email['date'],
                'content': email['content'],
                'risk_score': risk_analysis['total_risk_score'],
                'warning_word_score': risk_analysis['component_scores']['warning_word_score'],
                'exclamation_score': risk_analysis['component_scores']['exclamation_score'],
                'spam_word_score': risk_analysis['component_scores']['spam_word_score'],
                'grammar_score': grammar_analysis['grammar_score'],
                'warning_words_found': ', '.join(risk_analysis['details']['warning_words_found']),
                'spam_words_found': ', '.join(risk_analysis['details']['spam_words_found']),
                'exclamation_patterns': ', '.join(risk_analysis['details']['exclamation_sequences'])
            }
            
            processed_email['risk_label'] = self.determine_risk_label(processed_email['risk_score'])
            processed_emails.append(processed_email)

        df = pd.DataFrame(processed_emails)
        df.to_csv(output_file_path, index=False)

        summary = {
            'total_emails': len(processed_emails),
            'risk_distribution': df['risk_label'].value_counts().to_dict(),
            'average_risk_score': df['risk_score'].mean(),
            'average_grammar_score': df['grammar_score'].mean(),
            'component_averages': {
                'warning_word_score': df['warning_word_score'].mean(),
                'exclamation_score': df['exclamation_score'].mean(),
                'spam_word_score': df['spam_word_score'].mean(),
                'grammar_score': df['grammar_score'].mean()
            },
            'high_risk_emails': df[df['risk_label'] == 'High risk (needs attention)']['id'].tolist()
        }

        return df, summary

    def avg_risk_score(self, df):
        """Calculate average risk score."""
        return df['risk_score'].mean()
    
    def avg_grammar_score(self, df):
        """Calculate average grammar score."""
        return df['grammar_score'].mean()

def main():
    try:
        risk_assessment = RiskAssessment()
        json_file_path = r"/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/chirag_dataset.json"
        output_file_path = r"/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/enhanced_risk_assessment_results.csv"
        
        processed_df, summary = risk_assessment.process_json_dataset(json_file_path, output_file_path)
        
        # Save the summary to a JSON file
        summary_file_path = r"/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/risk_summary.json"
        with open(summary_file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_file_path}")
        
        print("\nEnhanced Risk Assessment Summary:")
        print(f"Total emails processed: {summary['total_emails']}")
        print("\nRisk Distribution:")
        for risk_level, count in summary['risk_distribution'].items():
            print(f"{risk_level}: {count}")
        print(f"\nAverage Risk Score: {summary['average_risk_score']:.2f}")
        print(f"Average Grammar Score: {summary['average_grammar_score']:.2f}")
        print("\nComponent Score Averages:")
        for component, score in summary['component_averages'].items():
            print(f"{component}: {score:.2f}")
        
        if summary['high_risk_emails']:
            print(f"\nNumber of High Risk Emails: {len(summary['high_risk_emails'])}")
            print("High Risk Email IDs:", summary['high_risk_emails'])

        print(f"\nDetailed results saved to: {output_file_path}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
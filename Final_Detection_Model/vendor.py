import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import language_tool_python
import logging
import re
from email.utils import parseaddr
import csv
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VendorCommunicationAnalyzer:
    def __init__(self, company_domain):
        self.company_domain = company_domain
        self.scaler = StandardScaler()
        self.grammar_tool = language_tool_python.LanguageTool('en-US')
        self.vader = SentimentIntensityAnalyzer()
        self.spam_words = [
            "Urgent", "Immediately", "Confidential", "Investment", "Opportunity", "Secure", 
            "Guaranteed", "Jackpot", "Win", "Bonus", "Credit offer", "Expires soon", 
            "Limited access", "Act fast", "Remedy", "Slim down", "Tap here", "Complimentary test", 
            "Opt-out", "Financial details", "Sweepstakes", "Bequest", "Security update", 
            "Identity verification", "Login required", "Authenticate", "Certified", "Gratis", 
            "Complete refund", "Champion", "Bravo", "Special deal", "Lowest rate", "Rebate", 
            "Cashback", "Zero danger", "Permanent", "Unbelievable offer"
        ]
        self.url_pattern = re.compile(
            r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
            r'www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|'
            r'https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|'
            r'www\.[a-zA-Z0-9]+\.[^\s]{2,})'
        )

    def is_company_employee(self, email):
        """Check if email belongs to company domain"""
        return self.company_domain in self.parse_email(email)

    def parse_email(self, email):
        """Extract email address from formatted strings"""
        _, addr = parseaddr(email)
        return addr.lower() if addr else email.lower()

    def analyze_sentiment(self, text):
        """Analyze sentiment using VADER and TextBlob"""
        vader_scores = self.vader.polarity_scores(text)
        blob = TextBlob(text)
        
        compound = vader_scores['compound']
        if compound >= 0.05:
            sentiment_category = "Positive"
        elif compound <= -0.05:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"
            
        return {
            "vader_scores": vader_scores,
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "sentiment_category": sentiment_category
        }

    def is_url(self, text):
        """Check if the given text is a URL"""
        return bool(self.url_pattern.match(text))

    def has_excess_caps(self, word):
        """
        Check if a single word has more than 2 capital letters.
        Returns True if the word has more than 2 capital letters, False otherwise.
        """
        # Skip words that are completely uppercase
        if word.isupper():
            return False
            
        # Count capital letters in this single word
        capital_count = sum(1 for char in word if char.isupper())
        return capital_count > 2

    def calculate_risk_score(self, text):
        """Calculate risk score for the given text"""
        # Split text into individual words and remove empty strings
        words = [word.strip() for word in text.split() if word.strip()]
    
        # Count words with excess capitalization
        excess_caps_count = 0
        for word in words:
            # Skip URLs
            if self.is_url(word):
                continue
                
            # Check this individual word for excess capitals
            if self.has_excess_caps(word):
                excess_caps_count += 1
    
        # Count multiple exclamation marks
        multiple_exclamations = len(re.findall(r'!{2,}', text))
    
        # Count spam words
        spam_words_count = sum(1 for word in self.spam_words if word.lower() in text.lower())
    
        # Calculate component scores
        excess_caps_score = min(40, excess_caps_count * 1)  # 1 point per word with excess caps
        exclamation_score = min(30, multiple_exclamations * 15)  # Cap exclamation score at 30
        spam_word_score = min(30, spam_words_count * 5)  # Cap spam word score at 30
    
        total_score = min(100, excess_caps_score + exclamation_score + spam_word_score)
    
        return {
            "total_risk_score": total_score,
            "components": {
                "excess_caps_score": excess_caps_score,
                "exclamation_score": exclamation_score,
                "spam_word_score": spam_word_score
            },
            "details": {
                "excess_caps_count": excess_caps_count,
                "multiple_exclamations": multiple_exclamations,
                "spam_words_found": spam_words_count
            }
        }

    def analyze_thread(self, messages):
        """Analyze a communication thread with validation"""
        if not messages:
            return None
            
        # Combine all message content
        combined_text = ' '.join([msg['content'] for msg in messages])
        
        # Get sender of first message
        first_sender = messages[0]['from']
        sender_email = self.parse_email(first_sender)
        
        # Perform sentiment analysis
        sentiment = self.analyze_sentiment(combined_text)
        
        # Calculate risk scores
        risk_analysis = self.calculate_risk_score(combined_text)
        
        result = {
            'thread_id': messages[0]['threadId'],
            'sender_email': sender_email,
            'sender_name': sender_email.split('@')[0],
            'sentiment_score': sentiment['vader_scores']['compound'],
            'grammar_errors': len(self.grammar_tool.check(combined_text)),
            'message_count': len(messages),
            'first_message_date': messages[0]['date'],
            'subject': messages[0]['subject'],
            **risk_analysis
        }
        
        # Determine status based on risk score
        result['status'] = 'suspicious' if risk_analysis['total_risk_score'] > 50 else 'approved'
        
        return result

    def process_dataset(self, file_path):
        """Process the dataset and analyze all communication threads"""
        try:
            # Read and parse JSON data
            with open(file_path, 'r') as f:
                data = f.read()
                # Handle both array and object formats
                try:
                    messages = json.loads(data)
                    # If messages is a dict with an 'emails' key, get the emails array
                    if isinstance(messages, dict) and 'emails' in messages:
                        messages = messages['emails']
                except json.JSONDecodeError:
                    # If the file contains multiple JSON objects, one per line
                    messages = [json.loads(line) for line in data.splitlines() if line.strip()]
            
            # Group messages by threadId
            threads = {}
            for message in messages:
                thread_id = message['threadId']
                if thread_id not in threads:
                    threads[thread_id] = []
                threads[thread_id].append(message)
            
            # Analyze each thread
            results = []
            for thread_messages in threads.values():
                analysis = self.analyze_thread(thread_messages)
                if analysis:
                    results.append(analysis)
            
            # Save results
            self.save_results(results)
            return results
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise

    def save_results(self, results):
        """Save results to CSV files"""
        approved = []
        suspicious = []
        
        for res in results:
            if res['status'] == 'approved':
                approved.append([
                    res['sender_name'],
                    res['sender_email'],
                    res['subject'],
                    res['first_message_date']
                ])
            else:
                suspicious.append([
                    res['sender_email'],
                    res['total_risk_score'],
                    res['components']['excess_caps_score'],
                    res['components']['exclamation_score'],
                    res['components']['spam_word_score'],
                    res['grammar_errors'],
                    res['subject'],
                    res['first_message_date']
                ])
        
        # Save approved communications
        with open('approved_vendors.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sender Name', 'Email Address', 'Subject', 'Date'])
            writer.writerows(approved)
        
        # Save suspicious communications
        with open('suspicious_communications.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sender Email', 'Total Risk Score', 'Excess Caps Score',
                           'Exclamation Score', 'Spam Words Score', 'Grammar Errors',
                           'Subject', 'Date'])
            writer.writerows(suspicious)

def main():
    # Initialize analyzer with your company domain
    analyzer = VendorCommunicationAnalyzer(company_domain='@cybercellar.in')
    
    # Process your dataset
    file_path = "/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/chirag_dataset.json"  # Update this path
    results = analyzer.process_dataset(file_path)
    
    if results:
        print("\nAnalysis Summary:")
        print(f"Total threads analyzed: {len(results)}")
        
        safe_count = sum(1 for r in results if r['status'] == 'approved')
        suspicious_count = sum(1 for r in results if r['status'] == 'suspicious')
        
        print(f"Approved communications: {safe_count}")
        print(f"Suspicious communications: {suspicious_count}")
        print("\nResults saved to approved_vendors.csv and suspicious_communications.csv")

if __name__ == "__main__":
    main()
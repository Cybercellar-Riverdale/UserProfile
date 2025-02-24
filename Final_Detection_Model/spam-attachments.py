import os
import pandas as pd
import numpy as np
from docx import Document
from PIL import Image
import pytesseract
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
from datetime import datetime
import openpyxl
import csv
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='spam_detection.log'
)

class Config:
    # Model Configuration
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 512
    BATCH_SIZE = 32
    
    # System Configuration
    CPU_WORKERS = 8
    CHUNK_SIZE = 1000
    
    # GPU Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # Spam Keywords
    SPAM_KEYWORDS = [
        "lottery", "win", "winner", "free", "cash", "price", "prize", "reward", 
        "congratulations", "claim", "offer", "urgent", "money", "million", "billion",
        "cash prize", "big winner", "cash load", "lottery wins", "cash reward",
        "discount", "limited time", "exclusive", "click here", "unsubscribe",
        "act now", "guaranteed", "risk-free", "special promotion"
    ]

def setup_gpu():
    """Configure GPU settings and return GPU availability status"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"GPU Setup Complete - Using {gpu_name}")
        logging.info(f"Available GPU Memory: {gpu_memory:.2f} GB")
        return True
    else:
        logging.warning("No GPU available! Using CPU instead.")
        return False

def read_txt(file_path):
    """Read text from a .txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def read_docx(file_path):
    """Read text from a .docx file"""
    try:
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def read_pdf(file_path):
    """Read text from a PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def read_image(file_path):
    """Extract text from an image using OCR"""
    try:
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return pytesseract.image_to_string(img)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def read_excel(file_path):
    """Read text from an Excel file"""
    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        return ' '.join(df.astype(str).values.flatten())
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def read_csv(file_path):
    """Read text from a CSV file"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        return ' '.join(df.astype(str).values.flatten())
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
            return ' '.join(df.astype(str).values.flatten())
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return None
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return None

def process_file(file_info):
    """Process a single file and extract its text content"""
    filename, file_path = file_info
    try:
        text = None
        ext = os.path.splitext(filename.lower())[1]
        
        # Extract true label from filename (assuming filename format: spam_001.txt or ham_001.txt)
        true_label = 1 if 'spam' in filename.lower() else 0
        
        if ext == '.txt':
            text = read_txt(file_path)
        elif ext == '.docx':
            text = read_docx(file_path)
        elif ext == '.pdf':
            text = read_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            text = read_image(file_path)
        elif ext in ['.xlsx', '.xls']:
            text = read_excel(file_path)
        elif ext == '.csv':
            text = read_csv(file_path)
        
        if text:
            return {
                'filename': filename, 
                'text': text,
                'true_label': true_label
            }
        return None

    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return None

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class SpamDetector:
    def __init__(self):
        self.setup_model()
        
    def setup_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME)
            self.model = self.model.to(Config.DEVICE)
            self.model.eval()
            torch.cuda.empty_cache()
            logging.info(f"Model loaded successfully on {Config.DEVICE}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def keyword_check(self, text):
        """Additional keyword-based spam detection"""
        if not text:
            return 0
        text_lower = text.lower()
        return int(any(keyword in text_lower for keyword in Config.SPAM_KEYWORDS))

    @torch.no_grad()
    def predict_batch(self, texts):
        try:
            dataset = TextDataset(texts, self.tokenizer, Config.MAX_LENGTH)
            dataloader = DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                num_workers=Config.CPU_WORKERS,
                pin_memory=Config.PIN_MEMORY
            )

            model_predictions = []
            keyword_predictions = []
            
            for batch in dataloader:
                # Model predictions
                input_ids = batch['input_ids'].to(Config.DEVICE)
                attention_mask = batch['attention_mask'].to(Config.DEVICE)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                model_predictions.extend(predictions.cpu().numpy())
                
                # Keyword predictions
                batch_texts = texts[len(keyword_predictions):len(keyword_predictions) + len(predictions)]
                keyword_predictions.extend([self.keyword_check(text) for text in batch_texts])

            # Combine predictions
            combined_predictions = [
                1 if (m == 1 or k == 1) else 0 
                for m, k in zip(model_predictions, keyword_predictions)
            ]

            return combined_predictions, model_predictions, keyword_predictions
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            return [0] * len(texts), [0] * len(texts), [0] * len(texts)

def calculate_metrics(y_true, y_pred):
    """Calculate and return classification metrics"""
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        return None

def save_results_safely(df, output_file):
    try:
        # Calculate metrics for each prediction type
        metrics = {
            'Combined': calculate_metrics(df['true_label'], df['is_spam']),
            'Model': calculate_metrics(df['true_label'], df['model_prediction']),
            'Keyword': calculate_metrics(df['true_label'], df['keyword_prediction'])
        }
        
        # Log metrics
        logging.info("\nClassification Metrics:")
        for method, method_metrics in metrics.items():
            logging.info(f"\n{method} Method Metrics:")
            for metric, value in method_metrics.items():
                logging.info(f"{metric.capitalize()}: {value:.4f}")
        
        # Save detailed results to CSV
        df.to_csv(output_file, 
                  index=False,
                  escapechar='\\',
                  encoding='utf-8',
                  quoting=csv.QUOTE_ALL)
        logging.info(f"Results saved successfully to {output_file}")
        
        # Save metrics to a separate CSV
        metrics_df = pd.DataFrame({
            'Method': [],
            'Metric': [],
            'Value': []
        })
        
        for method, method_metrics in metrics.items():
            for metric, value in method_metrics.items():
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Method': [method],
                    'Metric': [metric],
                    'Value': [value]
                })])
        
        metrics_file = f'metrics_{output_file}'
        metrics_df.to_csv(metrics_file, index=False)
        logging.info(f"Metrics saved separately to {metrics_file}")
        
        return metrics
        
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        try:
            safe_df = pd.DataFrame({
                'filename': df['filename'],
                'is_spam': df['is_spam'],
                'true_label': df['true_label']
            })
            safe_df.to_csv(f"safe_{output_file}",
                          index=False,
                          quoting=csv.QUOTE_ALL,
                          escapechar='\\')
            logging.info(f"Fallback results saved to safe_{output_file}")
        except Exception as e:
            logging.error(f"Critical error saving results: {e}")

def main():
    # Setup GPU
    setup_gpu()
    
    # Set your folder path
    folder_path = r'C:\Users\LAB-305-02\Downloads\SPAM-DETECT (2)\SPAM-DETECT\Dataset'
    
    # Initialize detector
    detector = SpamDetector()
    
    # Get all files
    all_files = []
    for root, _, files in os.walk(folder_path):
        all_files.extend((f, os.path.join(root, f)) for f in files)
    
    logging.info(f"Found {len(all_files)} files to process")
    
    # Process files in parallel
    processed_data = []
    with ThreadPoolExecutor(max_workers=Config.CPU_WORKERS) as executor:
        future_to_file = {executor.submit(process_file, file_info): file_info 
                         for file_info in all_files}
        
        with tqdm(total=len(all_files), desc="Processing files") as pbar:
            for future in as_completed(future_to_file):
                result = future.result()
                if result:
                    processed_data.append(result)
                pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    if df.empty:
        logging.error("No valid files were processed")
        return
    
    # Process in optimized batches
    logging.info("Starting spam detection...")
    combined_predictions = []
    model_predictions = []
    keyword_predictions = []
    
    for i in tqdm(range(0, len(df), Config.CHUNK_SIZE), desc="Detecting spam"):
        chunk = df['text'][i:i + Config.CHUNK_SIZE].tolist()
        chunk_combined, chunk_model, chunk_keyword = detector.predict_batch(chunk)
        combined_predictions.extend(chunk_combined)
        model_predictions.extend(chunk_model)
        keyword_predictions.extend(chunk_keyword)
    
    df['is_spam'] = combined_predictions
    df['model_prediction'] = model_predictions
    df['keyword_prediction'] = keyword_predictions
    
    # Save results and get metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'spam_detection_results_{timestamp}.csv'
    metrics = save_results_safely(df, output_file)
    
    # Print statistics
    spam_count = df['is_spam'].sum()
    total_count = len(df)
    logging.info(f"""
    Processing Complete:
    Total files processed: {total_count}
    Spam detected: {spam_count}
    Ham detected: {total_count - spam_count}
    Spam ratio: {spam_count/total_count*100:.2f}%
    Model-only spam detections: {df['model_prediction'].sum()}
    Keyword-only spam detections: {df['keyword_prediction'].sum()}
    Results saved to: {output_file}
    """)
    
    # Print metrics summary
    if metrics:
        print("\nClassification Metrics Summary:")
        for method, method_metrics in metrics.items():
            print(f"\n{method} Method:")
            for metric, value in method_metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
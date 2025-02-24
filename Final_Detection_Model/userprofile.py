import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter

# Load data from JSON files
with open('/Users/priyanshuwalia7/Downloads/Riverdale/Dataset_json/login_merged.json') as login_file:
    login_data = json.load(login_file)

with open('/Users/priyanshuwalia7/Downloads/Riverdale/Dataset_json/reports_response_merge.json') as reports_file:
    reports_data = json.load(reports_file)

# Load risk summary data from JSON
with open('/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/risk_summary.json') as f:
    risk_summary = json.load(f)
    
def get_latest_login(email):
    """Helper function to extract the latest login details for a user"""
    latest_login = None
    latest_time = None
    for item in login_data:
        if item.get('actor', {}).get('email') == email:
            login_time = datetime.strptime(item['id']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if not latest_time or login_time > latest_time:
                latest_time = login_time
                latest_login = item
    return latest_login

def get_latest_report(email):
    """Helper function to extract the latest report details for a user"""
    latest_report = None
    latest_time = None
    for item in reports_data:
        if item.get('actor', {}).get('email') == email:
            report_time = datetime.strptime(item['id']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if not latest_time or report_time > latest_time:
                latest_time = report_time
                latest_report = item
    return latest_report

def generate_profile(email):
    # Load risk analysis data
    risk_csv_path = "/Users/priyanshuwalia7/Downloads/Riverdale/detection_model/enhanced_risk_assessment_results.csv"
    risk_data = pd.read_csv(risk_csv_path)
    
    # Filter data for the specific email
    user_risk_data = risk_data[risk_data['from'] == email].copy()
    
    if user_risk_data.empty:
        print(f"No data found for email: {email}")
        return
    
    # Get averages from JSON summary
    avg_risk_score = risk_summary['average_risk_score']
    avg_grammar_score = risk_summary['average_grammar_score']
    
    # Calculate sentiment from component averages
    spam_score_avg = risk_summary['component_averages']['spam_word_score']
    negative_sentiment = min(100, spam_score_avg * 2)  # Scale spam score to 0-100
    positive_sentiment = max(0, 100 - negative_sentiment - 20)
    neutral_sentiment = 100 - positive_sentiment - negative_sentiment

    # Get login and report data
    login_data = get_latest_login(email)
    report_data = get_latest_report(email)
    
    # Get latest 10 emails for historical trends
    latest_emails = user_risk_data.sort_values('date', ascending=False).head(10)

    # Plotting
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor('#f3f4f6')
    
    # Header
    plt.suptitle('User Profile', fontsize=28, fontweight='bold', ha='center', color='#2e7d32')
    
    # User Details Section
    ax_details = fig.add_axes([0.1, 0.7, 0.8, 0.2])
    details_text = f"""
    Email ID: {login_data['actor']['email'] if login_data else email}
    IP Address: {login_data.get('ipAddress', 'N/A') if login_data else 'N/A'}
    Login Time: {login_data['id']['time'] if login_data else 'N/A'}
    Login Type: {next((param['value'] for param in login_data['events'][0]['parameters'] 
                     if param['name'] == 'login_type'), 'N/A') if login_data else 'N/A'}
    Suspicious Login: {next((param.get('boolValue', False) for param in login_data['events'][0]['parameters'] 
                          if param['name'] == 'is_suspicious'), False) if login_data else 'N/A'}
    Device Type: {next((param['value'] for param in report_data['events'][0]['parameters'] 
                     if param['name'] == 'DEVICE_TYPE'), 'N/A') if report_data else 'N/A'}
    OS Version: {next((param['value'] for param in report_data['events'][0]['parameters'] 
                    if param['name'] == 'OS_VERSION'), 'N/A') if report_data else 'N/A'}
    """
    ax_details.text(0.5, 0.5, details_text, fontsize=14, ha='center', va='center',
                   transform=ax_details.transAxes,
                   bbox=dict(facecolor='#ffffff', edgecolor='#2e7d32', boxstyle='round,pad=1'))
    ax_details.axis('off')
    
    # Left Subplot: Grammar Score
    ax_grammar = fig.add_axes([0.05, 0.4, 0.25, 0.25], aspect='equal')
    ax_grammar.pie([avg_grammar_score, 100 - avg_grammar_score], 
                  labels=[f'Score\n({avg_grammar_score:.1f}%)', ''],
                  colors=['#4caf50', '#cfd8dc'], startangle=90,
                  wedgeprops={'edgecolor': 'white'})
    ax_grammar.set_title('Grammar Score', fontsize=16, fontweight='bold', color='#2e7d32')
    
    # Middle Subplot: Sentiment Analysis
    ax_sentiment = fig.add_axes([0.37, 0.4, 0.25, 0.25], aspect='equal')
    sentiment_values = [positive_sentiment, negative_sentiment, neutral_sentiment]
    sentiment_labels = [f'Positive\n({positive_sentiment:.1f}%)',
                       f'Negative\n({negative_sentiment:.1f}%)',
                       f'Neutral\n({neutral_sentiment:.1f}%)']
    ax_sentiment.pie(sentiment_values, labels=sentiment_labels,
                    colors=['#81c784', '#e57373', '#ffb74d'], startangle=140,
                    wedgeprops={'edgecolor': 'white'})
    ax_sentiment.set_title('Sentiment Analysis', fontsize=16, fontweight='bold', color='#37474f')
    
    # Right Subplot: Risk Score
    ax_risk = fig.add_axes([0.7, 0.4, 0.25, 0.25], aspect='equal')
    ax_risk.pie([avg_risk_score, 100 - avg_risk_score],
                labels=[f'Score\n({avg_risk_score:.1f}%)', ''],
                colors=['#f44336', '#cfd8dc'], startangle=90,
                wedgeprops={'edgecolor': 'white'})
    ax_risk.set_title('Risk Score', fontsize=16, fontweight='bold', color='#d32f2f')
    
    # Bottom Subplot: Historical Trends
    ax_trends = fig.add_axes([0.1, 0.1, 0.8, 0.25])

    try:
        # Plot grammar vs risk scores for latest 10 emails
        x = range(1, len(latest_emails) + 1)
        ax_trends.plot(x, latest_emails['grammar_score'], 
                    label='Grammar Score', color='#4caf50', marker='o')
        ax_trends.plot(x, latest_emails['risk_score'], 
                    label='Risk Score', color='#f44336', marker='o')
        
        ax_trends.set_xticks(x)
        ax_trends.set_xlabel('Latest Emails (1 = Most Recent)', fontsize=12)
        ax_trends.set_ylabel('Score', fontsize=12)
        ax_trends.legend()
        ax_trends.grid(True, linestyle='--', alpha=0.6)
        ax_trends.set_ylim(-5, 105)  # Added 5% padding top/bottom
        
    except Exception as e:
        print(f"Error plotting historical trends: {str(e)}")
        ax_trends.text(0.5, 0.5, 'Historical data not available',
                    ha='center', va='center', transform=ax_trends.transAxes)
    
    ax_trends.set_title('Historical Trends - Last 10 Emails', fontsize=16, fontweight='bold', color='#37474f')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

# Rest of the UserAnalysis class and main function remains the same
# ... [Keep the rest of the code unchanged] ...
class UserAnalysis:
    def __init__(self):
        self.login_file_path = '/Users/priyanshuwalia7/Downloads/Riverdale/Dataset_json/login_merged.json'
        self.reports_file_path = '/Users/priyanshuwalia7/Downloads/Riverdale/Dataset_json/reports_response_merge.json'
        
    def parse_login_data(self) -> pd.DataFrame:
        """Parse and clean login data from JSON file."""
        with open(self.login_file_path, 'r') as f:
            data = json.load(f)
            
        extracted_data = []
        for record in data:
            # Extract email from actor object
            email = record.get('actor', {}).get('email')
            
            # Extract login time from id object
            login_time = record.get('id', {}).get('time')
            
            # Safely extract is_suspicious parameter
            events = record.get('events', [])
            is_suspicious = False
            if events:
                parameters = events[0].get('parameters', [])
                for param in parameters:
                    if param.get('name') == 'is_suspicious':
                        is_suspicious = param.get('boolValue', False)
                        break
            
            if email and login_time:
                extracted_data.append({
                    'email': email,
                    'login_time': login_time,
                    'ip_address': record.get('ipAddress'),
                    'is_suspicious': is_suspicious
                })
        
        df = pd.DataFrame(extracted_data)
        df['login_time'] = pd.to_datetime(df['login_time'])
        return df

    def format_time_axis(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"

    def detect_anomalies(self, user_data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Detect anomalous login patterns using K-means clustering."""
        X = np.column_stack([
            user_data['time_seconds'],
            user_data['login_time'].dt.dayofweek
        ])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_clusters = min(3, len(user_data))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        threshold = np.percentile(distances, 95)
        anomalies = distances > threshold
        
        descriptions = []
        for i, (is_anomaly, login_time) in enumerate(zip(anomalies, user_data['login_time'])):
            if is_anomaly:
                time_str = login_time.strftime('%Y-%m-%d %H:%M:%S')
                day_of_week = login_time.strftime('%A')
                
                if login_time.hour < 5 or login_time.hour > 22:
                    desc = f"Suspicious late-night login at {time_str} ({day_of_week})"
                else:
                    desc = f"Unusual login pattern detected at {time_str} ({day_of_week})"
                descriptions.append(desc)
        
        return anomalies, descriptions

    def create_login_visualization(self, df: pd.DataFrame, user_email: str) -> None:
        """Create visualization of login times with anomaly detection."""
        user_data = df[df['email'] == user_email].copy()
        
        if user_data.empty:
            print(f"No login data found for email: {user_email}")
            return
        
        user_data['login_date'] = user_data['login_time'].dt.date
        user_data['time_seconds'] = (
            user_data['login_time'].dt.hour * 3600 +
            user_data['login_time'].dt.minute * 60 +
            user_data['login_time'].dt.second
        )
        
        anomalies, anomaly_descriptions = self.detect_anomalies(user_data)
        
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.3)
        
        ax1 = fig.add_subplot(gs[0])
        
        normal_points = ~anomalies
        scatter_normal = ax1.scatter(
            x=user_data.loc[normal_points, 'login_date'],
            y=user_data.loc[normal_points, 'time_seconds'],
            c=user_data.loc[normal_points, 'time_seconds'],
            cmap='viridis',
            s=150,
            alpha=0.8,
            edgecolor='white',
            linewidth=1,
            label='Normal Login'
        )
        
        if any(anomalies):
            ax1.scatter(
                x=user_data.loc[anomalies, 'login_date'],
                y=user_data.loc[anomalies, 'time_seconds'],
                color='red',
                s=200,
                alpha=0.9,
                marker='X',
                label='Suspicious Login'
            )
        
        ax1.set_title(f'Login Activity Analysis for {user_email}', 
                    fontsize=22, pad=25, fontweight='bold')
        ax1.set_xlabel('Date of Login', fontsize=16, labelpad=15)
        ax1.set_ylabel('Time of Login', fontsize=16, labelpad=15)
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=12)
        
        y_ticks = range(0, 86401, 7200)
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels([self.format_time_axis(float(t)) for t in y_ticks], fontsize=12)
        
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right', fontsize=14)
        
        cbar = plt.colorbar(scatter_normal, ax=ax1)
        cbar.set_label('Time of Day', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        
        stats_text = (
            f"Total Logins: {len(user_data)}\n"
            f"First Login: {user_data['login_time'].min().strftime('%Y-%m-%d')}\n"
            f"Last Login: {user_data['login_time'].max().strftime('%Y-%m-%d')}\n"
            f"Suspicious Logins Detected: {sum(anomalies)}\n\n"
            "Suspicious Activity Details:\n" + "\n".join(anomaly_descriptions)
        )
        
        ax2.text(0.02, 0.95, stats_text,
                fontsize=12,
                ha='left',
                va='top',
                transform=ax2.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=10))
        
        plt.tight_layout()
        plt.show()

    def extract_email_data(self, email: str) -> list[dict]:
        """Extract device data for the given email."""
        with open(self.reports_file_path, 'r') as f:
            data = json.load(f)
            
        filtered_data = []
        for record in data:
            events = record.get('events', [])
            for event in events:
                parameters = event.get('parameters', [])
                device_info = {}
                for param in parameters:
                    if 'name' in param and 'value' in param:
                        value = param['value']
                        if param['name'] == 'OS_VERSION' and (not isinstance(value, str) or value == ''):
                            value = 'NaN'
                        device_info[param['name']] = value
                    elif 'intValue' in param:
                        device_info[param['name']] = param['intValue']
                if device_info.get('USER_EMAIL') == email:
                    filtered_data.append(device_info)
        return filtered_data

    def plot_pie_chart(self, data: list[dict], title: str, key: str) -> None:
        """Plot a pie chart for the given key."""
        counter = Counter(item.get(key, 'Unknown') for item in data)
        if counter:
            labels, sizes = zip(*counter.items())
            plt.figure()
            plt.pie(sizes, labels=[f"{label} ({size})" for label, size in zip(labels, sizes)], 
                   autopct='%1.1f%%', startangle=90)
            plt.title(title)
            plt.show()

    def plot_bar_chart(self, data: list[dict], title: str, key: str) -> None:
        """Plot a bar chart for the given key."""
        counter = Counter(item.get(key, 'NaN') if not item.get(key) else item[key] for item in data)
        if counter:
            labels, sizes = zip(*counter.items())
            if 'NaN' not in labels:
                labels = list(labels) + ['NaN']
                sizes = list(sizes) + [0]
            plt.figure()
            plt.bar(labels, sizes)
            plt.title(title)
            plt.xlabel(key)
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.show()

    def plot_last_sync_dates(self, data: list[dict]) -> None:
        """Plot last sync audit dates."""
        dates = [
            datetime.fromtimestamp(int(item['LAST_SYNC_AUDIT_DATE']) / 1000)
            for item in data if 'LAST_SYNC_AUDIT_DATE' in item
        ]
        if dates:
            dates.sort()
            counts = Counter(dates)
            plt.figure()
            plt.plot(list(counts.keys()), list(counts.values()), marker='o')
            plt.title("Last Sync Audit Dates")
            plt.xlabel("Date")
            plt.ylabel("Frequency")
            plt.grid()
            plt.gcf().autofmt_xdate()
            plt.show()

    def analyze_user(self, email: str) -> None:
        """Perform complete analysis for a user."""
        try:
            # Login analysis
            login_data = self.parse_login_data()
            self.create_login_visualization(login_data, email)
            
            # Device analysis
            device_data = self.extract_email_data(email)
            if device_data:
                self.plot_pie_chart(device_data, "Device Type Distribution", "DEVICE_TYPE")
                self.plot_bar_chart(device_data, "Device Model Distribution", "DEVICE_MODEL")
                self.plot_bar_chart(device_data, "OS Version Distribution", "OS_VERSION")
                self.plot_last_sync_dates(device_data)
                self.plot_bar_chart(device_data, "Resource ID Distribution", "RESOURCE_ID")
            else:
                print(f"No device data found for email: {email}")
                
        except FileNotFoundError as e:
            print(f"Error: File not found - {str(e)}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    """Main function to run the analysis."""
    print("Welcome, Admin! Generate a user profile.")
    email = input("Enter the user's email address: ").strip()
    generate_profile(email)
    
    visualize = input("Do you want to visualize the data? (yes/no): ").strip().lower()
    if visualize == 'yes':
        analyzer = UserAnalysis()
        analyzer.analyze_user(email)

if __name__ == "__main__":
    main()
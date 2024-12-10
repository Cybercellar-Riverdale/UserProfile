import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import random

# Load data from the JSON files
with open('/Users/priyanshuwalia7/Downloads/new river/river/merged.json') as login_file:
    login_data = json.load(login_file)

with open('/Users/priyanshuwalia7/Downloads/new river/river/reports_response_merge.json') as reports_file:
    reports_data = json.load(reports_file)


# Helper function to extract the latest login details for a user
def get_latest_login(email):
    latest_login = None
    latest_time = None
    for item in login_data:
        if item.get('actor', {}).get('email') == email:
            login_time = datetime.strptime(item['id']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if not latest_time or login_time > latest_time:
                latest_time = login_time
                latest_login = item
    return latest_login


# Helper function to extract the latest report details for a user
def get_latest_report(email):
    latest_report = None
    latest_time = None
    for item in reports_data:
        if item.get('actor', {}).get('email') == email:
            report_time = datetime.strptime(item['id']['time'], "%Y-%m-%dT%H:%M:%S.%fZ")
            if not latest_time or report_time > latest_time:
                latest_time = report_time
                latest_report = item
    return latest_report


# Simulate random data for historical trends
def simulate_scores(num_points=10):
    current_time = datetime.now()
    times = [current_time - timedelta(minutes=5 * i) for i in range(num_points)]
    grammar_scores = [random.randint(60, 100) for _ in range(num_points)]
    risk_scores = [random.randint(0, 50) for _ in range(num_points)]
    return times, grammar_scores, risk_scores


# Simulate random sentiment analysis data (ensures non-negative values)
def simulate_sentiments():
    positive = random.randint(30, 70)
    negative = random.randint(10, 50)
    neutral = max(0, 100 - (positive + negative))  # Ensure the sum does not exceed 100
    return {"Positive": positive, "Negative": negative, "Neutral": neutral}


# Function to generate a user profile for a specific user
def generate_profile(email):
    login_data = get_latest_login(email)
    report_data = get_latest_report(email)

    if not login_data and not report_data:
        print(f"No data found for the user: {email}")
        return

    # Extract details from login data
    login_email = login_data['actor']['email'] if login_data else "N/A"
    login_ip = login_data.get('ipAddress', 'N/A') if login_data else "N/A"
    login_time = login_data['id']['time'] if login_data else "N/A"
    login_type = (
        next(
            (param['value'] for param in login_data['events'][0]['parameters'] if param['name'] == 'login_type'),
            "N/A",
        )
        if login_data
        else "N/A"
    )
    is_suspicious = (
        next(
            (param.get('boolValue', False) for param in login_data['events'][0]['parameters'] if param['name'] == 'is_suspicious'),
            False,
        )
        if login_data
        else False
    )

    # Extract details from reports data
    device_type = (
        next(
            (param['value'] for param in report_data['events'][0]['parameters'] if param['name'] == 'DEVICE_TYPE'),
            "N/A",
        )
        if report_data
        else "N/A"
    )
    os_version = (
        next(
            (param['value'] for param in report_data['events'][0]['parameters'] if param['name'] == 'OS_VERSION'),
            "N/A",
        )
        if report_data
        else "N/A"
    )

    # Generate random data for historical trends and sentiment analysis
    times, grammar_scores, risk_scores = simulate_scores()
    sentiments = simulate_sentiments()

    # Plotting the profile
    fig = plt.figure(figsize=(15, 12))
    fig.patch.set_facecolor('#f3f4f6')

    # Header
    plt.suptitle('User Profile', fontsize=28, fontweight='bold', ha='center', color='#2e7d32')

    # User Details Section
    ax_details = fig.add_axes([0.1, 0.7, 0.8, 0.2])
    ax_details.text(0.5, 0.5, f"""
    Email ID: {login_email}
    IP Address: {login_ip}
    Login Time: {login_time}
    Login Type: {login_type}
    Suspicious Login: {is_suspicious}
    Device Type: {device_type}
    OS Version: {os_version}
    """, fontsize=14, ha='center', va='center', transform=ax_details.transAxes,
        bbox=dict(facecolor='#ffffff', edgecolor='#2e7d32', boxstyle='round,pad=1'))
    ax_details.axis('off')

    # Left Subplot: Gauge Chart for Grammar Score
    ax_grammar = fig.add_axes([0.05, 0.4, 0.25, 0.25], aspect='equal')
    grammar_score = grammar_scores[0]  # Latest grammar score
    ax_grammar.pie([grammar_score, 100 - grammar_score], labels=['Score', ''],
                   autopct='%1.1f%%', colors=['#4caf50', '#cfd8dc'], startangle=90, wedgeprops={'edgecolor': 'white'})
    ax_grammar.set_title('Grammar Score', fontsize=16, fontweight='bold', color='#2e7d32')

    # Middle Subplot: Pie Chart for Sentiment Analysis
    ax_sentiment = fig.add_axes([0.37, 0.4, 0.25, 0.25], aspect='equal')
    sentiment_labels = list(sentiments.keys())
    sentiment_values = list(sentiments.values())
    sentiment_colors = ['#81c784', '#e57373', '#ffb74d']
    ax_sentiment.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%',
                     colors=sentiment_colors, startangle=140, wedgeprops={'edgecolor': 'white'})
    ax_sentiment.set_title('Sentiment Analysis', fontsize=16, fontweight='bold', color='#37474f')

    # Right Subplot: Gauge Chart for Risk Score
    ax_risk = fig.add_axes([0.7, 0.4, 0.25, 0.25], aspect='equal')
    risk_score = risk_scores[0]  # Latest risk score
    ax_risk.pie([risk_score, 100 - risk_score], labels=['Score', ''],
                autopct='%1.1f%%', colors=['#f44336', '#cfd8dc'], startangle=90, wedgeprops={'edgecolor': 'white'})
    ax_risk.set_title('Risk Score', fontsize=16, fontweight='bold', color='#d32f2f')

    # Bottom Subplot: Dynamic Timeline for Historical Trends
    ax_trends = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    ax_trends.plot(times, grammar_scores, label='Grammar Score', color='#4caf50', marker='o')
    ax_trends.plot(times, risk_scores, label='Risk Score', color='#f44336', marker='o')
    ax_trends.set_title('Historical Trends', fontsize=16, fontweight='bold', color='#37474f')
    ax_trends.set_xlabel('Time', fontsize=12)
    ax_trends.set_ylabel('Score', fontsize=12)
    ax_trends.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax_trends.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    ax_trends.legend()
    ax_trends.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()


# Allow admin to input email
def main():
    print("Welcome, Admin! Generate a user profile.")
    email = input("Enter the user's email address: ").strip()
    generate_profile(email)


# Run the program
if __name__ == "__main__":
    main()

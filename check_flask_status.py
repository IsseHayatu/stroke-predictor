import requests

def check_flask_status():
    url = 'http://127.0.0.1:5000'  # Replace with your actual URL if deployed
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("Flask app is running!")
        else:
            print(f"Flask app is not running, received status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error while checking Flask status: {e}")

check_flask_status()

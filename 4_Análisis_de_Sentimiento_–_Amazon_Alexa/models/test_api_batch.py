import requests

# Run this script with the server running
url = "http://127.0.0.1:5000/predict"
payload = {
    "texts": [
        "I love my Alexa!",
        "This Alexa is terrible, it never works.",
        "The sound quality is amazing.",
    ]
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())

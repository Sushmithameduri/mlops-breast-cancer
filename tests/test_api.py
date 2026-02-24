"""
API testing script
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing Breast Cancer API\n")
    
    # Test 1: Root endpoint
    print("1. Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")
    
    # Test 2: Health check
    print("2. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")
    
    # Test 3: Make prediction
    print("3. Testing prediction endpoint...")
    payload = {
        "features": [
            17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
            0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
            0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")
    
    # Test 4: Model info
    print("4. Testing model info endpoint...")
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    try:
        test_api()
        print("✓ All API tests passed!")
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to API")
        print("  Make sure API is running: uvicorn app.main:app --reload")
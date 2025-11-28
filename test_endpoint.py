#!/usr/bin/env python3
"""Test script for the quiz endpoint"""

import requests
import json
import sys
from dotenv import load_dotenv
import os

load_dotenv()

def test_endpoint(base_url, email, secret):
    """Test the quiz endpoint"""
    
    # Test 1: Health check
    print("\n" + "="*60)
    print("Test 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 200, "Health check failed"
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {str(e)}")
        return False
    
    # Test 2: Invalid JSON
    print("\n" + "="*60)
    print("Test 2: Invalid JSON")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/quiz",
            data="invalid json",
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 400, "Should return 400 for invalid JSON"
        print("✓ Invalid JSON handling passed")
    except Exception as e:
        print(f"✗ Invalid JSON test failed: {str(e)}")
    
    # Test 3: Missing fields
    print("\n" + "="*60)
    print("Test 3: Missing Fields")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/quiz",
            json={"email": email},
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        assert response.status_code == 400, "Should return 400 for missing fields"
        print("✓ Missing fields handling passed")
    except Exception as e:
        print(f"✗ Missing fields test failed: {str(e)}")
    
    # Test 4: Invalid secret
    print("\n" + "="*60)
    print("Test 4: Invalid Secret")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/quiz",
            json={
                "email": email,
                "secret": "wrong_secret",
                "url": "https://example.com/quiz"
            },
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status Code: {response.status_code}")
        assert response.status_code == 403, "Should return 403 for invalid secret"
        print("✓ Invalid secret handling passed")
    except Exception as e:
        print(f"✗ Invalid secret test failed: {str(e)}")
    
    # Test 5: Demo quiz
    print("\n" + "="*60)
    print("Test 5: Demo Quiz (This will take a few minutes)")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/quiz",
            json={
                "email": email,
                "secret": secret,
                "url": "https://tds-llm-analysis.s-anand.net/demo"
            },
            headers={'Content-Type': 'application/json'},
            timeout=180
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Demo quiz test passed")
        else:
            print("✗ Demo quiz test failed")
    except Exception as e:
        print(f"✗ Demo quiz test failed: {str(e)}")
    
    print("\n" + "="*60)
    print("Testing Complete")
    print("="*60)
    
    return True

if __name__ == "__main__":
    # Get configuration
    base_url = input("Enter your endpoint URL (e.g., http://localhost:5000): ").strip()
    if not base_url:
        base_url = "http://localhost:5000"
    
    email = os.getenv('STUDENT_EMAIL') or input("Enter your email: ").strip()
    secret = os.getenv('SECRET_KEY') or input("Enter your secret: ").strip()
    
    if not email or not secret:
        print("Error: Email and secret are required")
        sys.exit(1)
    
    print(f"\nTesting endpoint: {base_url}")
    print(f"Email: {email}")
    print(f"Secret: {'*' * len(secret)}")
    
    test_endpoint(base_url, email, secret)
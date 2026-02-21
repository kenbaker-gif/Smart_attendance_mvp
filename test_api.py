#!/usr/bin/env python3
"""
Test script for FastAPI endpoints

USAGE:
======
1. Start the server:
   uvicorn app.main:app --reload

2. Run this script:
   python test_api.py

This will test all available endpoints and show results.

REQUIREMENTS:
- requests library: pip install requests
- API running on http://localhost:8000
- (Optional) Test image file for face verification
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_get_students():
    """Test get students endpoint"""
    print("\n👥 Testing Get Students...")
    response = requests.get(f"{BASE_URL}/students?limit=5")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data[:1] if data else data, indent=2)}")
    return response.status_code == 200

def test_get_attendance_records():
    """Test get attendance records endpoint"""
    print("\n📋 Testing Get Attendance Records...")
    response = requests.get(f"{BASE_URL}/attendance-records?limit=5")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Response: {json.dumps(data[:1] if data else data, indent=2)}")
    return response.status_code == 200

def test_verify_face(test_image_path=None):
    """Test face verification endpoint"""
    print("\n🔐 Testing Face Verification...")
    
    if not test_image_path:
        print("  ⏭️  Skipping (no test image provided)")
        return None
    
    if not Path(test_image_path).exists():
        print(f"  ⏭️  Skipping (test image not found: {test_image_path})")
        return None
    
    with open(test_image_path, "rb") as f:
        files = {"file": f}
        data = {"student_id": "test_student_id"}
        response = requests.post(f"{BASE_URL}/verify", files=files, data=data)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_admin_sync(admin_secret=None):
    """Test admin sync endpoint"""
    print("\n⚙️  Testing Admin Sync Encodings...")
    
    if not admin_secret:
        print("  ⏭️  Skipping (no ADMIN_SECRET provided)")
        return None
    
    headers = {"Authorization": f"Bearer {admin_secret}"}
    response = requests.post(f"{BASE_URL}/admin/sync-encodings", headers=headers)
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def main():
    print("=" * 60)
    print("Smart Attendance System - FastAPI Endpoint Tests")
    print("=" * 60)
    
    results = {
        "Health Check": test_health(),
        "Get Students": test_get_students(),
        "Get Attendance Records": test_get_attendance_records(),
    }
    
    # Optional advanced tests
    results["Face Verification"] = test_verify_face()
    results["Admin Sync"] = test_admin_sync()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, result in results.items():
        if result is None:
            status = "⏭️  Skipped"
        elif result:
            status = "✅ Passed"
        else:
            status = "❌ Failed"
        print(f"{test_name}: {status}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()

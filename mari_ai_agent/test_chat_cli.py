#!/usr/bin/env python3
"""
Simple CLI Chat Test for Mari AI Agent
Usage: python test_chat_cli.py
"""

import requests
import json
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"  # Adjust if different
TEST_USER_ID = "5893"  # Matricula ID - LUCIANA GOENAGA - 6° grado (78 notas, promedio 3.97)

def chat_with_mari(message, conversation_history=None):
    """Send a message to Mari AI chat endpoint"""
    if conversation_history is None:
        conversation_history = []
    
    payload = {
        "user_id": TEST_USER_ID,
        "message": message,
        "conversation_history": conversation_history
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"HTTP {response.status_code}",
                "details": response.text
            }
            
    except requests.exceptions.ConnectionError:
        return {
            "error": "Connection failed",
            "details": "Make sure the Mari AI server is running on localhost:8000"
        }
    except Exception as e:
        return {
            "error": "Request failed", 
            "details": str(e)
        }

def print_response(response):
    """Print Mari AI response in a nice format"""
    if "error" in response:
        print(f"❌ Error: {response['error']}")
        if "details" in response:
            print(f"   Details: {response['details']}")
        return
    
    print(f"🤖 Mari AI: {response.get('response', 'No response')}")
    
    # Show additional info if available
    if response.get('confidence'):
        print(f"   Confidence: {response['confidence']:.2f}")
    
    if response.get('student_grade'):
        print(f"   Student Grade: {response['student_grade']}")

def main():
    """Main CLI chat loop"""
    print("🚀 Mari AI Chat Test CLI")
    print("=" * 50)
    print(f"Testing with Matricula ID: {TEST_USER_ID}")
    print("Type your messages (or 'quit' to exit)")
    print("-" * 50)
    
    conversation_history = []
    
    while True:
        try:
            # Get user input
            user_message = input(f"\n👤 You: ").strip()
            
            if user_message.lower() in ['quit', 'exit', 'salir']:
                print("👋 ¡Hasta luego!")
                break
            
            if not user_message:
                continue
            
            print("💭 Thinking...")
            
            # Send message to Mari AI
            response = chat_with_mari(user_message, conversation_history)
            
            # Print response
            print_response(response)
            
            # Update conversation history
            if "error" not in response:
                conversation_history.append({
                    "role": "user",
                    "content": user_message
                })
                conversation_history.append({
                    "role": "assistant", 
                    "content": response.get('response', '')
                })
                
                # Keep history manageable (last 10 messages)
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
            
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

def test_scenarios():
    """Run some test scenarios"""
    print("🧪 Running Test Scenarios...")
    print("=" * 50)
    
    test_cases = [
        "Hola Mari AI, ¿cómo estás?",
        "¿Qué es la fotosíntesis?",
        "Explícame las ecuaciones cuadráticas",
        "¿Cómo puedo mejorar en matemáticas?"
    ]
    
    for i, message in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {message}")
        response = chat_with_mari(message)
        print_response(response)
        print("-" * 30)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_scenarios()
    else:
        main()

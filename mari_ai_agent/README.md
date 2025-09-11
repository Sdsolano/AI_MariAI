# Mari AI Agent

An intelligent AI agent for educational assistance and prediction capabilities.

## Installation and Setup

### 1. Install Dependencies

First, install the required packages by running:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

Start the development server with:

```bash
uvicorn app.main:app --reload
```

The `--reload` flag enables automatic reloading when code changes are detected during development.

The application will be available at: `http://localhost:8000`

## API Documentation

Once the application is running, you can access:

- Interactive API documentation: `http://localhost:8000/docs`

## Important Endpoint: Chat API

The main endpoint for interacting with Mari AI is `/api/v1/chat/`. This endpoint allows you to send messages to the AI agent and receive educational assistance.

### Usage

**Endpoint:** `POST /api/v1/chat/`

**Request Body:**

```json
{
  "user_id": "5893",
  "message": "¿Qué es la fotosíntesis?",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hola"
    },
    {
      "role": "assistant", 
      "content": "¡Hola! Soy Mari AI, tu asistente educativo."
    }
  ]
}
```

**Response:**

```json
{
  "response": "La fotosíntesis es el proceso mediante el cual las plantas...",
  "confidence": 0.95,
  "student_grade": "6°"
}
```

### Testing the Chat Endpoint

You can test the chat functionality using the provided test script:

```bash
python test_chat_cli.py
```

This will start an interactive chat session where you can test different educational questions and see how Mari AI responds.

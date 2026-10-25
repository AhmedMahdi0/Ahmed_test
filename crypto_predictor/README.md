# CryptoPredictor Web Application

A cryptocurrency price prediction application using deep learning models.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python3 run.py
```

### If above doesn't work, then Running the application (FIRST TIME SETUP)
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 run.py
```

## Use this after running above (if virtual environment)
```bash
source .venv/bin/activate && python3 run.py
```

This will start the Flask API server on http://127.0.0.1:5000 and open the web UI in your default browser.

### Command Line Options

```bash
python run.py --port 8080 # custom port
python run.py --debug # immediate response to changes
```

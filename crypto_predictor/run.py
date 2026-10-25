#!/usr/bin/env python3
"""
CryptoPredictor - Run Script
Starts the Flask API server with the web UI.
"""

import sys
import os
import argparse
import webbrowser
from pathlib import Path


# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='CryptoPredictor - Up / Down signals for 1h, 4h and 1d candles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Start server on default port 5000
  python run.py --port 8080        # Start server on port 8080
  python run.py --debug            # Start in debug mode
  python run.py --no-browser       # Don't open browser automatically
        """
    )
    
    parser.add_argument(
        '--host', 
        default='127.0.0.1', 
        help='Host to bind to (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000, 
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no-browser', 
        action='store_true', 
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports and to show better error messages
    try:
        from backend.api import app, socketio
    except ImportError as e:
        print(f"Error importing CryptoPredictor: {e}")
        print("\nMake sure you have installed the requirements:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    url = f"http://{args.host}:{args.port}"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 CryptoPredictor - AI Cryptocurrency Price Predictions   ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   Server starting at: {url:<35}  ║
║                                                              ║
║   Press Ctrl+C to stop the server                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Open browser
    if not args.no_browser:
        webbrowser.open(url)
    
    if socketio is not None:
        socketio.run(
        app,
        host=args.host,
        port=args.port,
        debug=args.debug,
        allow_unsafe_werkzeug=True,  # local development server
        )
    else:
        app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        )

if __name__ == '__main__':
    main()

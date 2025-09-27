#!/usr/bin/env python3
"""
Simple HTTPS server for serving web UI files.
"""

import ssl
import http.server
import socketserver
import os
import sys

def start_https_server(port=8443):
    """Start HTTPS server serving files from webui directory."""
    
    # Change to webui directory
    webui_dir = os.path.join(os.path.dirname(__file__), 'webui')
    os.chdir(webui_dir)
    
    # Create HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(('0.0.0.0', port), handler)
    
    # Wrap with SSL
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert_file = os.path.join(os.path.dirname(__file__), 'cert.pem')
    key_file = os.path.join(os.path.dirname(__file__), 'key.pem')
    
    try:
        context.load_cert_chain(cert_file, key_file)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        print(f"HTTPS server running on https://10.25.14.107:{port}")
        print(f"Serving files from: {os.getcwd()}")
        print(f"Certificate: {cert_file}")
        print(f"Key: {key_file}")
        
        httpd.serve_forever()
        
    except FileNotFoundError as e:
        print(f"Certificate error: {e}")
        print("Make sure cert.pem and key.pem exist in the visionmate directory")
        sys.exit(1)
    except OSError as e:
        print(f"Port {port} is already in use. Try a different port.")
        sys.exit(1)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8443
    start_https_server(port)

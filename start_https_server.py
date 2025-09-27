#!/usr/bin/env python3
"""
Simple HTTPS server for phone camera access.

Modern browsers require HTTPS for camera access on mobile devices.
This script provides a simple HTTPS server for the phone camera demo.

Author: Vaibhav Chandgir (Integration & Snapdragon lead)
"""

import http.server
import ssl
import socketserver
import os
import sys

def start_https_server(port=8443):
    """Start HTTPS server for phone camera access."""
    
    # Create a simple self-signed certificate
    cert_file = "server.crt"
    key_file = "server.key"
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Creating self-signed certificate...")
        try:
            import subprocess
            
            # Create self-signed certificate
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:4096", 
                "-keyout", key_file, "-out", cert_file, "-days", "365", "-nodes",
                "-subj", "/C=US/ST=CA/L=San Francisco/O=VisionMate/OU=Demo/CN=localhost"
            ], check=True, capture_output=True)
            
            print(f"✓ Certificate created: {cert_file}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠ OpenSSL not found. Creating basic certificate...")
            # Fallback: create minimal cert files
            with open(key_file, 'w') as f:
                f.write("# Placeholder key file\n")
            with open(cert_file, 'w') as f:
                f.write("# Placeholder cert file\n")
            print("⚠ Please install OpenSSL or use HTTP for testing")
    
    # Start HTTPS server
    try:
        os.chdir("webui")
        
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
        
        # Add SSL context
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(cert_file, key_file)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        
        print(f"🔒 HTTPS Server running on https://0.0.0.0:{port}")
        print(f"📱 Phone camera: https://<laptop-ip>:{port}/phone_camera.html")
        print(f"🎮 Controller: https://<laptop-ip>:{port}/controller.html")
        print("\nNote: You may need to accept the self-signed certificate in your browser")
        print("Press Ctrl+C to stop")
        
        httpd.serve_forever()
        
    except Exception as e:
        print(f"Error starting HTTPS server: {e}")
        print("Falling back to HTTP server...")
        
        # Fallback to HTTP
        handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("0.0.0.0", port), handler)
        
        print(f"🌐 HTTP Server running on http://0.0.0.0:{port}")
        print(f"📱 Phone camera: http://<laptop-ip>:{port}/phone_camera.html")
        print(f"🎮 Controller: http://<laptop-ip>:{port}/controller.html")
        print("\nNote: Camera may not work on mobile browsers without HTTPS")
        print("Press Ctrl+C to stop")
        
        httpd.serve_forever()

if __name__ == "__main__":
    port = 8443
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    start_https_server(port)

#!/usr/bin/env python3
"""
Phone Camera Stream Server for VisionMate.

This module receives video stream from phone camera and feeds it
to the VisionMate processing pipeline for real-time navigation guidance.

Author: Vaibhav Chandgir (Integration & Snapdragon lead)
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
import base64
import logging
from typing import Optional
import threading
import time

logger = logging.getLogger(__name__)

class PhoneStreamReceiver:
    """Receives video stream from phone camera via WebSocket."""
    
    def __init__(self, port: int = 8766):
        self.port = port
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.is_streaming = False
        self.connection_count = 0
        
    async def handle_phone_stream(self, websocket):
        """Handle incoming video stream from phone."""
        client_ip = websocket.remote_address[0]
        logger.info(f"📱 Phone camera stream connected from {client_ip}")
        
        with self.frame_lock:
            self.connection_count += 1
            self.is_streaming = True
            logger.info(f"📊 Connection count: {self.connection_count}, Streaming: {self.is_streaming}")
        
        try:
            async for message in websocket:
                try:
                    # Parse the incoming message
                    data = json.loads(message)
                    logger.debug(f"📨 Received message type: {data.get('type', 'unknown')} from {client_ip}")
                    
                    if data.get("type") == "frame":
                        # Decode base64 frame
                        try:
                            frame_data = base64.b64decode(data["data"])
                            nparr = np.frombuffer(frame_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                with self.frame_lock:
                                    self.current_frame = frame.copy()  # Store a copy to avoid race conditions
                                    logger.info(f"Stored frame {data.get('frame_id', 0)}: {frame.shape}")
                                
                                # Send acknowledgment back to phone
                                response = {
                                    "type": "ack",
                                    "timestamp": time.time(),
                                    "frame_id": data.get("frame_id", 0)
                                }
                                await websocket.send(json.dumps(response))
                            else:
                                logger.warning("Failed to decode frame from phone")
                        except Exception as e:
                            logger.error(f"Error processing frame: {e}")
                    
                    elif data.get("type") == "ping":
                        # Respond to ping
                        await websocket.send(json.dumps({"type": "pong"}))
                        
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON from phone")
                except Exception as e:
                    logger.error(f"Error processing phone stream: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Phone camera stream disconnected from {client_ip}")
        except Exception as e:
            logger.error(f"Phone stream error: {e}")
        finally:
            with self.frame_lock:
                self.connection_count -= 1
                if self.connection_count == 0:
                    self.is_streaming = False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame from phone camera."""
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            else:
                return None
    
    def is_phone_streaming(self) -> bool:
        """Check if phone is currently streaming."""
        with self.frame_lock:
            return self.is_streaming
    
    async def start_server(self):
        """Start the phone stream server with SSL support."""
        logger.info(f"Starting phone camera stream server on port {self.port}")
        
        async def handler(websocket):
            await self.handle_phone_stream(websocket)
        
        # Try to start with SSL first, fallback to regular WebSocket
        try:
            import ssl
            import os
            
            # Look for SSL certificates
            cert_file = os.path.join(os.path.dirname(__file__), '..', 'cert.pem')
            key_file = os.path.join(os.path.dirname(__file__), '..', 'key.pem')
            
            if os.path.exists(cert_file) and os.path.exists(key_file):
                # Create SSL context
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(cert_file, key_file)
                
                logger.info(f"Starting secure WebSocket server with SSL on port {self.port}")
                server = await websockets.serve(
                    handler,
                    "0.0.0.0",
                    self.port,
                    ssl=ssl_context,
                    ping_interval=20,
                    ping_timeout=10
                )
            else:
                raise FileNotFoundError("SSL certificates not found")
                
        except (FileNotFoundError, ImportError) as e:
            logger.warning(f"SSL setup failed: {e}. Starting regular WebSocket server.")
            server = await websockets.serve(
                handler,
                "0.0.0.0",
                self.port,
                ping_interval=20,
                ping_timeout=10
            )
        
        logger.info(f"Phone camera stream server running on port {self.port}")
        await server.wait_closed()

# Global phone stream receiver
_phone_stream = PhoneStreamReceiver(8766)

def get_phone_stream_receiver() -> PhoneStreamReceiver:
    """Get the global phone stream receiver."""
    return _phone_stream

async def start_phone_stream_server(port: int = 8766):
    """Start the phone stream server."""
    global _phone_stream
    # Use the existing receiver instance (it should already have the correct port)
    await _phone_stream.start_server()

def start_phone_stream_server_thread(port: int = 8766) -> threading.Thread:
    """Start phone stream server in background thread."""
    def run_server():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(start_phone_stream_server(port))
        except KeyboardInterrupt:
            logger.info("Phone stream server stopped")
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Starting Phone Camera Stream Server...")
    print("Connect your phone to: ws://<laptop-ip>:8766")
    asyncio.run(start_phone_stream_server())

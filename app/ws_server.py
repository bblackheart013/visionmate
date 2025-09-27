#!/usr/bin/env python3
"""
WebSocket server for phone controller communication.

This module provides a WebSocket server that allows phone controllers
to send commands to the VisionMate application (start/stop/mute/repeat/set_goal).

Author: Person 3 (Integration & Snapdragon lead)
"""

import asyncio
import websockets
import json
import logging
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

logger = logging.getLogger(__name__)

@dataclass
class ControllerState:
    """Thread-safe controller state."""
    is_muted: bool = False
    should_stop: bool = False
    should_repeat: bool = False
    goal_str: str = ""
    last_utterance: str = ""
    last_command_time: float = 0.0
    connection_count: int = 0

# Global controller state (thread-safe with locks)
_controller_state = ControllerState()
_state_lock = threading.Lock()

def get_controller_state() -> Dict[str, Any]:
    """Get current controller state as dictionary."""
    with _state_lock:
        return asdict(_controller_state)

def update_controller_state(updates: Dict[str, Any]):
    """Update controller state with new values."""
    with _state_lock:
        for key, value in updates.items():
            if hasattr(_controller_state, key):
                setattr(_controller_state, key, value)
        _controller_state.last_command_time = time.time()

async def handle_client(websocket):
    """Handle individual WebSocket client connections."""
    client_ip = websocket.remote_address[0]
    logger.info(f"Controller connected from {client_ip}")
    
    # Increment connection count
    with _state_lock:
        _controller_state.connection_count += 1
    
    try:
        async for message in websocket:
            try:
                # Parse JSON message
                data = json.loads(message)
                cmd = data.get("cmd", "").lower()
                arg = data.get("arg", "")
                
                logger.info(f"Received command: {cmd} with arg: {arg}")
                
                # Process command
                response = await process_command(cmd, arg)
                
                # Send response back to client
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError:
                error_response = {
                    "status": "error",
                    "message": "Invalid JSON format"
                }
                await websocket.send(json.dumps(error_response))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                error_response = {
                    "status": "error", 
                    "message": str(e)
                }
                await websocket.send(json.dumps(error_response))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Controller disconnected from {client_ip}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Decrement connection count
        with _state_lock:
            _controller_state.connection_count -= 1

async def process_command(cmd: str, arg: str) -> Dict[str, Any]:
    """
    Process controller commands.
    
    Args:
        cmd: Command name (start, stop, repeat, mute, unmute, set_goal)
        arg: Command argument
        
    Returns:
        Response dictionary
    """
    current_time = time.time()
    
    if cmd == "start":
        update_controller_state({"should_stop": False})
        return {
            "status": "success",
            "message": "Application started",
            "timestamp": current_time
        }
    
    elif cmd == "stop":
        update_controller_state({"should_stop": True})
        return {
            "status": "success", 
            "message": "Stop command sent",
            "timestamp": current_time
        }
    
    elif cmd == "repeat":
        update_controller_state({"should_repeat": True})
        return {
            "status": "success",
            "message": "Repeat command sent",
            "timestamp": current_time
        }
    
    elif cmd == "mute":
        update_controller_state({"is_muted": True})
        return {
            "status": "success",
            "message": "Audio muted",
            "timestamp": current_time
        }
    
    elif cmd == "unmute":
        update_controller_state({"is_muted": False})
        return {
            "status": "success",
            "message": "Audio unmuted", 
            "timestamp": current_time
        }
    
    elif cmd == "set_goal":
        if not arg.strip():
            return {
                "status": "error",
                "message": "Goal cannot be empty",
                "timestamp": current_time
            }
        
        update_controller_state({"goal_str": arg.strip()})
        return {
            "status": "success",
            "message": f"Goal set to: {arg}",
            "timestamp": current_time
        }
    
    elif cmd == "status":
        # Return current state
        state = get_controller_state()
        return {
            "status": "success",
            "data": state,
            "timestamp": current_time
        }
    
    else:
        return {
            "status": "error",
            "message": f"Unknown command: {cmd}",
            "timestamp": current_time
        }

def start_websocket_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Start the WebSocket server in the current thread.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    # Run the server
    async def run_server():
        server = await websockets.serve(
            handle_client, 
            host, 
            port,
            ping_interval=20,
            ping_timeout=10
        )
        logger.info("WebSocket server running. Press Ctrl+C to stop.")
        await server.wait_closed()
    
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(run_server())
    except KeyboardInterrupt:
        logger.info("WebSocket server stopped")
    finally:
        loop.close()

def start_websocket_server_thread(host: str = "0.0.0.0", port: int = 8765) -> threading.Thread:
    """
    Start the WebSocket server in a background thread.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        
    Returns:
        Thread running the WebSocket server
    """
    def run_server():
        logger.info(f"Starting WebSocket server on {host}:{port}")
        
        async def async_server():
            server = await websockets.serve(
                handle_client, 
                host, 
                port,
                ping_interval=20,
                ping_timeout=10
            )
            logger.info("WebSocket server running. Press Ctrl+C to stop.")
            await server.wait_closed()
        
        # Create and run event loop in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(async_server())
        except KeyboardInterrupt:
            logger.info("WebSocket server stopped")
        finally:
            loop.close()
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread

# Test client for development
async def test_client():
    """Test client to verify WebSocket functionality."""
    uri = "ws://localhost:8765"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket server")
            
            # Test commands
            test_commands = [
                {"cmd": "status"},
                {"cmd": "set_goal", "arg": "cafeteria"},
                {"cmd": "mute"},
                {"cmd": "repeat"},
                {"cmd": "unmute"},
                {"cmd": "status"}
            ]
            
            for cmd_data in test_commands:
                await websocket.send(json.dumps(cmd_data))
                response = await websocket.recv()
                print(f"Sent: {cmd_data}")
                print(f"Received: {json.loads(response)}")
                await asyncio.sleep(0.5)
                
    except Exception as e:
        print(f"Test client error: {e}")

if __name__ == "__main__":
    # Run as standalone WebSocket server
    logging.basicConfig(level=logging.INFO)
    
    print("Starting WebSocket server for VisionMate controller...")
    print("Connect phone to: ws://<laptop-ip>:8765")
    print("Press Ctrl+C to stop")
    
    start_websocket_server()

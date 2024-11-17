# tests/test_websocket_server.py

import asyncio
import pytest
import websockets

@pytest.mark.asyncio
async def test_basic_connection():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        assert websocket.open, "WebSocket connection should be open"

@pytest.mark.asyncio
async def test_message_format():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        await websocket.send("x:0.5,y:0.5,z:0.5")
        response = await websocket.recv()
        assert response.startswith("Echo:"), "Response format incorrect"

@pytest.mark.asyncio
async def test_multiple_connections():
    uri = "ws://localhost:8080"
    connections = []
    for _ in range(10):  # Simulate 10 simultaneous connections
        websocket = await websockets.connect(uri)
        connections.append(websocket)
        assert websocket.open, f"Connection {_} should be open"

    # Close all connections
    await asyncio.gather(*(ws.close() for ws in connections))

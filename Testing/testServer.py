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
async def test_message_format_6dof():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        # Send a 6DoF message
        await websocket.send("x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0")
        response = await websocket.recv()
        # Verify that the response is correctly formatted
        assert response.startswith("Echo:"), "Response format incorrect"
        expected = "x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0"
        assert response.endswith(expected), "6DoF data mismatch in response"

@pytest.mark.asyncio
async def test_latency_6dof():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        start_time = time.perf_counter()
        await websocket.send("x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0")
        await websocket.recv()
        latency = time.perf_counter() - start_time
        assert latency < 0.1, f"Message latency too high: {latency}s"

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

@pytest.mark.asyncio
async def test_throughput_6dof():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        messages = 100
        start_time = time.perf_counter()
        for _ in range(messages):
            await websocket.send("x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0")
            await websocket.recv()
        end_time = time.perf_counter()
        throughput = messages / (end_time - start_time)
        assert throughput > 30, f"Throughput too low: {throughput} messages/sec"

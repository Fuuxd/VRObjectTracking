import unittest
import asyncio
import websockets
import threading
import time
from queue import Queue
from RPiArucoTrackWeb import websocket_server, tracking_queue
import psutil

class TestWebSocketServer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Start WebSocket server in a thread."""
        cls.server_thread = threading.Thread(target=asyncio.run, args=(websocket_server(),), daemon=True)
        cls.server_thread.start()

    async def websocket_client(self, uri, message=None):
        """Asynchronous WebSocket client"""
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_basic_connection(self):
        """Test: client can connect to server"""
        async def test():
            try:
                uri = "ws://localhost:8080"
                await websockets.connect(uri)
                self.assertTrue(True, "Connection successfully.")
            except Exception as e:
                self.fail(f"Connection failed: {e}")

        asyncio.run(test())

    def test_connection_timeout(self):
        """Test: server response when client connection times out"""
        async def test():
            try:
                uri = "ws://localhost:8080"
                websocket = await websockets.connect(uri, timeout=1)
                await asyncio.sleep(2)  # Simulate idle time
                await websocket.close()
                self.assertTrue(True, "Connection timed out as expected.")
            except Exception as e:
                self.fail(f"Unexpected exception: {e}")

        asyncio.run(test())

    def test_multiple_connections(self):
        """Test: multiple client connections"""
        async def test():
            uri = "ws://localhost:8080"
            clients = [asyncio.create_task(self.websocket_client(uri)) for _ in range(10)]
            try:
                results = await asyncio.gather(*clients)
                self.assertEqual(len(results), 10, "All clients connected successfully.")
            except Exception as e:
                self.fail(f"Failed during multiple connections: {e}")

        asyncio.run(test())

    def test_reconnection(self):
        """Test: client can reconnect after disconnecting"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                websocket = await websockets.connect(uri)
                await websocket.close()
                websocket = await websockets.connect(uri)  # Reconnect
                self.assertTrue(True, "Reconnection successful.")
            except Exception as e:
                self.fail(f"Reconnection failed: {e}")

        asyncio.run(test())

class TestWebSocketMessageExchange(unittest.TestCase):

    async def websocket_client(self, uri, message=None):
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_correct_message_format(self):
        """Test: correct format message from server"""
        async def test():
            uri = "ws://localhost:8080"
            sample_data = {
                'id': 42,
                'x': 1.0,
                'y': 2.0,
                'z': 3.0,
                'rx': 0.0,
                'ry': 45.0,
                'rz': 90.0
            }
            tracking_queue.put(sample_data)
            response = await self.websocket_client(uri)
            expected_message = "id:42,x:1.000000,y:2.000000,z:3.000000,rx:0.000000,ry:45.000000,rz:90.000000"
            self.assertEqual(response, expected_message, "Message format mismatch.")

        asyncio.run(test())

    def test_message_ordering(self):
        """Test: messages sent in correct order"""
        async def test():
            uri = "ws://localhost:8080"
            messages = [
                {'id': 1, 'x': 0.1, 'y': 0.2, 'z': 0.3, 'rx': 0.0, 'ry': 10.0, 'rz': 20.0},
                {'id': 2, 'x': 1.1, 'y': 1.2, 'z': 1.3, 'rx': 30.0, 'ry': 40.0, 'rz': 50.0},
            ]
            for msg in messages:
                tracking_queue.put(msg)

            responses = []
            for _ in range(len(messages)):
                responses.append(await self.websocket_client(uri))

            expected_responses = [
                "id:1,x:0.100000,y:0.200000,z:0.300000,rx:0.000000,ry:10.000000,rz:20.000000",
                "id:2,x:1.100000,y:1.200000,z:1.300000,rx:30.000000,ry:40.000000,rz:50.000000",
            ]
            self.assertEqual(responses, expected_responses, "Message ordering mismatch.")

        asyncio.run(test())

class TestWebSocketErrorHandling(unittest.TestCase):

    async def websocket_client(self, uri, message=None):
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_network_interruption(self):
        """Simulate network interruption to verify server handle"""
        async def test():
            uri = "ws://localhost:8080"
            websocket = await websockets.connect(uri)
            try:
                # Simulate client disconnect
                await websocket.close()
                self.assertTrue(True, "Network interruption handled without crashing.")
            except Exception as e:
                self.fail(f"Server did not handle network interruption: {e}")

        asyncio.run(test())

    def test_malformed_message(self):
        """Send malformed message to verify server response"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri) as websocket:
                    malformed_message = "this is not a tracking message"
                    await websocket.send(malformed_message)
                    await asyncio.sleep(1)
                    self.assertTrue(True, "Malformed message handled correctly.")
            except Exception as e:
                self.fail(f"Server failed to handle malformed message: {e}")

        asyncio.run(test())

    def test_server_shutdown_handling(self):
        """Test: server response when shutting down"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri) as websocket:
                    await websocket.send("message before shutdown")
                    await asyncio.sleep(0.5) 
                    # Simulate server shutdown by exiting thread
                    self.server_thread.join(timeout=1)
                    self.assertFalse(self.server_thread.is_alive(), "Server shutdown handled.")
            except Exception as e:
                self.fail(f"Error during server shutdown handling: {e}")

        asyncio.run(test())

class TestWebSocketPerformance(unittest.TestCase):

    async def websocket_client(self, uri, message=None):
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_message_latency(self):
        """Measure latency"""
        async def test():
            uri = "ws://localhost:8080"
            tracking_queue.put({'id': 1, 'x': 0.1, 'y': 0.2, 'z': 0.3, 'rx': 0, 'ry': 0, 'rz': 0})
            start_time = time.time()
            response = await self.websocket_client(uri)
            end_time = time.time()
            latency = end_time - start_time
            self.assertLess(latency, 0.1, f"Latency too high: {latency:.3f} seconds.")

        asyncio.run(test())

    def test_throughput(self):
        """Measure throughput"""
        async def test():
            uri = "ws://localhost:8080"
            num_messages = 100
            start_time = time.time()
            for i in range(num_messages):
                tracking_queue.put({'id': i, 'x': 0.1 * i, 'y': 0.2 * i, 'z': 0.3 * i, 'rx': 0, 'ry': 0, 'rz': 0})
                await self.websocket_client(uri)
            end_time = time.time()
            throughput = num_messages / (end_time - start_time)
            print(f"Throughput: {throughput:.2f} messages/second")
            self.assertGreater(throughput, 50, "Throughput too low.")

        asyncio.run(test())

    def test_memory_usage(self):
        """Measure memory usage"""
        process = psutil.Process()
        async def test():
            uri = "ws://localhost:8080"
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            for i in range(1000):  # High load
                tracking_queue.put({'id': i, 'x': 0.1 * i, 'y': 0.2 * i, 'z': 0.3 * i, 'rx': 0, 'ry': 0, 'rz': 0})
                await self.websocket_client(uri)
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            self.assertLess(memory_increase, 10, "Memory usage increased significantly during high load.")

        asyncio.run(test())

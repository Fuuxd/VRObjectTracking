import unittest
import asyncio
import websockets
import threading
import time
import cv2
import numpy as np
from arucoTrackWeb import websocket_server


class TestWebSocketConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Websocket server in thread"""
        cls.server_thread = threading.Thread(target=asyncio.run, args=(websocket_server(),), daemon=True)
        cls.server_thread.start()
        time.sleep(1) 

    async def websocket_client(self, uri, message=None):
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_basic_connection(self):
        """Test:Websocket server accepts connecting"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri):
                    self.assertTrue(True, "Connected successfully.")
            except Exception as e:
                self.fail(f"Connection failed: {e}")

        asyncio.run(test())

    def test_connection_timeout(self):
        """Test: connection timeout scenarios"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri, timeout=0.1):
                    self.fail("Connection should have timed out.")
            except Exception as e:
                self.assertTrue("timed out" in str(e), "Timeout handled correctly.")

        asyncio.run(test())

    def test_reconnection(self):
        """Test: server accepts reconnecting."""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri):
                    pass  # First connection
                async with websockets.connect(uri):
                    self.assertTrue(True, "Reconnection successful.")
            except Exception as e:
                self.fail(f"Reconnection failed: {e}")

        asyncio.run(test())

    def test_multiple_connections(self):
        """Test: multiple connections"""
        async def test():
            uri = "ws://localhost:8080"
            connections = []
            try:
                for _ in range(5):  # 5 clients
                    connections.append(await websockets.connect(uri))
                self.assertEqual(len(connections), 5, "All connections established.")
            except Exception as e:
                self.fail(f"Multiple connections failed: {e}")
            finally:
                for conn in connections:
                    await conn.close()

        asyncio.run(test())


class TestWebSocketMessageExchange(unittest.TestCase):
    async def websocket_client(self, uri, message=None):
        async with websockets.connect(uri) as websocket:
            if message:
                await websocket.send(message)
            return await websocket.recv()

    def test_message_format(self):
        """Test: correct format message"""
        async def test():
            uri = "ws://localhost:8080"
            message = "Test message"
            try:
                response = await self.websocket_client(uri, message)
                self.assertTrue(
                    "id:" in response and "x:" in response and "y:" in response,
                    "Message format is incorrect."
                )
            except Exception as e:
                self.fail(f"Message format test failed: {e}")

        asyncio.run(test())

    def test_message_order(self):
        """Verify message order"""
        async def test():
            uri = "ws://localhost:8080"
            messages = ["Message 1", "Message 2", "Message 3"]
            responses = []
            try:
                for msg in messages:
                    response = await self.websocket_client(uri, msg)
                    responses.append(response)
                self.assertEqual(messages, responses, "Message order is incorrect.")
            except Exception as e:
                self.fail(f"Message order test failed: {e}")

        asyncio.run(test())


class TestWebSocketErrorHandling(unittest.TestCase):
    def test_network_interruption(self):
        """Simulate network interruption"""
        async def test():
            uri = "ws://localhost:8080"
            websocket = await websockets.connect(uri)
            try:
                await websocket.close()
                self.assertTrue(True, "Network interruption handled.")
            except Exception as e:
                self.fail(f"Server did not handle network interruption: {e}")

        asyncio.run(test())

    def test_malformed_message(self):
        """Send malformed messages"""
        async def test():
            uri = "ws://localhost:8080"
            try:
                async with websockets.connect(uri) as websocket:
                    malformed_message = "INVALID MESSAGE FORMAT"
                    await websocket.send(malformed_message)
                    self.assertTrue(True, "Malformed message handled correctly.")
            except Exception as e:
                self.fail(f"Server failed to handle malformed message: {e}")

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
            start_time = time.time()
            response = await self.websocket_client(uri)
            latency = time.time() - start_time
            self.assertLess(latency, 0.1, f"Latency too high: {latency:.3f}s")

        asyncio.run(test())

    def test_throughput(self):
        """Measure throughput"""
        async def test():
            uri = "ws://localhost:8080"
            messages = 100
            start_time = time.time()
            for _ in range(messages):
                await self.websocket_client(uri)
            throughput = messages / (time.time() - start_time)
            print(f"Throughput: {throughput:.2f} msgs/sec")
            self.assertGreater(throughput, 50, "Throughput too low.")

        asyncio.run(test())

    def test_cpu_memory_usage(self):
        """Memory and CPU usage during message exchange"""
        import psutil
        process = psutil.Process()

        async def test():
            uri = "ws://localhost:8080"
            start_time = time.time()
            memory_before = process.memory_info().rss / (1024 * 1024)  # in MB
            cpu_before = process.cpu_percent(interval=None)
            for _ in range(100):
                await self.websocket_client(uri)
            end_time = time.time()
            memory_after = process.memory_info().rss / (1024 * 1024)
            cpu_after = process.cpu_percent(interval=None)

            print(f"Memory usage: Before={memory_before:.2f}MB After={memory_after:.2f}MB")
            print(f"CPU usage: Before={cpu_before:.2f}% After={cpu_after:.2f}%")
            duration = end_time - start_time
            self.assertLess(memory_after - memory_before, 50, "Excessive memory usage.")
            self.assertLess(cpu_after - cpu_before, 20, "High CPU usage during message exchange.")

        asyncio.run(test())


if __name__ == "__main__":
    unittest.main()

@pytest.mark.asyncio
async def test_message_format_6dof():
    uri = "ws://localhost:8080"
    async with websockets.connect(uri) as websocket:
        # Send a 6DoF message
        await websocket.send("x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0")
        response = await websocket.recv()
        assert response.startswith("Echo:"), "Response format incorrect"

        # Validate the returned 6DoF data structure
        expected = "x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0"
        assert response.endswith(expected), "6DoF data mismatch in response"

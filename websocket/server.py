import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        print(f"Received: {message}")
        await websocket.send(f"Echo: {message}")

async def send_data(websocket, path):
    count = 0.5
    while True:
        message = f"x:{count:.6f},y:{count:.6f},z:{count:.6f}"  # Format to 6 decimal places
        print(f"Sending: {message}")
        await websocket.send(message)  # Send message to the client
        count += 0.001
        await asyncio.sleep(0.0303)  # simulates 33Hz




# Start the server on localhost:8080
start_server = websockets.serve(send_data, "localhost", 8080) #IF RECIEVING IS WANTED CHANGE send_data TO echo


# Run the server event loop
asyncio.get_event_loop().run_until_complete(start_server)
print("WebSocket server started on ws://localhost:8080")
asyncio.get_event_loop().run_forever()
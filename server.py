import asyncio
import websockets
import math

# WebSocket server configuration
HOST = "localhost"
PORT = 8080

# Duration for each phase in seconds
ID_1_INITIAL_DURATION = 5  # Send ID 1 messages for 5 seconds initially
ID_0_DURATION = 5          # Then send ID 0 messages for 5 seconds
# After that, continue sending ID 1 messages indefinitely

# Function to simulate a moving object's position (ID 1), rotating around the Z-axis
def generate_moving_position(t):
    radius = 5.0  # Increased radius for larger movement
    x = 5 + radius * math.cos(t)  # Shifted to center around (5, 5, 5)
    y = 5 + radius * math.sin(t)  # Shifted to center around (5, 5, 5)
    z = 5 + radius * math.sin(t / 2)  # Amplified Z-axis movement
    return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

# Function to simulate user position (ID 0)
def generate_user_position(t):
    radius = 2.0  # Larger movement for testing offset
    x = 10 + radius * math.cos(t / 2)  # Shifted to center around (10, 10, 10)
    y = 10 + radius * math.sin(t / 2)  # Shifted to center around (10, 10, 10)
    z = 10  # Fixed Z-axis value
    return {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}

# Function to create WebSocket messages
def create_message(id, t):
    if id == 0:
        # User position (ID 0)
        user_position = generate_user_position(t)
        return f"id:0,x:{user_position['x']},y:{user_position['y']},z:{user_position['z']},rx:0,ry:0,rz:0"
    elif id == 1:
        # Object position (ID 1)
        object_position = generate_moving_position(t)
        return f"id:1,x:{object_position['x']},y:{object_position['y']},z:{object_position['z']},rx:{t*10:.2f},ry:{t*10:.2f},rz:{t*10:.2f}"

# WebSocket handler
async def handle_connection(websocket, path):
    print(f"Client connected: {path}")
    try:
        t = 0
        cycle_start_time = asyncio.get_event_loop().time()
        current_id = 1  # Start by sending ID 1 messages

        while True:
            current_time = asyncio.get_event_loop().time()
            elapsed_time = current_time - cycle_start_time

            # Phase 1: Send ID 1 messages for ID_1_INITIAL_DURATION seconds
            if current_id == 1 and elapsed_time >= ID_1_INITIAL_DURATION:
                current_id = 0
                cycle_start_time = current_time  # Reset cycle start time
                print("Switching to ID 0 messages")

            # Phase 2: Send ID 0 messages for ID_0_DURATION seconds
            elif current_id == 0 and elapsed_time >= ID_0_DURATION:
                current_id = 1
                cycle_start_time = current_time  # Reset cycle start time
                print("Switching to ID 1 messages indefinitely")

            # Phase 3: After initial cycles, continue sending ID 1 messages forever
            # We can set current_id to 1 and skip sending ID 0 again
            # To achieve this, once we've switched back to ID 1 after ID 0, we stop toggling
            # Introduce a flag to ensure we don't switch back to ID 0 again
            if current_id == 1 and elapsed_time >= ID_1_INITIAL_DURATION + ID_0_DURATION:
                # Once we've sent ID 1 and ID 0, switch to ID 1 forever
                current_id = 1
                cycle_start_time = current_time  # Reset cycle start time
                # No condition to switch back to ID 0
                print("Continuing with ID 1 messages only")

            # Generate and send message based on current_id
            message = create_message(current_id, t)
            await websocket.send(message)
            print(f"Sent: {message}")

            # Increment time
            t += 0.033  # Approximate time step for 30 Hz (1/30 seconds)
            await asyncio.sleep(1 / 30)  # 30 Hz update rate

    except websockets.ConnectionClosed as e:
        print(f"Client disconnected: {e}")

# Main server entry point
async def main():
    async with websockets.serve(handle_connection, HOST, PORT):
        print(f"WebSocket server running on ws://{HOST}:{PORT}")
        await asyncio.Future()  # Keep running

# Run the server
if __name__ == "__main__":
    asyncio.run(main())

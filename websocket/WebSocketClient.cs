using UnityEngine;
using System;
using System.Threading.Tasks;
using NativeWebSocket;

public class WebSocketClient : MonoBehaviour
{
    private WebSocket ws;
    private Vector3 receivedPosition = Vector3.zero;
    private bool isConnected = false;

    // WebSocket server URL
    private string serverUrl = "ws://localhost:8080";

    async void Start()
    {
        await ConnectToServer();
    }

    async Task ConnectToServer()
    {
        ws = new WebSocket(serverUrl);

        ws.OnOpen += () =>
        {
            Debug.Log("Connected to WebSocket server");
            isConnected = true;
        };

        ws.OnError += (e) =>
        {
            Debug.LogError("WebSocket Error: " + e);
        };

        ws.OnClose += (e) =>
        {
            Debug.Log("WebSocket connection closed");
            isConnected = false;
        };

        ws.OnMessage += (bytes) =>
        {
            // Get the message as string
            string message = System.Text.Encoding.UTF8.GetString(bytes);
            ParseMessage(message);
        };

        // Connect to the server
        await ws.Connect();
    }

    void ParseMessage(string message)
    {
        try
        {
            // Split the message into components
            // Expected format: "x:0.500000,y:0.500000,z:0.500000"
            string[] components = message.Split(',');

            if (components.Length == 3)
            {
                float x = float.Parse(components[0].Split(':')[1]);
                float y = float.Parse(components[1].Split(':')[1]);
                float z = float.Parse(components[2].Split(':')[1]);

                // Update the position vector
                receivedPosition = new Vector3(x, y, z);

                // Log the received position
                Debug.Log($"Received position: {receivedPosition}");
                this.transform.position = receivedPosition;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error parsing message: {e.Message}");
        }
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (ws != null)
        {
            // Dispatch WebSocket events, except for WebGL builds
            ws.DispatchMessageQueue();
        }
#endif

        // You can use receivedPosition here to update game objects
        // For example:
        // transform.position = receivedPosition;
    }

    private async void OnApplicationQuit()
    {
        if (ws != null && isConnected)
        {
            await ws.Close();
        }
    }

    private async void OnDisable()
    {
        if (ws != null && isConnected)
        {
            await ws.Close();
        }
    }
}
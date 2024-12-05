using UnityEngine;
using System;
using System.Collections.Generic; // For Queue
using System.Threading.Tasks;
using NativeWebSocket;
using System.IO;

public class WebSocketClient : MonoBehaviour
{
    private WebSocket ws;
    private bool isConnected = false;
    // Kalman filter for smoothing position updates
    private KalmanFilterVector3 kalmanFilter;
    private Vector3 userPosition;
    private Vector3 offset = Vector3.zero;
    private string serverUrl = "ws://192.168.0.216:8080";
    private string logFilePath = "position_log.txt";
    private GameObject box;
    private GameObject user;
    private GameObject bowl;
    private long milliseconds;

    // Sliding window for position smoothing (holds last 12 position updates)
    private Queue<Vector3> positionQueue = new Queue<Vector3>();
    private const int MaxQueueSize = 12;

    // Sliding window for rotation smoothing (holds last 12 rotation updates)
    private Queue<Vector3> rotationQueue = new Queue<Vector3>();
    
    async void Start()
    {
        Debug.Log($"Script started");
        box = GameObject.Find("BOX1");
        if (box == null)
        {
            Debug.LogError("BOX1 not found in the scene. Please check the object name.");
        }
        user = GameObject.Find("player");
        if (user == null)
        {
            Debug.LogError("player not found in the scene. Please check the object name.");
        }
        bowl = GameObject.Find("bowl");
        if (bowl == null)
        {
            Debug.LogError("bowl not found in the scene. Please check the object name.");
        }

        // Initialize the Kalman filter
        kalmanFilter = new KalmanFilterVector3();
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
            Debug.LogError($"WebSocket Error: {e}");
        };

        ws.OnClose += (closeCode) =>
        {
            Debug.Log($"WebSocket connection closed with close code: {closeCode}");
            isConnected = false;
        };

        ws.OnMessage += (bytes) =>
        {
            string message = System.Text.Encoding.UTF8.GetString(bytes);
            ParseMessage(message);
        };

        await ws.Connect();
    }

    private readonly object lockObject = new object();
    void ParseMessage(string message)
    {
        lock (lockObject)
        {
            try
            {
                string[] components = message.Split(',');

                // Assuming data format includes ID at components[0] like "id:1,x:1.0,y:2.0,z:3.0"
                int id = int.Parse(components[0].Split(':')[1]);

                float x = float.Parse(components[1].Split(':')[1]);
                float y = float.Parse(components[2].Split(':')[1]);
                float z = float.Parse(components[3].Split(':')[1]);
                float rx = float.Parse(components[3].Split(':')[1]);
                float ry = float.Parse(components[4].Split(':')[1]);
                float rz = float.Parse(components[5].Split(':')[1]);

                Vector3 position = new Vector3(x, y, z);
                Vector3 rotation = new Vector3(rx, ry, rz);

                // Sliding window filter for position (store and average last 12 positions)
                if (positionQueue.Count >= MaxQueueSize)
                {
                    positionQueue.Dequeue(); // Remove the oldest value
                }
                positionQueue.Enqueue(position); // Add the new position

                // Calculate the average position from the sliding window
                Vector3 averagePosition = CalculateAveragePosition(positionQueue);

                // Sliding window filter for rotation (store and average last 12 rotations)
                if (rotationQueue.Count >= MaxQueueSize)
                {
                    rotationQueue.Dequeue(); // Remove the oldest value
                }
                rotationQueue.Enqueue(rotation); // Add the new rotation

                // Calculate the average rotation from the sliding window
                Vector3 averageRotation = CalculateAveragePosition(rotationQueue);

                if (id == 0) // 0 == user
                {
                    // Find offset between real-world position and Unity camera position
                    Vector3 cameraPosition = Camera.main.transform.position;
                    offset = position - cameraPosition;

                    Debug.Log($"User Position: {position}, Camera Position: {cameraPosition}, Offset: {offset}");
                }
                else if (id == 1) // 1 == box
                {
                    Vector3 truePosition = averagePosition - offset;
                    UpdateObjectPosition(truePosition, averageRotation, box);
                }
                else if (id >= 2 && id <= 7 && ry >= 0)
                {
                    Vector3 truePosition = averagePosition - offset;

                    int trackerIndex = id - 2;
                    Quaternion cubeRotation = CalculateCubeRotation(trackerIndex, averageRotation);

                    box.transform.position = truePosition;
                    box.transform.rotation = cubeRotation;

                    Debug.Log($"ID: {id}");
                    Debug.Log($"Updated BOX1 to rotation: {cubeRotation.eulerAngles}");
                }
                else
                {
                    Debug.Log($"Received message for ID {id}, but this object is not assigned to this ID.");
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Error parsing message: {e.Message}");
            }
        }
    }

    /// <summary>
    /// Calculate the average position from the sliding window queue
    /// </summary>
    /// <param name="queue">Queue containing the last n position updates</param>
    /// <returns>Average position</returns>
    Vector3 CalculateAveragePosition(Queue<Vector3> queue)
    {
        Vector3 sum = Vector3.zero;
        foreach (var position in queue)
        {
            sum += position;
        }
        return sum / queue.Count;
    }

    /// <summary>
    /// Handles multiple arUco ids on a single cube, assumes the following indexes are placed
    /// in the following order 0 = Front, 1 = Back, 2 = Top, 3 = Bottom, 4 = Left, 5= Right
    /// </summary>
    /// <param name="trackerIndex"> integer between 0-6 </param>
    /// <param name="rotation"> Vector3 with rotation of trackerIndex as given by </param>
    /// <returns></returns>
    Quaternion CalculateCubeRotation(int trackerIndex, Vector3 rotation)
    {
        switch (trackerIndex)
        {
        case 0:
            // Do nothing
            break;
        
        case 1:
            rotation.y = rotation.y + 180;
            break;

        case 2:
            rotation.x = rotation.x + 90;
            break;

        case 3:
            rotation.x = rotation.x + 90;
            break;

        case 4:
            rotation.y = rotation.y - 90;
            break;

        case 5:
            rotation.y = rotation.y - 90;
            break; 
        default:
            Debug.LogError("Invalid trackerIndex sent to CalculateCubeRotation.");
            break;
        }

        return Quaternion.Euler(rotation);
    }

    void UpdateObjectPosition(Vector3 position, Vector3 rotation, GameObject obj)
    {
        // Apply the new position directly to the object
        obj.transform.position = position;
        obj.transform.rotation = Quaternion.Euler(rotation); // Use rotation here
        milliseconds = DateTimeOffset.Now.ToUnixTimeMilliseconds();
        Debug.Log($"Time: {milliseconds}, Object {obj.gameObject.name} updated to position: {obj.transform.position} and rotation: {obj.transform.rotation}");
    }

    void LogPositionToFile(Vector3 position, Vector3 rotation)
    {
        try
        {
            // Prepare the log entry as a string
            string logEntry = $"{position.x:F2}, {position.y:F2}, {position.z:F2}, {rotation.x:F2}, {rotation.y:F2}, {rotation.z:F2}\n";

            // Write to file (append mode)
            File.AppendAllText(logFilePath, logEntry);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error logging position to file: {e.Message}");
        }
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        if (ws != null)
        {
            ws.DispatchMessageQueue();
        }
#endif
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

using UnityEngine;
using System;
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
    //private string serverUrl = "ws://192.168.0.216:8080";
    private string serverUrl = "ws://localhost:8080";
    private string logFilePath = "position_log.txt";
    private GameObject object1;
    private GameObject object2;
    private Vector3[] trackerPositions = new Vector3[6];
    private Vector3[] trackerRotations = new Vector3[6];

    
    async void Start()
    {
        Debug.Log($"Script started");
        object1 = GameObject.Find("BOX1");
        if (object1 == null)
        {
            Debug.LogError("BOX1 not found in the scene. Please check the object name.");
        }
        object2 = GameObject.Find("player");
        if (object2 == null)
        {
            Debug.LogError("player not found in the scene. Please check the object name.");
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
                
                if(id == 0) // 0 == user
                {
                    // Find offset between real-world position and Unity camera position
                    Vector3 cameraPosition = Camera.main.transform.position;
                    offset = userPosition - cameraPosition;

                    Debug.Log($"User Position: {userPosition}, Camera Position: {cameraPosition}, Offset: {offset}");
                }
                else if (id == 1) // 1 == object1
                {
                    
                    Vector3 truePosition = position - offset;
                    // Apply Kalman filter to smooth position
                    //Vector3 filteredPosition = kalmanFilter.Update(position, newQ: 0.05f, newR: 0.005f);
                    // Update object position using the filtered position
                    UpdateObjectPosition(truePosition, rotation, object1);
                    //Vector3 filteredRotation = kalmanFilter.Update(rotation, newQ: 0.05f, newR: 0.005f);
                    object1.transform.rotation = Quaternion.Euler(rotation.x, rotation.y, rotation.z);
                    //LogPositionToFile(truePosition, rotation);
                }
                else if (id >= 2 && id <= 7)
                {
                    int trackerIndex = id - 2;
                    trackerPositions[trackerIndex] = new Vector3(x, y, z);
                    trackerRotations[trackerIndex] = new Vector3(rx, ry, rz);
                    Quaternion cubeRotation = CalculateCubeRotation();
                    object1.transform.rotation = cubeRotation;

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

    Quaternion CalculateCubeRotation()
    {
        if (trackerPositions.Length < 6)
        {
            Debug.LogError("Not all tracker positions are available.");
            return Quaternion.identity;
        }

        // Example: Use two orthogonal vectors to define the cube's orientation
        Vector3 upVector = (trackerPositions[4] - trackerPositions[5]).normalized;  // Top - Bottom
        Vector3 forwardVector = (trackerPositions[0] - trackerPositions[1]).normalized;  // Front - Back

        // Ensure orthogonality for precision
        Vector3 rightVector = Vector3.Cross(upVector, forwardVector).normalized;
        forwardVector = Vector3.Cross(rightVector, upVector).normalized;

        // Build rotation from vectors
        return Quaternion.LookRotation(forwardVector, upVector);
    }

    void UpdateObjectPosition(Vector3 position, Vector3 rotation, GameObject obj)
    {
        // Apply the new position directly to the object
        obj.transform.position = position;
        obj.transform.rotation = Quaternion.Euler(rotation.x, rotation.y, rotation.z);
        Debug.Log($"Object {obj.gameObject.name} updated to position: {obj.transform.position} and rotation: {obj.transform.rotation}");
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

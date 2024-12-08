using UnityEngine;
using System;
using System.Collections.Generic;  // For Queue
using System.Threading.Tasks;
using NativeWebSocket;
using System.IO;

public class WebSocketClient : MonoBehaviour {
  private WebSocket ws;
  private bool isConnected = false;
  // Kalman filter for smoothing position updates
  private KalmanFilterVector3 kalmanFilter;
  private Vector3 userPosition;
  private Quaternion userRotation;
  private Vector3 offset = Vector3.zero;
  private string serverUrl = "ws://192.168.0.216:8080";
  private string logFilePath = "position_log.txt";
  private GameObject box;
  private GameObject user;
  private GameObject box2;
  private long milliseconds;

  // Sliding window for position smoothing (holds last 12 position updates)
  private Queue<Vector3> position1Queue = new Queue<Vector3>();
  private const int MaxQueueSize = 12;
  private Queue<Vector3> position2Queue = new Queue<Vector3>();

  // Sliding window for rotation smoothing (holds last 12 rotation updates)
  // private Queue<Vector3> rotationQueue = new Queue<Vector3>();

  async void Start() {
    Debug.Log($"Script started");
    box = GameObject.Find("BOX1");
    if (box == null) {
      Debug.LogError(
          "BOX1 not found in the scene. Please check the object name.");
    }
    user = GameObject.Find("player");
    if (user == null) {
      Debug.LogError(
          "player not found in the scene. Please check the object name.");
    }
    box2 = GameObject.Find("BOX2");
    if (box2 == null) {
      Debug.LogError(
          "bowl not found in the scene. Please check the object name.");
    }

    // Initialize the Kalman filter
    kalmanFilter = new KalmanFilterVector3();

    offset.x = -0.20f;
    offset.y = -1.36144f;
    offset.z = 0.6f;

    await ConnectToServer();
  }

  async Task ConnectToServer() {
    ws = new WebSocket(serverUrl);

    ws.OnOpen += () => {
      Debug.Log("Connected to WebSocket server");
      isConnected = true;
    };

    ws.OnError += (e) => { Debug.LogError($"WebSocket Error: {e}"); };

    ws.OnClose += (closeCode) => {
      Debug.Log($"WebSocket connection closed with close code: {closeCode}");
      isConnected = false;
    };

    ws.OnMessage += (bytes) => {
      string message = System.Text.Encoding.UTF8.GetString(bytes);
      ParseMessage(message);
    };

    await ws.Connect();
  }

  // Define a buffer to hold the last 10 IDs for each object (IDs 2-7)
  private Dictionary<int, Queue<int>> idBuffers =
      new Dictionary<int, Queue<int>>();
  private const int ConsecutiveThreshold =
      20;  // Number of consecutive sightings required to switch to the new ID

  private readonly object lockObject = new object();
  void ParseMessage(string message) {
    lock (lockObject) {
      try {
        string[] components = message.Split(',');

        // Assuming data format includes ID at components[0] like
        // "id:1,x:1.0,y:2.0,z:3.0"
        int id = int.Parse(components[0].Split(':')[1]);

        float x = float.Parse(components[1].Split(':')[1]);
        float y = float.Parse(components[2].Split(':')[1]);
        float z = float.Parse(components[3].Split(':')[1]);
        float rx = float.Parse(components[4].Split(':')[1]);
        float ry = float.Parse(components[5].Split(':')[1]);
        float rz = float.Parse(components[6].Split(':')[1]);
        float rw = float.Parse(components[7].Split(':')[1]);

        Vector3 position = new Vector3(x, y, z);
        Quaternion rotation = new Quaternion(rx, ry, rz, rw);

        if (id == 0)  // 0 == user
        {
          // Find offset between real-world position and Unity camera position
          Vector3 cameraPosition = Camera.main.transform.position;
          // offset = position - cameraPosition;

          userRotation = rotation;

          Debug.Log(
              $"User Position: {position}, Camera Position: {cameraPosition}, Offset: {offset}");
        } else if (id == 1)  // 1 == box
        {
          // Sliding window filter for position (store and average last 12
          // positions)
          if (position1Queue.Count >= MaxQueueSize) {
            position1Queue.Dequeue();  // Remove the oldest value
          }
          position1Queue.Enqueue(position);  // Add the new position

          // Calculate the average position from the sliding window
          Vector3 averagePosition = CalculateAveragePosition(position1Queue);

          Vector3 truePosition = averagePosition - offset;
          UpdateObjectPosition(truePosition, rotation, box);

        } else if (id == 2) {
          // Sliding window filter for position (store and average last 12
          // positions)
          if (position2Queue.Count >= MaxQueueSize) {
            position2Queue.Dequeue();  // Remove the oldest value
          }
          position2Queue.Enqueue(position);  // Add the new position

          // Calculate the average position from the sliding window
          Vector3 averagePosition = CalculateAveragePosition(position2Queue);

          Vector3 truePosition = averagePosition - offset;
          UpdateObjectPosition(truePosition, rotation, box2);

        } else if (id >= 3 && id <= 8)  // For IDs 2-7 (trackable objects)
        {
          // Check if we have an existing buffer for this ID
          if (!idBuffers.ContainsKey(id)) {
            idBuffers[id] = new Queue<int>(
                new int[ConsecutiveThreshold]);  // Initialize with a buffer of
                                                 // size 20
          }

          // Update the buffer with the current ID
          idBuffers[id].Enqueue(id);
          if (idBuffers[id].Count > ConsecutiveThreshold) {
            idBuffers[id]
                .Dequeue();  // Remove the oldest ID to maintain the buffer size
          }

          // Check if the ID has been consistently seen in the last 10 messages
          int consecutiveCount = 0;
          foreach (var bufferedId in idBuffers[id]) {
            if (bufferedId == id) {
              consecutiveCount++;
            }
          }

          if (consecutiveCount >= ConsecutiveThreshold) {
            // Switch to the new ID
            Vector3 truePosition = position + offset;
            int trackerIndex = id - 2;
            Quaternion cubeRotation =
                CalculateCubeRotation(trackerIndex, rotation);

            // Update the object with the true position and calculated rotation
            UpdateObjectPosition(truePosition, rotation, box);
            Debug.Log(
                $"ID: {id} switched after {ConsecutiveThreshold} consecutive sightings.");
          } else {
            Debug.Log(
                $"ID: {id} not switched yet, only {consecutiveCount} consecutive sightings.");
          }
        } else {
          Debug.Log(
              $"Received message for ID {id}, but this object is not assigned to this ID.");
        }
      } catch (Exception e) {
        Debug.LogError($"Error parsing message: {e.Message}");
      }
    }
  }

  /// <summary>
  /// Calculate the average position from the sliding window queue
  /// </summary>
  /// <param name="queue">Queue containing the last n position updates</param>
  /// <returns>Average position</returns>
  Vector3 CalculateAveragePosition(Queue<Vector3> queue) {
    Vector3 sum = Vector3.zero;
    foreach (var position in queue) {
      sum += position;
    }
    return sum / queue.Count;
  }

  /// <summary>
  /// Handles multiple ArUco IDs on a single cube, assumes the following indexes
  /// are placed in the following order: 0 = Front, 1 = Back, 2 = Top, 3 =
  /// Bottom, 4 = Left, 5 = Right
  /// </summary>
  /// <param name="trackerIndex">Integer between 0-5.</param>
  /// <param name="rotation">Quaternion representing the rotation of the
  /// trackerIndex.</param> <returns>Adjusted Quaternion for the cube's
  /// rotation.</returns>
  Quaternion CalculateCubeRotation(int trackerIndex, Quaternion rotation) {
    // Define the adjustment rotations as quaternions
    Quaternion adjustment = Quaternion.identity;  // Default is no adjustment

    switch (trackerIndex) {
      case 0:
        // No adjustment needed for the front face
        adjustment = Quaternion.identity;
        break;

      case 1:
        // Rotate around the Y-axis by 180 degrees (flip front to back)
        adjustment = Quaternion.AngleAxis(180, Vector3.up);
        break;

      case 2:
        // Rotate around the X-axis by 90 degrees (top face)
        adjustment = Quaternion.AngleAxis(90, Vector3.right);
        break;

      case 3:
        // Rotate around the X-axis by -90 degrees (bottom face)
        adjustment = Quaternion.AngleAxis(-90, Vector3.right);
        break;

      case 4:
        // Rotate around the Y-axis by -90 degrees (left face)
        adjustment = Quaternion.AngleAxis(-90, Vector3.up);
        break;

      case 5:
        // Rotate around the Y-axis by 90 degrees (right face)
        adjustment = Quaternion.AngleAxis(90, Vector3.up);
        break;

      default:
        Debug.LogError("Invalid trackerIndex sent to CalculateCubeRotation.");
        break;
    }

    // Combine the adjustment quaternion with the input rotation quaternion
    return adjustment * rotation;
  }

  void UpdateObjectPosition(Vector3 position, Quaternion rotation,
                            GameObject obj) {
    // Apply the new position directly to the object
    obj.transform.position = position;
    // obj.transform.rotation = Quaternion.Slerp(obj.transform.rotation,
    // rotation.normalized, Time.deltaTime * 5.0f); // Use rotation here
    Vector3 eulerAngs = rotation.eulerAngles;
    eulerAngs.z = -eulerAngs.z;
    Quaternion fixedrotation = Quaternion.Euler(eulerAngs);
    obj.transform.rotation = fixedrotation;
    milliseconds = DateTimeOffset.Now.ToUnixTimeMilliseconds();
    Debug.Log(
        $"Time: {milliseconds}, Object {obj.gameObject.name} updated to position: {obj.transform.position} and rotation: {obj.transform.rotation}");
  }

  void LogPositionToFile(Vector3 position, Vector3 rotation) {
    try {
      // Prepare the log entry as a string
      string logEntry =
          $"{position.x:F2}, {position.y:F2}, {position.z:F2}, {rotation.x:F2}, {rotation.y:F2}, {rotation.z:F2}\n";

      // Write to file (append mode)
      File.AppendAllText(logFilePath, logEntry);
    } catch (Exception e) {
      Debug.LogError($"Error logging position to file: {e.Message}");
    }
  }

  void Update() {
#if !UNITY_WEBGL || UNITY_EDITOR
    if (ws != null) {
      ws.DispatchMessageQueue();
    }
#endif
  }

  private async void OnApplicationQuit() {
    if (ws != null && isConnected) {
      await ws.Close();
    }
  }

  private async void OnDisable() {
    if (ws != null && isConnected) {
      await ws.Close();
    }
  }
}
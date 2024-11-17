using UnityEngine.TestTools;
using NUnit.Framework;
using System.Collections;

public class WebSocketClientTests
{
    private WebSocketClient client;

    [SetUp]
    public void Setup()
    {
        GameObject go = new GameObject();
        client = go.AddComponent<WebSocketClient>();
    }
    
    [UnityTest]
    public IEnumerator TestBasicConnection()
    {
        yield return client.ConnectToServer(); // Connect to WebSocket
        Assert.IsTrue(client.IsConnected(), "Client should be connected to the server.");
    }
    
    [UnityTest]
    public IEnumerator TestReceivedMessageFormat6DoF()
    {
        yield return client.ConnectToServer();
        
        // Simulate receiving a 6DoF message
        client.ParseMessage("x:0.5,y:0.5,z:0.5,pitch:30.0,yaw:45.0,roll:60.0");
    
        // Expected position and rotation
        Vector3 expectedPosition = new Vector3(0.5f, 0.5f, 0.5f);
        Vector3 expectedRotation = new Vector3(30.0f, 45.0f, 60.0f); // pitch, yaw, roll
    
        // Assert position
        Assert.AreEqual(expectedPosition, client.GetReceivedPosition(), "Parsed position should match expected.");
    
        // Assert rotation
        Assert.AreEqual(expectedRotation, client.GetReceivedRotation(), "Parsed rotation should match expected.");
    }
}

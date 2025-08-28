using Microsoft.MixedReality.Toolkit;
using Newtonsoft.Json;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

public class PegVisualizer : MonoBehaviour
{
    public GameObject pegPrefab;
    public GameObject mainCamera;
    public float roiThreshold = 0.02f;

    private Dictionary<string, GameObject> pegs = new Dictionary<string, GameObject>();

    private string remoteIP = "127.0.0.1";
    private int remotePort = 9991;
    private int remotePort2 = 9992;
    private IPEndPoint remoteEndPoint;
    private UdpClient udpClient;
    private Thread receiveThread;
    private int listenPort = 9989;
    private string latestMessageData;
    private int pegOfInterest = 0;
    private int movingPegFromTracker = 0; // ADDITION: Store moving peg from Python


    private List<GazeEntry> gazeLog = new List<GazeEntry>();
    [Tooltip("Filename to export CSV and JSON (without extension)")]
    public string logFileName = "GazeLog_org_dir";
    private string csvPath;
    private string jsonPath;

    private Thread thread;
    System.Diagnostics.Process process;

    public Material gazeHitMaterial;
    public Material gazeMissMaterial;

    void Start()
    {
        RunMainPhaseScript();

        string folder = Application.persistentDataPath;
        Debug.Log($"Persistent Data Path: {folder}");
        csvPath = Path.Combine(folder, logFileName + ".csv");
        jsonPath = Path.Combine(folder, logFileName + ".json");
        WriteCSVHeader();
        Debug.Log($"Logging to:\nCSV: {csvPath}\nJSON: {jsonPath}");

        udpClient = new UdpClient(listenPort);
        remoteEndPoint = new IPEndPoint(IPAddress.Any, listenPort);
        receiveThread = new Thread(ReceiveData);
        receiveThread.IsBackground = true;
        receiveThread.Start();

        PegInitialization();
    }

    void Update()
    {
        if(Input.GetKeyDown(KeyCode.Space))
        {
            Debug.Log("Space key was pressed.");
            string Singal = "Signal";
            byte[] datas = Encoding.UTF8.GetBytes(Singal);
            udpClient.Send(datas, datas.Length, remoteIP, remotePort2);
        }

        ApplyPegPositions();

        var gazeProvider = CoreServices.InputSystem?.EyeGazeProvider;

        if (gazeProvider == null || latestMessageData == null) return;

        Vector3 origin = gazeProvider.GazeOrigin;
        Vector3 direction = gazeProvider.GazeDirection;

        Vector3 localGazeOrigin = mainCamera.transform.InverseTransformPoint(origin);
        Vector3 localGazeDirection = mainCamera.transform.InverseTransformDirection(direction);

        Dictionary<string, float> pegScores = new Dictionary<string, float>();
        float sigma = roiThreshold / 2f;
        float maxScore = -1f;
        string selectedPeg = "";

        foreach (var kvp in pegs)
        {
            string id = kvp.Key;
            Vector3 currentPegPos = kvp.Value.transform.position;
            (float score, float distance) = GazeProbability(currentPegPos, origin, direction, sigma);
            pegScores[id] = score;

            if (score > maxScore && distance < roiThreshold)
            {
                maxScore = score;
                selectedPeg = id;
            }
        }

        Dictionary<string, PegInfo> tmpInfo = new Dictionary<string, PegInfo>();
        foreach (var kvp in pegs)
        {
            string id = kvp.Key;
            Vector3 pegLocalPos = kvp.Value.transform.localPosition;

            tmpInfo[id] = new PegInfo
            {
                location = new Dictionary<string, float>
                {
                    { "x", pegLocalPos.x },
                    { "y", pegLocalPos.y },
                    { "z", pegLocalPos.z }
                },
                isHitByGaze = (id == selectedPeg)
            };
        }

        GazeEntry entry = new GazeEntry
        {
            timestamp = System.DateTime.UtcNow.ToString("o"),
            gaze_origin = new Dictionary<string, float>
            {
                { "x", localGazeOrigin.x },
                { "y", localGazeOrigin.y },
                { "z", localGazeOrigin.z }
            },
            gaze_direction = new Dictionary<string, float>
            {
                { "x", localGazeDirection.x },
                { "y", localGazeDirection.y },
                { "z", localGazeDirection.z }
            },
            peg = tmpInfo
        };

        PegHitByGazeCheck(entry);

        gazeLog.Add(entry);
        AppendCSV(entry);

        //Sending Data to GUI
        float[,] gazeLineArray = new float[2, 3]
        {
                { localGazeOrigin.x, localGazeOrigin.y, localGazeOrigin.z },
                { localGazeDirection.x, localGazeDirection.y, localGazeDirection.z }
        };

        float[,] pegLocalPositions = new float[6, 3];
        for (int i = 1; i <= 6; i++)
        {
            Vector3 pegLocalPos = pegs[i.ToString()].transform.localPosition;
            pegLocalPositions[i - 1, 0] = pegLocalPos.x;
            pegLocalPositions[i - 1, 1] = pegLocalPos.y;
            pegLocalPositions[i - 1, 2] = pegLocalPos.z;
        }

        // Use moving peg from tracker if available, otherwise use pegOfInterest
        int targetPeg = movingPegFromTracker > 0 ? movingPegFromTracker - 1 : pegOfInterest;
        bool intercept = (selectedPeg == (targetPeg + 1).ToString());
        
        // Calculate closest point on gaze line to target peg
        Vector3 pegPos = pegs[(targetPeg + 1).ToString()].transform.position;
        Vector3 closestPointOnGaze = ClosestPointOnLine(origin, origin + direction, pegPos);
        float gazeDistance = Vector3.Distance(closestPointOnGaze, pegPos) * 1000f; // Convert to millimeters
        
        var dict = new Dictionary<string, object>
        {
            { "pegs", pegLocalPositions},
            { "gaze_line", gazeLineArray },
            { "intercept", intercept ? 1 : 0 },
            { "gaze_distance", gazeDistance },
            { "moving_peg", pegOfInterest + 1 }  // ADD THIS - send which peg we're tracking
        };

        string dictString = JsonConvert.SerializeObject(dict);
        string message = 'G' + dictString;
        byte[] data = Encoding.UTF8.GetBytes(message);
        udpClient.Send(data, data.Length, remoteIP, remotePort);

        if (Input.GetKeyDown(KeyCode.Return))  // or KeyCode.Space
        {
            byte[] unlockMessage = Encoding.UTF8.GetBytes("U");
            udpClient.Send(unlockMessage, unlockMessage.Length, remoteIP, remotePort2);
            Debug.Log("[Unity] Sent unlock command to Python.");
        }
    }

    private void PegHitByGazeCheck(GazeEntry entry)
    {
        for (int i = 1; i <= 6; i++)
        {
            if (entry.peg[i.ToString()].isHitByGaze == true)
            {
                pegs[i.ToString()].GetComponent<Renderer>().material = gazeHitMaterial;
            }
            else
            {
                pegs[i.ToString()].GetComponent<Renderer>().material = gazeMissMaterial;
            }
        }
    }

    public void HideTransformedPeg()
    {
        for (int i = 1; i <= 6; i++)
        {
            pegs[i.ToString()].GetComponent<Renderer>().enabled = false;
        }
    }

    public void ShowTransformedPeg()
    {
        for (int i = 1; i <= 6; i++)
        {
            pegs[i.ToString()].GetComponent<Renderer>().enabled = true;
        }
    }

    // Helper method to find closest point on a line (defined by two points) to a point
    private Vector3 ClosestPointOnLine(Vector3 lineStart, Vector3 lineEnd, Vector3 point)
    {
        Vector3 lineDirection = (lineEnd - lineStart).normalized;
        float lineLength = Vector3.Distance(lineStart, lineEnd);
        float projection = Vector3.Dot(point - lineStart, lineDirection);
        projection = Mathf.Clamp(projection, 0f, lineLength);
        return lineStart + lineDirection * projection;
    }

    private float DistancePointToRay(Vector3 point, Vector3 rayOrigin, Vector3 rayDir)
    {
        Vector3 toPoint = point - rayOrigin;
        Vector3 projection = Vector3.Dot(toPoint, rayDir) * rayDir;
        Vector3 closestPoint = rayOrigin + projection;
        return Vector3.Distance(point, closestPoint);
    }

    private  (float probability, float distance) GazeProbability(Vector3 pegPos, Vector3 origin, Vector3 direction, float sigma)
    {
        float d = DistancePointToRay(pegPos, origin, direction);
        float probability = Mathf.Exp(-Mathf.Pow(d, 2) / (2 * Mathf.Pow(sigma, 2)));
        return (probability, d);
    }
    
    private void ReceiveData()
    {
        while (true)
        {
            byte[] data = udpClient.Receive(ref remoteEndPoint);
            string jsonString = Encoding.UTF8.GetString(data);

            if (jsonString.StartsWith("P"))
            {
                if (int.TryParse(jsonString.Substring(1), out int pegIndex) && pegIndex >= 1 && pegIndex <= 6)
                {
                    pegOfInterest = pegIndex - 1;  
                    Debug.Log($"[Unity] Peg of interest updated to: {pegOfInterest + 1}");
                }
            }
            else if (jsonString.StartsWith("M"))
            {
                if (int.TryParse(jsonString.Substring(1), out int movingPeg))
                {
                    // 1..6 means a peg is moving; 0 means "none"
                    movingPegFromTracker = movingPeg;   // <— actually store it

                    // Optional: also auto-focus pegOfInterest when a mover is announced
                    if (movingPeg > 0)
                        pegOfInterest = movingPeg - 1;

                    // Console sanity check (prints only on change)
                    if (!PlayerPrefs.HasKey("lastMovingPeg")) PlayerPrefs.SetInt("lastMovingPeg", -1);
                    int last = PlayerPrefs.GetInt("lastMovingPeg");
                    if (last != movingPeg)
                    {
                        Debug.Log($"[Unity] Moving peg (from Python): {(movingPeg == 0 ? "None" : movingPeg.ToString())}");
                        PlayerPrefs.SetInt("lastMovingPeg", movingPeg);
                    }
                }
            }
            else
            {
                latestMessageData = jsonString;
            }
        }
    }

    private void ApplyPegPositions()
    {
        if (string.IsNullOrEmpty(latestMessageData))
        {
            return;
        }

        try
        {
            var pegDict = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, float>>>(latestMessageData);

            for (int i = 1; i <= 6; i++)
            {
                if (pegDict.ContainsKey(i.ToString()))
                {
                    pegs[i.ToString()].transform.localPosition = new Vector3(
                        pegDict[i.ToString()]["x"] / 1000,
                        pegDict[i.ToString()]["y"] / 1000,
                        pegDict[i.ToString()]["z"] / 1000
                    );
                }
            }
        }
        catch { }
    }

    private void PegInitialization()
    {
        for (int i = 1; i <= 6; i++)
        {
            pegs.Add(i.ToString(), Instantiate(pegPrefab, new Vector3(0f, 0f, 0.0f), Quaternion.identity, mainCamera.transform));
        }
    }

    private void RunMainPhaseScript()
    {
        string pythonExe = "C:\\Users\\kiansadat\\AppData\\Local\\anaconda3\\python.exe";
        string scriptPath = "C:\\Users\\kiansadat\\Desktop\\Azimi Project\\3D Eye Gaze\\Assets\\Scripts\\peg_map.py";

        thread = new System.Threading.Thread(() =>
        {
            System.Diagnostics.ProcessStartInfo start = new System.Diagnostics.ProcessStartInfo();
            start.FileName = pythonExe;
            start.Arguments = $"\"{scriptPath}\"";
            start.UseShellExecute = false;
            start.RedirectStandardOutput = true;
            start.RedirectStandardError = true;
            start.RedirectStandardInput = true;
            start.CreateNoWindow = true;
            start.StandardOutputEncoding = Encoding.UTF8;
            start.StandardErrorEncoding = Encoding.UTF8;

            process = new System.Diagnostics.Process();
            process.StartInfo = start;
            process.EnableRaisingEvents = true;

            process.Start();

            //string output = process.StandardOutput.ReadToEnd();
            //string error = process.StandardError.ReadToEnd();

            //Debug.Log("Python script finished.");
            //Debug.Log("Output: " + output);
            //if (!string.IsNullOrEmpty(error))
            //{
            //    Debug.LogError("Python Error: " + error);
            //}
        });
        thread.IsBackground = true;
        thread.Start();
    }

    private void OnApplicationQuit()
    {
        var settings = new JsonSerializerSettings
        {
            ReferenceLoopHandling = ReferenceLoopHandling.Ignore
        };
        string json = JsonConvert.SerializeObject(gazeLog, Formatting.Indented, settings);
        File.WriteAllText(jsonPath, json);
        Debug.Log($"Gaze data exported on quit in {jsonPath}");

        udpClient.Close();
        receiveThread.Abort();
        if (process != null)
        {
            process.Kill();
        }
    }

    private void WriteCSVHeader()
    {
        string header = "timestamp,gaze_origin_x,gaze_origin_y,gaze_origin_z,gaze_direction_x,gaze_direction_y,gaze_direction_z";
        File.WriteAllText(csvPath, header + "\n");
    }

    private void AppendCSV(GazeEntry entry)
    {
        string line = $"{entry.timestamp},{entry.gaze_origin["x"]},{entry.gaze_origin["y"]},{entry.gaze_origin["z"]}," +
                      $"{entry.gaze_direction["x"]},{entry.gaze_direction["y"]},{entry.gaze_direction["z"]},";
        File.AppendAllText(csvPath, line + "\n");
    }

    public static class JsonHelper
    {
        public static T[] FromJson<T>(string json)
        {
            Wrapper<T> wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
            return wrapper.Items;
        }

        public static string ToJson<T>(T[] array, bool prettyPrint = false)
        {
            Wrapper<T> wrapper = new Wrapper<T>();
            wrapper.Items = array;
            return JsonUtility.ToJson(wrapper, prettyPrint);
        }

        [System.Serializable]
        private class Wrapper<T>
        {
            public T[] Items;
        }
    }

    [System.Serializable]
    private class PegInfo
    {
        public Dictionary<string, float> location;
        public bool isHitByGaze;
    }

    [System.Serializable]
    private class GazeEntry
    {
        public string timestamp;
        public Dictionary<string, float> gaze_origin;
        public Dictionary<string, float> gaze_direction;

        public Dictionary<string, PegInfo> peg = new Dictionary<string, PegInfo>
        {
            { "1", new PegInfo() },
            { "2", new PegInfo() },
            { "3", new PegInfo() },
            { "4", new PegInfo() },
            { "5", new PegInfo() },
            { "6", new PegInfo() }
        };
    }
}

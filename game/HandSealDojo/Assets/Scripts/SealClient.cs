using System;
using System.Collections.Concurrent;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

[Serializable]
public class SealMessage
{
    public double t;
    public string stable_seal;
    public float hold;
    public bool new_confirm;
    public string raw;
    public float raw_score;
    public string expect;
    public float expect_score;
    public float expect_thr;
    public int hands;
    public float fps;
}

[Serializable]
public class FrameMessage
{
    public string type;
    public string jpg;
}

public class SealClient : MonoBehaviour
{
    public string url = "ws://127.0.0.1:8765";
    public bool showDebug = true;

    public static readonly string[] Seals =
        { "Rat", "Ox", "Tiger", "Hare", "Dragon", "Snake",
          "Horse", "Ram", "Monkey", "Bird", "Dog", "Boar" };

    public event Action<SealMessage> OnMessage;
    public event Action<string> OnSealConfirmed;

    public SealMessage Latest { get; private set; }
    public Texture2D Preview { get; private set; }
    public bool Connected { get; private set; }

    const string FramePrefix = "{\"type\":\"frame\"";

    readonly ConcurrentQueue<string> _incoming = new ConcurrentQueue<string>();
    ClientWebSocket _ws;
    CancellationTokenSource _cts;
    string _expect = "";
    string _lastConfirmed = "-";

    void OnEnable()
    {
        _cts = new CancellationTokenSource();
        _ = RunAsync(_cts.Token);
    }

    void OnDisable()
    {
        _cts?.Cancel();
        try { _ws?.Dispose(); } catch { }
        Connected = false;
        if (Preview != null) Destroy(Preview);
        Preview = null;
    }

    async Task RunAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                _ws = new ClientWebSocket();
                await _ws.ConnectAsync(new Uri(url), ct);
                Connected = true;
                await SendExpectAsync(_expect);   // re-send the target after a reconnect

                var buffer = new byte[16384];
                var sb = new StringBuilder();
                while (_ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
                {
                    sb.Clear();
                    WebSocketReceiveResult result;
                    do
                    {
                        result = await _ws.ReceiveAsync(new ArraySegment<byte>(buffer), ct);
                        if (result.MessageType == WebSocketMessageType.Close) break;
                        sb.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));
                    } while (!result.EndOfMessage);

                    if (sb.Length > 0) _incoming.Enqueue(sb.ToString());
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception e) { Debug.LogWarning("SealClient: " + e.Message); }

            Connected = false;
            try { await Task.Delay(1000, ct); } catch (OperationCanceledException) { }
        }
    }

    public void SetExpect(string seal)
    {
        _expect = seal ?? "";
        _ = SendExpectAsync(_expect);
    }

    async Task SendExpectAsync(string seal)
    {
        if (_ws == null || _ws.State != WebSocketState.Open) return;
        string json = string.IsNullOrEmpty(seal)
            ? "{\"expect\": null}"
            : "{\"expect\": \"" + seal + "\"}";
        var bytes = Encoding.UTF8.GetBytes(json);
        try
        {
            await _ws.SendAsync(new ArraySegment<byte>(bytes),
                                WebSocketMessageType.Text, true, CancellationToken.None);
        }
        catch (Exception e) { Debug.LogWarning("SealClient send failed: " + e.Message); }
    }

    void Update()
    {
        if (!Connected && _incoming.IsEmpty) Latest = null;

        string lastFrame = null;
        while (_incoming.TryDequeue(out var raw))
        {
            if (raw.StartsWith(FramePrefix)) { lastFrame = raw; continue; }

            SealMessage msg;
            try { msg = JsonUtility.FromJson<SealMessage>(raw); }
            catch { continue; }
            if (msg == null) continue;

            Latest = msg;
            OnMessage?.Invoke(msg);
            if (msg.new_confirm && !string.IsNullOrEmpty(msg.stable_seal))
            {
                _lastConfirmed = msg.stable_seal;
                OnSealConfirmed?.Invoke(msg.stable_seal);
            }
        }
        if (lastFrame != null) DecodeFrame(lastFrame);   // only decode the newest frame
    }

    void DecodeFrame(string raw)
    {
        try
        {
            var fm = JsonUtility.FromJson<FrameMessage>(raw);
            if (fm == null || string.IsNullOrEmpty(fm.jpg)) return;
            var bytes = Convert.FromBase64String(fm.jpg);
            if (Preview == null) Preview = new Texture2D(2, 2, TextureFormat.RGB24, false);
            ImageConversion.LoadImage(Preview, bytes);
        }
        catch (Exception e) { Debug.LogWarning("SealClient frame decode failed: " + e.Message); }
    }

    void OnGUI()
    {
        if (!showDebug) return;

        GUI.Label(new Rect(10, 10, 800, 25), Connected
            ? "Perception: connected"
            : "Perception: NOT connected (start perception/service.py)");

        var m = Latest;
        if (m != null)
        {
            string target = string.IsNullOrEmpty(m.expect) ? "(free)" : m.expect;
            GUI.Label(new Rect(10, 35, 1000, 25),
                $"raw: {m.raw} {m.raw_score:F2} | target: {target} {m.expect_score:F2}/{m.expect_thr:F2} | hold {m.hold:F2} | hands {m.hands} | fps {m.fps:F0}");
        }
        GUI.Label(new Rect(10, 60, 800, 25), "Last confirmed: " + _lastConfirmed);

        for (int i = 0; i < Seals.Length; i++)
        {
            var r = new Rect(10 + (i % 6) * 100, 95 + (i / 6) * 35, 95, 30);
            if (GUI.Button(r, Seals[i])) SetExpect(Seals[i]);
        }
        if (GUI.Button(new Rect(10, 170, 200, 30), "Free mode (any seal)")) SetExpect("");
    }
}
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class AcademyController : MonoBehaviour
{
    class Sequence
    {
        public string name;
        public string[] seals;
        public string payoff;      // set for jutsu chains only
    }

    struct StepResult { public string seal; public float seconds; public float reach; public float peak; }
    enum State { Menu, Running, StepDone, Finished }

    public SealClient client;
    public float stepDoneDelay = 0.9f;

    readonly List<Sequence> _seqs = new List<Sequence>();
    readonly List<StepResult> _results = new List<StepResult>();
    State _state = State.Menu;
    int _seqIdx, _step;
    string _target = "";
    float _stepStart, _doneAt, _peak, _reach = -1f;
    float _flashStart = -10f, _flashLen;
    Color _flashColor = Color.white;
    Texture2D _ref;
    GUIStyle _title, _big, _small, _mid, _left, _btn, _wrap;

    bool Connected { get { return client != null && client.Connected; } }

    void Awake()
    {
        // Lessons
        _seqs.Add(new Sequence { name = "Lesson 1: Rat, Ox, Tiger", seals = new[] { "Rat", "Ox", "Tiger" } });
        _seqs.Add(new Sequence { name = "Lesson 2: Hare, Dragon, Snake", seals = new[] { "Hare", "Dragon", "Snake" } });
        _seqs.Add(new Sequence { name = "Lesson 3: Horse, Ram, Monkey", seals = new[] { "Horse", "Ram", "Monkey" } });
        _seqs.Add(new Sequence { name = "Lesson 4: Bird, Dog, Boar", seals = new[] { "Bird", "Dog", "Boar" } });
        // Jutsu chains, from Kazuhito00's setting/jutsu.csv
        // Jutsu chains, from Kazuhito00's setting/jutsu.csv
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Fireball",
            seals = new[] { "Snake", "Ram", "Monkey", "Boar", "Horse", "Tiger" },
            payoff = "FIREBALL JUTSU!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Clone",
            seals = new[] { "Ram", "Snake", "Tiger" },
            payoff = "CLONE JUTSU!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Summoning",
            seals = new[] { "Dog", "Boar", "Bird", "Monkey", "Ram" },
            payoff = "SUMMONING JUTSU!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Phoenix Flower",
            seals = new[] { "Rat", "Tiger", "Dog", "Ox", "Hare", "Tiger" },
            payoff = "PHOENIX FLOWER JUTSU!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Water Trumpet",
            seals = new[] { "Dragon", "Tiger", "Hare" },
            payoff = "WATER TRUMPET!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Dragon Flame",
            seals = new[] { "Snake", "Dragon", "Hare", "Tiger" },
            payoff = "DRAGON FLAME JUTSU!"
        });
        _seqs.Add(new Sequence
        {
            name = "Jutsu: Substitution",
            seals = new[] { "Ram", "Boar", "Ox", "Dog", "Snake" },
            payoff = "SUBSTITUTION JUTSU!"
        });
    }

    void Start()
    {
        if (client == null) client = GetComponent<SealClient>();
        if (client == null)
        {
            Debug.LogError("AcademyController needs a SealClient on the same GameObject.");
            enabled = false;
            return;
        }
        client.showDebug = false;
        client.OnMessage += HandleMessage;
    }

    void OnDestroy()
    {
        if (client != null) client.OnMessage -= HandleMessage;
        if (_ref != null) Destroy(_ref);
        if (_tex != null) Destroy(_tex);
        if (_texHover != null) Destroy(_texHover);
        if (_texActive != null) Destroy(_texActive);
    }

    void HandleMessage(SealMessage m)
    {
        if (_state != State.Running) return;
        if (m.expect != _target) return;            // ignore messages from before the target switched
        _peak = Mathf.Max(_peak, m.expect_score);
        if (_reach < 0f && m.expect_score >= m.expect_thr) _reach = Time.unscaledTime - _stepStart;
        if (m.new_confirm && m.stable_seal == _target) CompleteStep();
    }

    void LoadReference(string seal)
    {
        if (_ref != null) { Destroy(_ref); _ref = null; }
        string path = Path.GetFullPath(Path.Combine(
            Application.dataPath, "..", "..", "..", "shared", "reference", seal + ".jpg"));
        if (!File.Exists(path)) return;
        try
        {
            var tex = new Texture2D(2, 2, TextureFormat.RGB24, false);
            if (ImageConversion.LoadImage(tex, File.ReadAllBytes(path))) _ref = tex;
            else Destroy(tex);
        }
        catch (System.Exception e) { Debug.LogWarning("Reference load failed: " + e.Message); }
    }

    void StartSequence(int i)
    {
        _seqIdx = i;
        _step = 0;
        _results.Clear();
        _state = State.Running;
        BeginStep();
    }

    void BeginStep()
    {
        _target = _seqs[_seqIdx].seals[_step];
        _stepStart = Time.unscaledTime;
        _peak = 0f;
        _reach = -1f;
        LoadReference(_target);
        client.SetExpect(_target);
    }

    void CompleteStep()
    {
        _results.Add(new StepResult
        {
            seal = _target,
            seconds = Time.unscaledTime - _stepStart,
            reach = _reach,
            peak = _peak
        });
        _state = State.StepDone;
        _doneAt = Time.unscaledTime;
        Flash(new Color(0.2f, 0.9f, 0.4f), 0.4f);
    }

    void Finish()
    {
        _state = State.Finished;
        client.SetExpect("");
        var seq = _seqs[_seqIdx];
        if (seq.payoff != null) Flash(new Color(1f, 0.55f, 0.1f), 1.6f);
        else Flash(new Color(0.2f, 0.9f, 0.4f), 0.8f);
    }

    void GoMenu()
    {
        _state = State.Menu;
        _target = "";
        client.SetExpect("");
    }

    void Flash(Color c, float len)
    {
        _flashColor = c;
        _flashLen = len;
        _flashStart = Time.unscaledTime;
    }

    void Update()
    {
        if (_state == State.StepDone && Time.unscaledTime - _doneAt >= stepDoneDelay)
        {
            _step++;
            if (_step >= _seqs[_seqIdx].seals.Length) Finish();
            else { _state = State.Running; BeginStep(); }
        }
    }

    // ---------------- UI (IMGUI placeholder) ----------------

    Texture2D _tex, _texHover, _texActive;

    static Texture2D Solid(Color c)
    {
        var t = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        t.SetPixel(0, 0, c);
        t.Apply();
        return t;
    }

    static GUIStyle MakeLabel(int size, FontStyle fs, TextAnchor anchor, bool wrap = false)
    {
        var st = new GUIStyle
        {
            fontSize = size,
            fontStyle = fs,
            alignment = anchor,
            wordWrap = wrap
        };
        st.normal.textColor = Color.white;
        return st;
    }

    void InitStyles()
    {
        if (_title != null) return;
        _title = MakeLabel(40, FontStyle.Bold, TextAnchor.MiddleCenter);
        _big = MakeLabel(90, FontStyle.Bold, TextAnchor.MiddleCenter);
        _mid = MakeLabel(26, FontStyle.Normal, TextAnchor.MiddleCenter);
        _small = MakeLabel(18, FontStyle.Normal, TextAnchor.MiddleCenter);
        _wrap = MakeLabel(18, FontStyle.Normal, TextAnchor.MiddleCenter, true);
        _left = MakeLabel(18, FontStyle.Normal, TextAnchor.MiddleLeft);

        _btn = MakeLabel(20, FontStyle.Bold, TextAnchor.MiddleCenter);
        _tex = Solid(new Color(0.22f, 0.25f, 0.32f, 0.95f));
        _texHover = Solid(new Color(0.30f, 0.35f, 0.45f, 0.95f));
        _texActive = Solid(new Color(0.40f, 0.30f, 0.15f, 0.95f));
        _btn.normal.background = _tex;
        _btn.hover.background = _texHover;
        _btn.active.background = _texActive;
        _btn.hover.textColor = Color.white;
        _btn.active.textColor = Color.white;
    }

    void OnGUI()
    {
        InitStyles();
        float s = Screen.height / 720f;
        GUI.matrix = Matrix4x4.Scale(new Vector3(s, s, 1f));
        float W = Screen.width / s, H = 720f;

        var e = Event.current;
        if (e.type == EventType.KeyDown && e.keyCode == KeyCode.Escape && _state != State.Menu)
        {
            GoMenu();
            e.Use();
        }

        DrawFlash(W, H);
        switch (_state)
        {
            case State.Menu: DrawMenu(W); break;
            case State.Running:
            case State.StepDone: DrawRun(W); break;
            case State.Finished: DrawFinished(W); break;
        }
        GUI.matrix = Matrix4x4.identity;
    }

    void DrawFlash(float W, float H)
    {
        float t = Time.unscaledTime - _flashStart;
        if (t < 0f || t >= _flashLen) return;
        var old = GUI.color;
        GUI.color = new Color(_flashColor.r, _flashColor.g, _flashColor.b, 0.55f * (1f - t / _flashLen));
        GUI.DrawTexture(new Rect(0, 0, W, H), Texture2D.whiteTexture);
        GUI.color = old;
    }

    void DrawMenu(float W)
    {
        GUI.Label(new Rect(0, 30, W, 60), "HAND SEAL DOJO", _title);
        GUI.Label(new Rect(0, 90, W, 30),
            Connected ? "Perception service: connected" : "Waiting for perception service (run perception/service.py)...",
            _small);

        float colW = 380f, x0 = W / 2f - colW - 20f, x1 = W / 2f + 20f, y0 = 190f;
        GUI.Label(new Rect(x0, y0 - 36f, colW, 30), "ACADEMY: learn the seals", _mid);
        GUI.Label(new Rect(x1, y0 - 36f, colW, 30), "JUTSU: chain the seals", _mid);

        int lessonRow = 0, jutsuRow = 0;
        for (int i = 0; i < _seqs.Count; i++)
        {
            bool jutsu = _seqs[i].payoff != null;
            float x = jutsu ? x1 : x0;
            float y = y0 + (jutsu ? jutsuRow++ : lessonRow++) * 62f;
            if (GUI.Button(new Rect(x, y, colW, 52f), _seqs[i].name, _btn)) StartSequence(i);
        }
    }

    void DrawPanel(Rect r, Texture tex, string empty)
    {
        var old = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.55f);
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = old;
        if (tex != null) GUI.DrawTexture(r, tex, ScaleMode.ScaleToFit);
        else GUI.Label(new Rect(r.x + 10f, r.y, r.width - 20f, r.height), empty, _wrap);
    }

    void DrawRun(float W)
    {
        var seq = _seqs[_seqIdx];
        GUI.Label(new Rect(0, 12, W, 28), seq.name + "     (Esc = menu)", _small);

        string line = "";
        for (int i = 0; i < seq.seals.Length; i++)
        {
            string tag = i < _step || (i == _step && _state == State.StepDone) ? "[done] " : (i == _step ? ">> " : "");
            line += tag + seq.seals[i] + (i < seq.seals.Length - 1 ? "   -   " : "");
        }
        GUI.Label(new Rect(0, 42, W, 28), line, _small);

        string head = _state == State.StepDone ? "OK!" : _target.ToUpper();
        GUI.Label(new Rect(0, 66, W, 110), head, _big);

        var mm = client != null ? client.Latest : null;
        bool needRelax = _state == State.Running && mm != null && mm.expect == _target && !mm.armed;
        string tip = needRelax
            ? "Relax first: lower your hands or break the pose, then make the seal."
            : (_target == "Hare" || _target == "Ram" || _target == "Snake")
                ? "Tip: match the reference pose AND your camera framing. Small changes move the score a lot."
                : "Keep both hands in view at chest height, facing the camera. Hold still until the bar fills.";

        // reference + live view
        float refW = 300f, liveW = 400f, gap = 40f, panelH = 225f, py = 236f;
        float px0 = W / 2f - (refW + gap + liveW) / 2f;
        GUI.Label(new Rect(px0, py - 26f, refW, 24f), "Reference (your best capture)", _small);
        GUI.Label(new Rect(px0 + refW + gap, py - 26f, liveW, 24f), "You (live, mirrored)", _small);
        DrawPanel(new Rect(px0, py, refW, panelH), _ref,
                  "No reference yet. Make this seal once and the game saves one.");
        DrawPanel(new Rect(px0 + refW + gap, py, liveW, panelH),
                  Connected && client != null ? client.Preview : null, "No camera feed");

        var m = client != null ? client.Latest : null;
        bool valid = m != null && m.expect == _target;
        float score = valid ? m.expect_score : 0f;
        float thr = valid ? m.expect_thr : 0.6f;
        float hold = valid ? m.hold : 0f;

        float bw = 600f, bx = W / 2f - bw / 2f;
        DrawBar(new Rect(bx, 520, bw, 28), score, thr,
                score >= thr ? new Color(0.2f, 0.9f, 0.4f) : new Color(1f, 0.65f, 0.2f),
                "Match: " + score.ToString("F2") + "  (needs " + thr.ToString("F2") + ")");
        DrawBar(new Rect(bx, 590, bw, 28), hold, -1f, new Color(0.3f, 0.7f, 1f), "Hold steady");

        if (!Connected)
            GUI.Label(new Rect(0, 630, W, 26), "Perception service not connected - start perception/service.py", _small);
        else if (m != null)
            GUI.Label(new Rect(0, 668, W, 26),
                "raw: " + m.raw + " " + m.raw_score.ToString("F2") + "  |  hands " + m.hands + "  |  fps " + m.fps.ToString("F0"),
                _small);
    }

    void DrawBar(Rect r, float value, float marker, Color fill, string label)
    {
        GUI.Label(new Rect(r.x, r.y - 26f, r.width, 24f), label, _left);
        var old = GUI.color;
        GUI.color = new Color(0f, 0f, 0f, 0.55f);
        GUI.DrawTexture(r, Texture2D.whiteTexture);
        GUI.color = fill;
        GUI.DrawTexture(new Rect(r.x, r.y, r.width * Mathf.Clamp01(value), r.height), Texture2D.whiteTexture);
        if (marker >= 0f)
        {
            GUI.color = Color.white;
            GUI.DrawTexture(new Rect(r.x + r.width * Mathf.Clamp01(marker) - 1.5f, r.y - 5f, 3f, r.height + 10f),
                            Texture2D.whiteTexture);
        }
        GUI.color = old;
    }

    void DrawFinished(float W)
    {
        var seq = _seqs[_seqIdx];
        GUI.Label(new Rect(0, 40, W, 60), seq.payoff ?? "LESSON COMPLETE!", _title);
        GUI.Label(new Rect(0, 100, W, 30), seq.name, _small);

        float y = 160f;
        float total = 0f;
        foreach (var r in _results)
        {
            string reach = r.reach >= 0f ? r.reach.ToString("F1") + " s" : "-";
            GUI.Label(new Rect(W / 2f - 340f, y, 680f, 28f),
                r.seal + ":  confirmed " + r.seconds.ToString("F1") + " s   (matched at " + reach +
                ")   best " + r.peak.ToString("F2"), _left);
            total += r.seconds;
            y += 30f;
        }
        GUI.Label(new Rect(W / 2f - 340f, y + 6f, 680f, 28f), "Total: " + total.ToString("F1") + " s", _left);

        if (GUI.Button(new Rect(W / 2f - 210f, y + 60f, 200f, 50f), "Retry", _btn)) StartSequence(_seqIdx);
        if (GUI.Button(new Rect(W / 2f + 10f, y + 60f, 200f, 50f), "Menu", _btn)) GoMenu();
    }
}
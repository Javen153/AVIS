using System.Diagnostics;
using Avis.Audio;
using Avis.Core;

namespace Avis.App;

/// <summary>
/// Owns capture, the FE-v1 frontend, the object source and the analysis thread.
/// The UI thread only reads <see cref="Latest"/> (lock-free publication).
/// </summary>
public sealed class AvisEngine : IDisposable
{
    private readonly AppSettings _settings;
    private readonly FrontendWeights _weights;
    private readonly object _lock = new();
    private readonly List<double> _analysisMs = new();
    private IObjectSource? _source;
    private FrontendStream _frontend;
    private AudioCapture? _audio;
    private Thread? _thread;
    private volatile bool _running;
    private AnalysisFrame? _latest;
    private FrameJson.Writer? _writer;
    private int _frameCount;
    private float[]? _wav;
    private int _wavPos;

    public string Kind { get; private set; } = "dsp";
    public string Status { get; private set; } = "";
    public double ObjectsAvg { get; private set; }
    public long Frames => _frameCount;
    public AnalysisFrame? Latest => Volatile.Read(ref _latest);

    public AvisEngine(AppSettings settings)
    {
        _settings = settings;
        _weights = new FrontendWeights(AvisPaths.Weights);
        _frontend = new FrontendStream(_weights);
    }

    public string WavPath { get; set; } = "";
    public volatile bool Paused;

    public static bool ModelAvailable(string dir) => File.Exists(Path.Combine(dir, "model.onnx"));

    /// <summary>Creates the requested source. Falls back to DSP when the model is missing.</summary>
    public bool SetSource(string kind)
    {
        kind = (kind ?? "").ToLowerInvariant();
        string status;
        IObjectSource src;
        try
        {
            if (kind == "neural")
            {
                string dir = string.IsNullOrEmpty(_settings.ModelDirOverride)
                    ? AvisPaths.ModelDir(_settings.ModelName) : _settings.ModelDirOverride;
                if (!ModelAvailable(dir)) throw new FileNotFoundException("нет model.onnx в " + dir);
                var neural = new NeuralObjectSource(_weights, dir, _settings);
                Warmup(neural);
                src = neural;
                status = $"Нейро {neural.ModelVersion} · {neural.Provider}";
            }
            else if (kind == "mock")
            {
                src = new MockObjectSource(6);
                status = "Mock · синтетика";
            }
            else
            {
                kind = "dsp";
                src = new DspObjectSource(_weights);
                status = "DSP · спектральные партиалы";
            }
        }
        catch (Exception ex)
        {
            src = new DspObjectSource(_weights);
            kind = "dsp";
            status = "DSP · " + ex.Message;
        }
        lock (_lock)
        {
            _source = src;
            _frontend.Reset();
            Kind = kind;
            Status = status;
        }
        return true;
    }

    /// <summary>First ORT call allocates arenas and compiles kernels (~tens of ms).
    /// Feed silence a few times so the realtime loop never pays that cost.</summary>
    private static void Warmup(NeuralObjectSource n)
    {
        var f = new FrontendFrame
        {
            TimeSec = 0,
            BandsDbM = new float[FrontendWeights.Bands],
            PM = new float[FrontendWeights.Bands],
            PL = new float[FrontendWeights.Bands],
            PR = new float[FrontendWeights.Bands],
            Features = new float[3 * FrontendWeights.Bands],
        };
        for (int i = 0; i < 3; i++) n.Process(f, 0);
        n.Reset();
    }

    public void ApplySettings(AppSettings s)
    {
        if (_audio != null) _audio.GainDb = s.GainDb;
        if (_source is NeuralObjectSource n) n.UpdateSettings(s);
    }

    public void Start()
    {
        if (_thread != null) return;
        if (WavPath.Length > 0)
        {
            var wav = WavIo.Read(WavPath);
            var stereo = WavIo.ToStereo(wav.Interleaved, wav.Channels);
            _wav = wav.SampleRate == FrontendWeights.SampleRate
                ? stereo : HermiteResampler.Resample(stereo, wav.SampleRate, FrontendWeights.SampleRate);
            Status = "WAV ? " + Path.GetFileName(WavPath);
        }
        else if (Kind != "mock")
        {
            _audio = new AudioCapture { GainDb = _settings.GainDb };
            if (!_audio.Start())
            {
                _audio.Dispose();
                _audio = null;
                Status = "Нет устройства вывода — mock";
                SetSource("mock");
            }
            else
            {
                Status = "WASAPI · " + _audio.DeviceName;
            }
        }
        _running = true;
        _thread = new Thread(Loop)
        {
            IsBackground = true,
            Name = "avis-analysis",
            Priority = ThreadPriority.AboveNormal,
        };
        _thread.Start();
    }

    private void Loop()
    {
        var hop = new float[FrontendWeights.Hop * 2];
        double t = 0;
        var empty = new FrontendFrame
        {
            TimeSec = 0,
            BandsDbM = new float[192],
            PM = new float[192],
            PL = new float[192],
            PR = new float[192],
            Features = new float[576],
        };
        while (_running)
        {
            if (Paused) { Thread.Sleep(20); continue; }
            IObjectSource? src;
            lock (_lock) src = _source;
            if (src == null) { Thread.Sleep(5); continue; }
            if (Kind == "mock")
            {
                Thread.Sleep(10);
                t += 0.01;
                Process(src, empty, t);
                continue;
            }
            if (_wav != null)
            {
                if (_wavPos + hop.Length > _wav.Length) _wavPos = 0;
                Array.Copy(_wav, _wavPos, hop, 0, hop.Length);
                _wavPos += hop.Length;
                foreach (var f in _frontend.Push(hop)) Process(src, f, f.TimeSec);
                Thread.Sleep(10);
                continue;
            }
            if (_audio != null && _audio.TryReadHop(hop))
            {
                foreach (var f in _frontend.Push(hop)) Process(src, f, f.TimeSec);
            }
            else if (_audio != null) _audio.WaitForData(8);
            else Thread.Sleep(5);
        }
    }

    private void Process(IObjectSource src, FrontendFrame f, double t)
    {
        if (Kind == "mock") f = new FrontendFrame
        {
            TimeSec = t,
            BandsDbM = f.BandsDbM,
            PM = f.PM,
            PL = f.PL,
            PR = f.PR,
            Features = f.Features,
        };
        var sw = Stopwatch.StartNew();
        AnalysisFrame frame;
        try { frame = src.Process(f, t); }
        catch (Exception ex) { Status = "ошибка анализа: " + ex.Message; Thread.Sleep(50); return; }
        sw.Stop();
        lock (_lock)
        {
            if (_analysisMs.Count >= 600) _analysisMs.RemoveRange(0, 300);
            _analysisMs.Add(sw.Elapsed.TotalMilliseconds);
            _writer?.Write(frame);
        }
        ObjectsAvg = ObjectsAvg <= 0 ? frame.Objects.Count : ObjectsAvg * 0.98 + frame.Objects.Count * 0.02;
        Interlocked.Increment(ref _frameCount);
        Volatile.Write(ref _latest, frame);
    }

    public double[] AnalysisTimes()
    {
        lock (_lock) return _analysisMs.ToArray();
    }

    public void ClearTimes()
    {
        lock (_lock) _analysisMs.Clear();
    }

    public void StartRecording(string path)
    {
        lock (_lock)
        {
            _writer?.Dispose();
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            _writer = new FrameJson.Writer(path);
        }
    }

    public void StopRecording()
    {
        lock (_lock) { _writer?.Dispose(); _writer = null; }
    }

    public void Dispose()
    {
        _running = false;
        _thread?.Join(500);
        StopRecording();
        _audio?.Dispose();
    }
}

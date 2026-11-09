using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Avis.Core;

/// <summary>M-v1 ONNX inference + ObjectTracker. Strictly causal, state kept in the GRU.</summary>
public sealed class NeuralObjectSource : IObjectSource
{
    public string Name => "neural";
    private readonly FrontendWeights _w;
    private readonly InferenceSession _session;
    private readonly ObjectTracker _tracker;
    private readonly float[] _h = new float[2 * 384];
    private readonly float[] _feat = new float[3 * 192];
    private readonly DenseTensor<float> _featTensor;
    private readonly DenseTensor<float> _hTensor;
    private readonly List<NamedOnnxValue> _inputs;
    private readonly float[][] _emb = new float[192][];
    private readonly float[] _act = new float[192];
    private readonly float[] _pitch = new float[192];
    private readonly float[] _sal = new float[192];
    private readonly AppSettings _settings;
    private readonly float[] _pcaMean = new float[20];
    private readonly float[][] _pcaComp = new float[2][];
    public string ModelVersion { get; } = "?";
    public string Provider { get; private set; } = "cpu";
    public double LastInferenceMs { get; private set; }

    public NeuralObjectSource(FrontendWeights w, string modelDir, AppSettings settings)
    {
        _w = w;
        _settings = settings;
        string onnx = Path.Combine(modelDir, "model.onnx");
        if (!File.Exists(onnx)) throw new FileNotFoundException("model.onnx not found", onnx);
        var opts = new SessionOptions
        {
            IntraOpNumThreads = OrtThreads(),
            InterOpNumThreads = 1,
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        };
        if (settings.Provider == InferenceProvider.DirectML)
        {
            try { opts.AppendExecutionProvider_DML(0); Provider = "directml"; }
            catch { Provider = "cpu"; }
        }
        _session = new InferenceSession(onnx, opts);
        _featTensor = new DenseTensor<float>(_feat, new[] { 1, 3, 192 });
        _hTensor = new DenseTensor<float>(_h, new[] { 2, 1, 384 });
        _inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("feat", _featTensor),
            NamedOnnxValue.CreateFromTensor("h_in", _hTensor),
        };
        _tracker = new ObjectTracker(w, TrackerConfig.FromSettings(settings));
        var mj = Path.Combine(modelDir, "model.json");
        if (File.Exists(mj))
        {
            try
            {
                using var doc = JsonDocument.Parse(File.ReadAllText(mj));
                var root = doc.RootElement;
                if (root.TryGetProperty("version", out var v)) ModelVersion = v.GetString() ?? "?";
                if (root.TryGetProperty("color_pca", out var pca))
                {
                    var mean = pca.GetProperty("mean");
                    for (int i = 0; i < 20; i++) _pcaMean[i] = mean[i].GetSingle();
                    var comps = pca.GetProperty("components");
                    _pcaComp[0] = comps[0].EnumerateArray().Select(e => e.GetSingle()).ToArray();
                    _pcaComp[1] = comps[1].EnumerateArray().Select(e => e.GetSingle()).ToArray();
                }
            }
            catch { /* model.json is optional metadata */ }
        }
        if (_pcaComp[0] == null) { _pcaComp[0] = new float[20]; _pcaComp[1] = new float[20]; }
    }

    public void UpdateSettings(AppSettings settings)
    {
        _tracker.UpdateConfig(TrackerConfig.FromSettings(settings));
    }

    private static int OrtThreads()
    {
        var env = Environment.GetEnvironmentVariable("AVIS_ORT_THREADS");
        return int.TryParse(env, out var n) && n >= 0 ? n : 0;
    }

    public void Reset()
    {
        Array.Clear(_h);
        _tracker.Reset();
    }

    /// <summary>Run one GRU step and return raw band outputs (used by parity checks).</summary>
    public void RunRaw(ReadOnlySpan<float> features, float[][] emb, float[] act, float[] pitch, float[] sal)
    {
        features[.._feat.Length].CopyTo(_feat);
        var t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        using var res = _session.Run(_inputs);
        LastInferenceMs = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
        var embT = res.First(r => r.Name == "emb").AsTensor<float>();
        var actT = res.First(r => r.Name == "act").AsTensor<float>();
        var pitT = res.First(r => r.Name == "pitch").AsTensor<float>();
        var salT = res.First(r => r.Name == "sal").AsTensor<float>();
        var hT = res.First(r => r.Name == "h_out").AsTensor<float>();

        int idx = 0;
        foreach (var v in hT) _h[idx++] = v;
        for (int k = 0; k < 192; k++)
        {
            var e = emb[k] ?? (emb[k] = new float[20]);
            for (int d = 0; d < 20; d++) e[d] = embT[0, k, d];
            act[k] = actT[0, k];
            pitch[k] = pitT[0, k];
            sal[k] = salT[0, k];
        }
    }

    public AnalysisFrame Process(in FrontendFrame f, double timeSec)
    {
        RunRaw(f.Features, _emb, _act, _pitch, _sal);
        var frame = _tracker.Process(f, _emb, _act, _pitch, _sal);
        frame.Source = Name;
        if (_settings.Color == ColorMode.Timbre)
        {
            foreach (var o in frame.Objects)
            {
                if (o.Embedding == null) continue;
                float x = 0, y = 0;
                for (int d = 0; d < 20; d++)
                {
                    float v = o.Embedding[d] - _pcaMean[d];
                    x += v * _pcaComp[0][d];
                    y += v * _pcaComp[1][d];
                }
                double hue = (Math.Atan2(y, x) / (2 * Math.PI) + 0.5) % 1.0;
                o.Color = ObjectParams.OklchToRgb(0.74, 0.15, hue);
            }
        }
        return frame;
    }
}

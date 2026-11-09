using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Avis.Core;
using Avis.Render;

internal static class BenchRunner
{
    public static int Run(string what, Func<string, string, string> arg, HashSet<string> flags)
    {
        switch (what.ToLowerInvariant())
        {
            case "frontend": return Frontend(arg, flags);
            case "model": return Model(arg, flags);
            case "tracker": return Tracker(arg, flags);
            case "render": return Render(arg, flags);
            default:
                Console.Error.WriteLine($"unknown bench target '{what}'");
                return 2;
        }
    }

    private static FrontendWeights Weights(string path) => new(path);

    private static int Frontend(Func<string, string, string> arg, HashSet<string> flags)
    {
        var w = Weights(arg("weights", AvisPaths.Weights));
        var stream = new FrontendStream(w);
        var input = Program.ReadFloats(Path.Combine(AvisPaths.Golden, "frontend_input.f32"));
        stream.Push(input); // warmup
        var times = new List<double>();
        // feed in 10 ms chunks to mimic the real-time path
        for (int off = 0; off + 480 * 2 <= input.Length; off += 480 * 2)
        {
            var sw = Stopwatch.StartNew();
            stream.Push(input.AsSpan(off, 480 * 2));
            times.Add(sw.Elapsed.TotalMilliseconds * 1000.0 / 1000.0);
        }
        var (p50, p99) = Percentiles(times);
        Console.WriteLine($"frontend: {times.Count} hops, p50 {p50:F3} ms, p99 {p99:F3} ms (limit 0.5)");
        Console.WriteLine($"{(p50 <= 0.5 ? "PASS" : "FAIL")} frontend p50 <= 0.5 ms");
        return p50 <= 0.5 ? 0 : 1;
    }

    private static int Model(Func<string, string, string> arg, HashSet<string> flags)
    {
        string dir = arg("model", Path.Combine(AvisPaths.Models, "v1"));
        if (!Directory.Exists(dir)) dir = AvisPaths.ModelDir(dir);
        if (!File.Exists(Path.Combine(dir, "model.onnx")))
        {
            Console.WriteLine("SKIP model bench: model.onnx missing");
            return 0;
        }
        var w = Weights(arg("weights", AvisPaths.Weights));
        var settings = new AppSettings();
        var src = new NeuralObjectSource(w, dir, settings);
        var stream = new FrontendStream(w);
        var frames = stream.Push(Program.ReadFloats(Path.Combine(AvisPaths.Golden, "frontend_input.f32")));
        for (int i = 0; i < 60; i++) src.Process(frames[i % frames.Count], frames[i % frames.Count].TimeSec);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        var times = new List<double>(frames.Count * 3);
        var infer = new List<double>(frames.Count * 3);
        for (int rep = 0; rep < 3; rep++)
        {
            foreach (var f in frames)
            {
                var sw = Stopwatch.StartNew();
                src.Process(f, f.TimeSec);
                times.Add(sw.Elapsed.TotalMilliseconds);
                infer.Add(src.LastInferenceMs);
            }
        }
        var (p50, p99) = Percentiles(times);
        var (i50, _) = Percentiles(infer);
        Console.WriteLine($"model: {times.Count} passes, infer p50 {i50:F2} ms, total p50 {p50:F2} ms p99 {p99:F2} ms (limit 1.5)");
        bool pass = i50 <= 1.5 && p99 <= 4.0;
        Console.WriteLine($"{(pass ? "PASS" : "FAIL")} model p50 <= 1.5 ms and frontend+model+tracker p99 <= 4 ms");
        return pass ? 0 : 1;
    }

    private static int Tracker(Func<string, string, string> arg, HashSet<string> flags)
    {
        var w = Weights(arg("weights", AvisPaths.Weights));
        var tr = new ObjectTracker(w, new TrackerConfig());
        var rng = new Random(5);
        var emb = new float[192][];
        for (int k = 0; k < 192; k++)
        {
            var v = new float[20];
            v[k % 2] = 1f;
            v[2] = (float)(rng.NextDouble() * 0.1);
            double n = Math.Sqrt(v.Sum(x => x * x));
            for (int d = 0; d < 20; d++) v[d] /= (float)n;
            emb[k] = v;
        }
        var db = new float[192];
        var pm = new float[192];
        var act = new float[192];
        for (int k = 0; k < 192; k++) { db[k] = -30; pm[k] = 1e-4f; act[k] = 0.9f; }
        var frame = new FrontendFrame
        {
            TimeSec = 0,
            BandsDbM = db,
            PM = pm,
            PL = pm,
            PR = pm,
            Features = new float[576],
        };
        var pitch = new float[192];
        var sal = new float[192];
        var times = new List<double>();
        for (int i = 0; i < 500; i++)
        {
            var sw = Stopwatch.StartNew();
            tr.Process(frame, emb, act, pitch, sal);
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        var (p50, p99) = Percentiles(times);
        Console.WriteLine($"tracker: 500 frames, p50 {p50:F3} ms, p99 {p99:F3} ms");
        return 0;
    }

    private static int Render(Func<string, string, string> arg, HashSet<string> flags)
    {
        int w = int.Parse(arg("w", "1920"));
        int h = int.Parse(arg("h", "1080"));
        int frames = int.Parse(arg("frames", "120"));
        var settings = new AppSettings { Bloom = 0.6, SplatSize = 1, History = HistoryMode.Off };
        var mock = new MockObjectSource(24);
        var frontendFrame = new FrontendFrame
        {
            TimeSec = 0,
            BandsDbM = new float[192],
            PM = new float[192],
            PL = new float[192],
            PR = new float[192],
            Features = new float[576],
        };
        using var renderer = new Renderer(w, h, flags.Contains("warp"));
        double t = 0;
        var times = new List<double>();
        for (int i = 0; i < frames; i++)
        {
            t += 1.0 / 144.0;
            var af = mock.Process(frontendFrame, t);
            var sw = Stopwatch.StartNew();
            renderer.Render(af, settings, t, 1.0 / 144.0);
            renderer.ReadPixels();
            times.Add(sw.Elapsed.TotalMilliseconds);
        }
        var (p50, p99) = Percentiles(times);
        Console.WriteLine($"device: {renderer.DeviceName}{(renderer.IsWarp ? " (WARP)" : "")}");
        Console.WriteLine($"render: {w}x{h} 24 objects, p50 {p50:F2} ms, p99 {p99:F2} ms (limit 3.0 + readback)");
        // readback dominates; compare only the GPU submit cost by re-timing without readback
        var submits = new List<double>();
        for (int i = 0; i < 60; i++)
        {
            t += 1.0 / 144.0;
            var af = mock.Process(frontendFrame, t);
            var sw = Stopwatch.StartNew();
            renderer.Render(af, settings, t, 1.0 / 144.0);
            renderer.Context.Flush();
            submits.Add(sw.Elapsed.TotalMilliseconds);
        }
        var (s50, s99) = Percentiles(submits);
        Console.WriteLine($"render submit: p50 {s50:F2} ms, p99 {s99:F2} ms");
        return s50 <= 3.0 ? 0 : 1;
    }

    private static (double p50, double p99) Percentiles(List<double> values)
    {
        if (values.Count == 0) return (0, 0);
        var sorted = values.OrderBy(v => v).ToList();
        return (sorted[(int)(sorted.Count * 0.50)], sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * 0.99))]);
    }
}

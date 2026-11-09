using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avis.Core;
using Avis.Render;

internal static class Program
{
    private static readonly Dictionary<string, string> Args = new(StringComparer.OrdinalIgnoreCase);
    private static readonly HashSet<string> Flags = new(StringComparer.OrdinalIgnoreCase);

    public static int Main(string[] raw)
    {
        if (raw.Length == 0 || raw[0] is "-h" or "--help" or "help")
        {
            PrintHelp();
            return raw.Length == 0 ? 1 : 0;
        }
        string command = raw[0].ToLowerInvariant();
        for (int i = 1; i < raw.Length; i++)
        {
            if (raw[i].StartsWith("--"))
            {
                string key = raw[i][2..];
                if (i + 1 < raw.Length && !raw[i + 1].StartsWith("--")) Args[key] = raw[++i];
                else Flags.Add(key);
            }
        }
        try
        {
            return command switch
            {
                "analyze" => Analyze(),
                "render" => Render(),
                "check" => Check(),
                "bench" => Bench(),
                "parity" => Parity(),
                "capture-test" => CaptureTest(),
                _ => Unknown(command),
            };
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"ERROR: {ex.Message}");
            if (Flags.Contains("verbose")) Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static int Unknown(string command)
    {
        Console.Error.WriteLine($"unknown command '{command}'");
        PrintHelp();
        return 2;
    }

    private static void PrintHelp()
    {
        Console.WriteLine("""
            avis-cli — AVIS offline analysis and rendering
              analyze --in a.wav [--source dsp|neural|mock] [--model models/v1] [--debug] --out frames.jsonl
              render  --frames frames.jsonl --out dir [--view main|spectro|masks|table] [--every 10] [--w 960] [--h 360]
              check   --frames frames.jsonl --expect S01.expect.json --out result.json
              bench   --what frontend|model|tracker|render [--seconds 5]
              parity  --golden shared/golden [--model models/v1]
              capture-test --seconds 5 --out cap.wav
            Common: --weights <path> --set group.name=value --verbose --warp
            """);
    }

    private static string Arg(string key, string fallback = "")
        => Args.TryGetValue(key, out var v) ? v : fallback;

    private static string Require(string key)
        => Args.TryGetValue(key, out var v) && v.Length > 0 ? v : throw new ArgumentException($"missing --{key}");

    /// <summary>Accepts an explicit model directory path or a bare model name under models/.</summary>
    private static string ResolveModelDir(string arg)
        => Directory.Exists(arg) ? Path.GetFullPath(arg) : AvisPaths.ModelDir(arg);

    private static AppSettings LoadSettings()
    {
        var path = Path.Combine(AvisPaths.LocalAppData(), "settings.json");
        var s = File.Exists(path) ? AppSettings.Load(path) : new AppSettings();
        if (Args.TryGetValue("preset", out var preset))
        {
            var built = AppSettings.BuiltInPresets();
            if (built.TryGetValue(preset, out var p)) s = p;
        }
        foreach (var kv in Args.Where(k => k.Key.Equals("set", StringComparison.OrdinalIgnoreCase)))
            s.SetByPath(kv.Value.Split('=')[0], kv.Value.Split('=')[1]);
        return s;
    }

    private static FrontendWeights Weights()
    {
        var path = Arg("weights", AvisPaths.Weights);
        return new FrontendWeights(path);
    }

    private static IObjectSource MakeSource(string kind, FrontendWeights w, AppSettings settings)
    {
        return kind.ToLowerInvariant() switch
        {
            "mock" => new MockObjectSource(6),
            "dsp" => new DspObjectSource(w),
            "neural" => new NeuralObjectSource(w, ResolveModelDir(Arg("model", settings.ModelName)), settings),
            _ => throw new ArgumentException($"unknown source '{kind}'"),
        };
    }

    private static string SourceKind(AppSettings settings)
        => Arg("source", Flags.Contains("neural") ? "neural" : "dsp").ToLowerInvariant();

    // ------------------------------------------------------------- analyze
    private static int Analyze()
    {
        string input = Require("in");
        string output = Arg("out", "");
        if (output.Length == 0) output = Path.ChangeExtension(input, ".jsonl");
        var settings = LoadSettings();
        var w = Weights();
        var wav = WavIo.Read(input);
        var stereo = WavIo.ToStereo(wav.Interleaved, wav.Channels);
        if (wav.SampleRate != FrontendWeights.SampleRate)
            stereo = HermiteResampler.Resample(stereo, wav.SampleRate, FrontendWeights.SampleRate);
        Console.WriteLine($"input: {Path.GetFileName(input)} {wav.SampleRate} Hz {wav.Channels} ch {wav.Duration:F2} s");

        var source = MakeSource(SourceKind(settings), w, settings);
        var pipeline = new AnalysisPipeline(w, source);
        bool debug = Flags.Contains("debug");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
        int frames = 0, objects = 0;
        using (var writer = new FrameJson.Writer(output))
        {
            foreach (var frame in pipeline.Push(stereo))
            {
                if (debug && frame.Debug == null)
                    frame.Debug = new DebugInfo { BandsDb = Array.Empty<float>() };
                writer.Write(frame);
                objects += frame.Objects.Count;
                frames++;
            }
        }
        Console.WriteLine($"frames: {frames}, avg objects: {(frames > 0 ? objects / (double)frames : 0):F2}");
        Console.WriteLine($"out: {output}");
        return 0;
    }

    // ------------------------------------------------------------- render
    private static int Render()
    {
        string framesPath = Require("frames");
        string outDir = Require("out");
        var settings = LoadSettings();
        var view = Arg("view", "main").ToLowerInvariant();
        settings.View = view switch
        {
            "spectro" => DebugView.Spectro,
            "masks" => DebugView.Masks,
            "table" => DebugView.Table,
            "embedding" => DebugView.Embedding,
            _ => DebugView.Main,
        };
        int every = Math.Max(1, int.Parse(Arg("every", "10"), CultureInfo.InvariantCulture));
        int bw = int.Parse(Arg("w", "960"), CultureInfo.InvariantCulture);
        int bh = int.Parse(Arg("h", "360"), CultureInfo.InvariantCulture);
        Console.WriteLine($"view: {settings.View}, bloom {settings.Bloom}, splat {settings.SplatSize}");
        var frames = FrameJson.ReadAll(framesPath);
        if (frames.Count == 0) throw new InvalidOperationException("no frames in input");
        Directory.CreateDirectory(outDir);
        using var renderer = new Renderer(bw, bh, Flags.Contains("warp"));
        int index = 0, written = 0;
        for (int i = 0; i < frames.Count; i += every)
        {
            renderer.Render(frames[i], settings, frames[i].T, every / 100.0);
            var rgba = renderer.ReadPixels();
            var path = Path.Combine(outDir, $"frame_{index:D4}.png");
            PngWriter.Write(path, bw, bh, rgba);
            index++; written++;
        }
        Console.WriteLine($"device: {renderer.DeviceName}{(renderer.IsWarp ? " (WARP)" : "")}");
        Console.WriteLine($"png: {written} files in {outDir}");
        return 0;
    }

    // ------------------------------------------------------------- check
    private static int Check()
    {
        string framesPath = Require("frames");
        string expectPath = Require("expect");
        string outPath = Arg("out", "");
        var frames = FrameJson.ReadAll(framesPath);
        var result = Checks.Evaluate(frames, expectPath);
        var json = JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true });
        if (outPath.Length > 0)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
            File.WriteAllText(outPath, json);
        }
        foreach (var c in result.Checks)
            Console.WriteLine($"{(c.Pass ? "PASS" : "FAIL")} {c.Type,-18} {c.Detail}");
        Console.WriteLine($"{(result.Pass ? "PASS" : "FAIL")} {Path.GetFileName(expectPath)} — {result.Passed}/{result.Total} checks");
        return result.Pass ? 0 : 1;
    }

    // ------------------------------------------------------------- bench
    private static int Bench() => BenchRunner.Run(Arg("what", "frontend"), Arg, Flags);

    // ------------------------------------------------------------- parity
    private static int Parity()
    {
        string golden = Arg("golden", AvisPaths.Golden);
        var w = Weights();
        var inputPath = Path.Combine(golden, "frontend_input.f32");
        var expectedPath = Path.Combine(golden, "frontend_expected.f32");
        bool ok = true;
        if (File.Exists(inputPath) && File.Exists(expectedPath))
        {
            var input = ReadFloats(inputPath);
            var expected = ReadFloats(expectedPath);
            var stream = new FrontendStream(w);
            var frames = stream.Push(input);
            double max = 0;
            int idx = 0;
            foreach (var f in frames)
                foreach (var v in f.Features)
                    max = Math.Max(max, Math.Abs(v - expected[idx++]));
            bool pass = max <= 2e-3;
            ok &= pass;
            Console.WriteLine($"{(pass ? "PASS" : "FAIL")} frontend max|diff|={max:E3} (limit 2e-3)");
        }
        else Console.WriteLine("SKIP frontend golden missing");

        string modelDir = ResolveModelDir(Arg("model", Path.Combine(AvisPaths.Models, "v1")));
        string gdir = Path.Combine(golden, "model");
        if (File.Exists(Path.Combine(modelDir, "model.onnx")) && File.Exists(Path.Combine(gdir, "input.f32")))
        {
            var settings = LoadSettings();
            var src = new NeuralObjectSource(w, modelDir, settings);
            var input = ReadFloats(Path.Combine(gdir, "input.f32"));
            var expEmb = ReadFloats(Path.Combine(gdir, "expected_emb.f32"));
            var expAct = ReadFloats(Path.Combine(gdir, "expected_act.f32"));
            var expPitch = ReadFloats(Path.Combine(gdir, "expected_pitch.f32"));
            var expSal = ReadFloats(Path.Combine(gdir, "expected_sal.f32"));
            int frames = input.Length / (3 * 192);
            double maxEmb = 0, maxAct = 0, maxPitch = 0, maxSal = 0;
            var feat = new float[3 * 192];
            var emb = new float[192][];
            var act = new float[192]; var pitch = new float[192]; var sal = new float[192];
            for (int fi = 0; fi < frames; fi++)
            {
                Array.Copy(input, fi * 3 * 192, feat, 0, feat.Length);
                src.RunRaw(feat, emb, act, pitch, sal);
                for (int k = 0; k < 192; k++)
                {
                    for (int d = 0; d < 20; d++)
                        maxEmb = Math.Max(maxEmb, Math.Abs(emb[k][d] - expEmb[(fi * 192 + k) * 20 + d]));
                    maxAct = Math.Max(maxAct, Math.Abs(act[k] - expAct[fi * 192 + k]));
                    maxPitch = Math.Max(maxPitch, Math.Abs(pitch[k] - expPitch[fi * 192 + k]));
                    maxSal = Math.Max(maxSal, Math.Abs(sal[k] - expSal[fi * 192 + k]));
                }
            }
            double worst = Math.Max(Math.Max(maxEmb, maxAct), Math.Max(maxPitch, maxSal));
            bool pass = worst <= 1e-3;
            ok &= pass;
            Console.WriteLine($"{(pass ? "PASS" : "FAIL")} model max|diff|={worst:E3} emb={maxEmb:E2} act={maxAct:E2} pitch={maxPitch:E2} sal={maxSal:E2} (limit 1e-3, {frames} frames)");
        }
        else Console.WriteLine("SKIP model parity (model.onnx or golden model io missing)");
        return ok ? 0 : 1;
    }

    // ------------------------------------------------------------- capture
    private static int CaptureTest()
    {
        double seconds = double.Parse(Arg("seconds", "5"), CultureInfo.InvariantCulture);
        string outPath = Require("out");
        using var capture = new Avis.Audio.LoopbackCapture();
        Console.WriteLine($"device: {capture.DeviceName}");
        var samples = capture.Record(seconds);
        WavIo.Write(outPath, samples, 2, FrontendWeights.SampleRate);
        double peak = samples.Length > 0 ? samples.Max(Math.Abs) : 0;
        Console.WriteLine($"captured {samples.Length / 2} frames, peak {peak:F4}, out {outPath}");
        return 0;
    }

    internal static float[] ReadFloats(string path)
    {
        var raw = File.ReadAllBytes(path);
        var data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return data;
    }
}

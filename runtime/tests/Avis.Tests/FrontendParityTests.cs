using System;
using System.IO;
using Avis.Core;
using Xunit;

namespace Avis.Tests;

public class FrontendParityTests
{
    [Fact]
    public void CSharpFrontendMatchesPythonGolden()
    {
        var weights = new FrontendWeights(TestPaths.Weights);
        var inputPath = Path.Combine(TestPaths.Golden, "frontend_input.f32");
        var expectedPath = Path.Combine(TestPaths.Golden, "frontend_expected.f32");
        Assert.True(File.Exists(inputPath), inputPath);
        Assert.True(File.Exists(expectedPath), expectedPath);

        var raw = File.ReadAllBytes(inputPath);
        var input = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, input, 0, raw.Length);

        var stream = new FrontendStream(weights);
        var frames = stream.Push(input);
        Assert.Equal(300, frames.Count);

        var expRaw = File.ReadAllBytes(expectedPath);
        var expected = new float[expRaw.Length / 4];
        Buffer.BlockCopy(expRaw, 0, expected, 0, expRaw.Length);
        Assert.Equal(300 * 3 * 192, expected.Length);

        double maxAbs = 0;
        int count = 0;
        for (int t = 0; t < frames.Count; t++)
        {
            var f = frames[t].Features;
            for (int i = 0; i < f.Length; i++)
            {
                double d = Math.Abs(f[i] - expected[count++]);
                if (double.IsNaN(d)) throw new Xunit.Sdk.XunitException($"NaN at frame {t} index {i}");
                if (d > maxAbs) maxAbs = d;
            }
        }
        Assert.True(maxAbs <= 2e-3, $"max abs diff {maxAbs:E3} exceeds 2e-3");
    }

    [Fact]
    public void OneKilohertzSineIsAboutMinusSixDb()
    {
        var weights = new FrontendWeights(TestPaths.Weights);
        var stream = new FrontendStream(weights);
        int sr = FrontendWeights.SampleRate;
        int n = sr * 2;
        var x = new float[n * 2];
        for (int i = 0; i < n; i++)
        {
            float v = (float)Math.Sin(2 * Math.PI * 1000.0 * i / sr);
            x[2 * i] = v;
            x[2 * i + 1] = v;
        }
        var frames = stream.Push(x);
        Assert.True(frames.Count > 20);
        var last = frames[^1];
        int k = 0;
        float best = -200;
        for (int i = 0; i < 192; i++) if (last.BandsDbM[i] > best) { best = last.BandsDbM[i]; k = i; }
        Assert.InRange(weights.Fc[k], 950, 1050);
        Assert.InRange(best, -7.2, -4.8);
    }

    [Fact]
    public void WhiteNoiseHasNoResolutionSteps()
    {
        var weights = new FrontendWeights(TestPaths.Weights);
        var stream = new FrontendStream(weights);
        int sr = FrontendWeights.SampleRate;
        int n = sr * 3;
        var rng = new Random(7);
        var x = new float[n * 2];
        for (int i = 0; i < n; i++)
        {
            float v = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
            x[2 * i] = v;
            x[2 * i + 1] = v;
        }
        var frames = stream.Push(x);
        var acc = new double[192];
        int used = 0;
        for (int t = frames.Count - 50; t < frames.Count; t++)
        {
            if (t < 0) continue;
            for (int k = 0; k < 192; k++) acc[k] += frames[t].BandsDbM[k];
            used++;
        }
        Assert.True(used > 0);
        for (int k = 0; k < 192; k++) acc[k] /= used;
        // 1/3-octave smoothing of the average spectrum removes the noise floor ripple;
        // then no neighbouring band pair may jump more than 1.5 dB.
        var smooth = new double[192];
        for (int k = 0; k < 192; k++)
        {
            double s = 0; int c = 0;
            for (int j = Math.Max(0, k - 10); j <= Math.Min(191, k + 10); j++) { s += acc[j]; c++; }
            smooth[k] = s / c;
        }
        double maxStep = 0;
        for (int k = 1; k < 192; k++) maxStep = Math.Max(maxStep, Math.Abs(smooth[k] - smooth[k - 1]));
        Assert.True(maxStep <= 1.5, $"max smoothed step {maxStep:F2} dB > 1.5 dB");
    }
}

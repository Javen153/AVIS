using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Text;

namespace Avis.Core;

/// <summary>FE-v1: loads Python-generated weights.bin and streams causal 10 ms frames.</summary>
public sealed class FrontendWeights
{
    public const int Bands = 192;
    public const int SampleRate = 48000;
    public const int Hop = 480;
    public static readonly int[] Resolutions = { 4096, 2048, 1024, 512 };
    public const uint Magic = 0x41564657;

    public int[] Nfft { get; } = new int[Bands];
    public int[][] Bins { get; } = new int[Bands][];
    public float[][] Weights { get; } = new float[Bands][];
    public float[] Fc { get; } = new float[Bands];

    public FrontendWeights(string path, float fmin = 30f, float fmax = 16000f)
    {
        for (int k = 0; k < Bands; k++)
            Fc[k] = fmin * MathF.Pow(fmax / fmin, k / (float)(Bands - 1));
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false);
        uint magic = br.ReadUInt32();
        uint ver = br.ReadUInt32();
        uint bands = br.ReadUInt32();
        if (magic != Magic || ver != 1 || bands != Bands)
            throw new InvalidDataException($"bad weights.bin header magic={magic:X} ver={ver} bands={bands}");
        for (int k = 0; k < Bands; k++)
        {
            Nfft[k] = br.ReadInt32();
            int count = br.ReadInt32();
            if (count < 2 || count > 64) throw new InvalidDataException($"bad band {k} count {count}");
            var bins = new int[count];
            for (int i = 0; i < count; i++) bins[i] = br.ReadInt32();
            var w = new float[count];
            for (int i = 0; i < count; i++) w[i] = br.ReadSingle();
            Bins[k] = bins;
            Weights[k] = w;
        }
    }
}

public sealed class FrontendStream
{
    private readonly FrontendWeights _w;
    private readonly FftPlan[] _plans;
    private readonly float[][] _windows;
    private readonly float[] _sumW;
    private readonly float[] _l;
    private readonly float[] _r;
    private readonly float[] _m;
    private readonly float[] _s;
    private readonly float[] _powL, _powR, _powM, _powS;
    private float[] _buf;
    private int _len;
    private long _frameIndex;
    private const int N0 = 4096;

    public FrontendStream(FrontendWeights weights)
    {
        _w = weights;
        _plans = new FftPlan[FrontendWeights.Resolutions.Length];
        _windows = new float[FrontendWeights.Resolutions.Length][];
        _sumW = new float[FrontendWeights.Resolutions.Length];
        for (int i = 0; i < _plans.Length; i++)
        {
            int n = FrontendWeights.Resolutions[i];
            _plans[i] = new FftPlan(n);
            var win = new float[n];
            double sum = 0;
            for (int j = 0; j < n; j++)
            {
                win[j] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * j / n));
                sum += win[j];
            }
            _windows[i] = win;
            _sumW[i] = (float)(sum * sum);
        }
        int maxBins = 4096 / 2 + 1;
        _l = new float[N0];
        _r = new float[N0];
        _m = new float[N0];
        _s = new float[N0];
        _powL = new float[maxBins];
        _powR = new float[maxBins];
        _powM = new float[maxBins];
        _powS = new float[maxBins];
        _buf = new float[(N0 + 48000) * 2];
        // history represents samples [-N0+HOP, 0)
        _len = (N0 - FrontendWeights.Hop) * 2;
        _frameIndex = 0;
    }

    public long FrameIndex => _frameIndex;

    public void Reset()
    {
        Array.Clear(_buf, 0, _len);
        _len = (N0 - FrontendWeights.Hop) * 2;
        _frameIndex = 0;
    }

    /// <summary>Push interleaved stereo float samples; returns completed frames.</summary>
    public List<FrontendFrame> Push(ReadOnlySpan<float> stereoInterleaved)
    {
        int add = stereoInterleaved.Length;
        if (add == 0) return new List<FrontendFrame>(0);
        if (_len + add > _buf.Length)
        {
            int cap = Math.Max(_buf.Length * 2, _len + add);
            Array.Resize(ref _buf, cap);
        }
        stereoInterleaved.CopyTo(_buf.AsSpan(_len));
        _len += add;
        var frames = new List<FrontendFrame>(Math.Max(1, (_len / 2) / FrontendWeights.Hop));
        while (_len >= N0 * 2)
        {
            frames.Add(ProcessFrame());
            int shift = FrontendWeights.Hop * 2;
            Array.Copy(_buf, shift, _buf, 0, _len - shift);
            _len -= shift;
            _frameIndex++;
        }
        return frames;
    }

    private FrontendFrame ProcessFrame()
    {
        // deinterleave the first N0 samples of the buffer
        for (int i = 0; i < N0; i++)
        {
            _l[i] = _buf[2 * i];
            _r[i] = _buf[2 * i + 1];
            _m[i] = 0.5f * (_l[i] + _r[i]);
            _s[i] = 0.5f * (_l[i] - _r[i]);
        }
        var pm = new float[FrontendWeights.Bands];
        var ps = new float[FrontendWeights.Bands];
        var pl = new float[FrontendWeights.Bands];
        var pr = new float[FrontendWeights.Bands];
        for (int ri = 0; ri < _plans.Length; ri++)
        {
            int n = FrontendWeights.Resolutions[ri];
            var plan = _plans[ri];
            plan.ForwardReal4(_m, _l, _r, _s, N0 - n, _windows[ri]);
            float denom = _sumW[ri];
            int half = n / 2;
            for (int b = 0; b <= half; b++)
            {
                Vector4 re = plan.Re[b], im = plan.Im[b];
                _powM[b] = (re.X * re.X + im.X * im.X) / denom;
                _powL[b] = (re.Y * re.Y + im.Y * im.Y) / denom;
                _powR[b] = (re.Z * re.Z + im.Z * im.Z) / denom;
                _powS[b] = (re.W * re.W + im.W * im.W) / denom;
            }
            for (int k = 0; k < FrontendWeights.Bands; k++)
            {
                if (_w.Nfft[k] != n) continue;
                var bins = _w.Bins[k];
                var w = _w.Weights[k];
                float sm = 0, ss = 0, sl = 0, sr = 0;
                for (int j = 0; j < bins.Length; j++)
                {
                    int b = bins[j];
                    float wj = w[j];
                    sm += wj * _powM[b];
                    ss += wj * _powS[b];
                    sl += wj * _powL[b];
                    sr += wj * _powR[b];
                }
                pm[k] = sm; ps[k] = ss; pl[k] = sl; pr[k] = sr;
            }
        }
        var dbm = new float[FrontendWeights.Bands];
        var feat = new float[3 * FrontendWeights.Bands];
        for (int k = 0; k < FrontendWeights.Bands; k++)
        {
            dbm[k] = 10f * MathF.Log10(pm[k] + 1e-12f);
            float dbs = 10f * MathF.Log10(ps[k] + 1e-12f);
            float dbl = 10f * MathF.Log10(pl[k] + 1e-12f);
            float dbr = 10f * MathF.Log10(pr[k] + 1e-12f);
            feat[k] = (dbm[k] + 60f) / 30f;
            feat[FrontendWeights.Bands + k] = (dbs + 60f) / 30f;
            feat[2 * FrontendWeights.Bands + k] = Math.Clamp((dbl - dbr) / 20f, -1f, 1f);
        }
        return new FrontendFrame
        {
            TimeSec = _frameIndex * FrontendWeights.Hop / (double)FrontendWeights.SampleRate,
            BandsDbM = dbm,
            PM = pm,
            PL = pl,
            PR = pr,
            Features = feat,
        };
    }

}

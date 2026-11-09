using System;
using System.Collections.Generic;
using System.Linq;

namespace Avis.Core;

/// <summary>Per-object measurements shared by the neural and DSP sources (SPEC F).</summary>
public static class ObjectParams
{
    public static float AWeightDb(double f)
    {
        double f2 = f * f;
        double ra = 12194.0 * 12194.0 * f2 * f2 /
                    ((f2 + 20.6 * 20.6) * Math.Sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) *
                     (f2 + 12194.0 * 12194.0));
        return (float)(20.0 * Math.Log10(ra) + 2.0);
    }

    public static float BandToHz(FrontendWeights w, double band)
    {
        int k = Math.Clamp((int)Math.Floor(band), 0, w.Fc.Length - 1);
        int k1 = Math.Min(k + 1, w.Fc.Length - 1);
        double frac = band - k;
        return (float)(w.Fc[k] * Math.Pow(w.Fc[k1] / (double)w.Fc[k], frac));
    }

    [ThreadStatic] private static float[]? _eScratch;
    [ThreadStatic] private static List<(int k, float strength, double delta)>? _peakScratch;
    [ThreadStatic] private static List<(int k, float strength, double delta)>? _bestScratch;

    public static ObjectState Build(FrontendFrame f, float[] membership, int id, float[] color,
        float[]? embedding, float ageSec, bool isTransient, FrontendWeights w,
        float[] pitch, float[] sal, float presenceOverride = -1)
    {
        int bands = w.Fc.Length;
        var e = _eScratch != null && _eScratch.Length == bands ? _eScratch : (_eScratch = new float[bands]);
        double total = 0, atotal = 0;
        for (int k = 0; k < bands; k++)
        {
            e[k] = membership[k] * f.PM[k];
            total += e[k];
            double a = Math.Pow(10.0, AWeightDb(w.Fc[k]) / 20.0);
            atotal += e[k] * a;
        }
        float loud = (float)(10.0 * Math.Log10(atotal + 1e-12));
        float presence = presenceOverride >= 0 ? presenceOverride : Math.Clamp((loud + 70f) / 20f, 0f, 1f);

        // partial loudness: x^0.23 compression, subtract 30% of the rest
        double partialSum = 0;
        for (int k = 0; k < bands; k++)
        {
            double nOwn = Math.Pow(Math.Max(e[k], 0), 0.23);
            double rest = Math.Max(f.PM[k] - e[k], 0);
            double nRest = Math.Pow(rest, 0.23);
            partialSum += Math.Max(0.0, nOwn - 0.3 * nRest);
        }
        float partial = (float)(10.0 * Math.Log10(Math.Pow(partialSum, 1.0 / 0.23) + 1e-12));

        // salience weighted by energy
        double sw = 0, sv = 0;
        for (int k = 0; k < bands; k++) { sw += e[k]; sv += e[k] * sal[k]; }
        float salience = (sw > 1e-12 ? (float)(sv / sw) : 0f) * presence;
        salience = Math.Clamp(salience, 0f, 1f);

        // pitches: local maxima of pitch*membership, refined by two parabolas
        var peaks = _peakScratch ??= new List<(int k, float strength, double delta)>();
        peaks.Clear();
        for (int k = 1; k < bands - 1; k++)
        {
            float y1 = pitch[k] * membership[k];
            if (y1 <= 0.3f) continue;
            if (y1 < pitch[k - 1] * membership[k - 1] || y1 < pitch[k + 1] * membership[k + 1]) continue;
            double d2 = Parabola(f.BandsDbM[k - 1], f.BandsDbM[k], f.BandsDbM[k + 1]);
            peaks.Add((k, y1, Math.Clamp(d2, -0.5, 0.5)));
        }
        var bestPeaks = _bestScratch ??= new List<(int k, float strength, double delta)>();
        bestPeaks.Clear();
        Span<bool> chosen = stackalloc bool[peaks.Count];
        for (int pick = 0; pick < 6; pick++)
        {
            int bi = -1; float bs = -1;
            for (int pi = 0; pi < peaks.Count; pi++)
            {
                if (!chosen[pi] && peaks[pi].strength > bs)
                {
                    bs = peaks[pi].strength; bi = pi;
                }
            }
            if (bi < 0) break;
            chosen[bi] = true;
            bestPeaks.Add(peaks[bi]);
        }
        bestPeaks.Sort((a, b) => a.k.CompareTo(b.k));
        var pitches = new List<PitchEstimate>(bestPeaks.Count);
        float maxP = 0f;
        foreach (var p in bestPeaks)
        {
            pitches.Add(new PitchEstimate { Hz = BandToHz(w, p.k + p.delta), Strength = Math.Clamp(p.strength, 0f, 1f) });
            if (p.strength > maxP) maxP = p.strength;
        }

        double logSum = 0; int nz = 0; double sum = 0;
        for (int k = 0; k < bands; k++)
        {
            if (membership[k] <= 0) continue;
            double v = Math.Max(e[k], 1e-20);
            logSum += Math.Log(v); sum += v; nz++;
        }
        double flat = nz > 0 ? Math.Exp(logSum / nz) / (sum / nz + 1e-20) : 1.0;
        float tonality = Math.Clamp(maxP * (float)(1.0 - flat), 0f, 1f);

        // centroid and percentiles in log-frequency (5/95 keeps quiet but real
        // spectral edges visible: white noise spans ~87 % of the axis, D-017)
        double le = 0, ee = 0;
        for (int k = 0; k < bands; k++) { le += e[k] * Math.Log(w.Fc[k]); ee += e[k]; }
        float centroid = ee > 1e-12 ? (float)Math.Exp(le / ee) : 0f;
        double cum = 0, lowT = ee * 0.05, highT = ee * 0.95; float low = centroid, high = centroid;
        bool lowSet = false, highSet = false;
        for (int k = 0; k < bands; k++)
        {
            cum += e[k];
            if (!lowSet && cum >= lowT) { low = w.Fc[k]; lowSet = true; }
            if (cum >= highT) { high = w.Fc[k]; highSet = true; break; }
        }
        if (!lowSet) low = centroid;
        if (!highSet) high = Math.Max(low, centroid);

        var env = new float[32];
        for (int g = 0; g < 32; g++)
        {
            double s = 0;
            for (int k = g * 6; k < Math.Min((g + 1) * 6, bands); k++) s += e[k];
            env[g] = (float)s;
        }

        double panNum = 0;
        for (int k = 0; k < bands; k++)
        {
            if (e[k] <= 0) continue;
            double dl = 10.0 * Math.Log10(f.PL[k] + 1e-12);
            double dr = 10.0 * Math.Log10(f.PR[k] + 1e-12);
            panNum += e[k] * (dl - dr) / 20.0;
        }
        float pan = ee > 1e-12 ? (float)Math.Clamp(panNum / ee, -1, 1) : 0f;

        return new ObjectState
        {
            Id = id,
            Color = color,
            Presence = presence,
            State = "active",
            LoudnessDb = loud,
            PartialLoudness = partial,
            Salience = salience,
            Tonality = tonality,
            Pitches = pitches,
            CentroidHz = centroid,
            LowHz = low,
            HighHz = high,
            Envelope = env,
            Pan = pan,
            IsTransient = isTransient,
            AgeSec = ageSec,
            Embedding = embedding,
        };
    }

    private static double Parabola(double y0, double y1, double y2)
    {
        double den = y0 - 2 * y1 + y2;
        if (Math.Abs(den) < 1e-9) return 0;
        return 0.5 * (y0 - y2) / den;
    }

    // ------------------------------------------------------------ color
    public static float[] PaletteColor(int id)
    {
        double h = (id * 0.618033988749895) % 1.0;
        // OKLCH -> sRGB
        double L = 0.74, C = 0.15;
        return OklchToRgb(L, C, h);
    }

    public static float[] OklchToRgb(double L, double C, double h)
    {
        double a = C * Math.Cos(2 * Math.PI * h);
        double b = C * Math.Sin(2 * Math.PI * h);
        double l_ = L + 0.3963377774 * a + 0.2158037573 * b;
        double m_ = L - 0.1055613458 * a - 0.0638541728 * b;
        double s_ = L - 0.0894841775 * a - 1.2914855480 * b;
        double l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
        double r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
        double g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
        double bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;
        return new[] { Gamma(r), Gamma(g), Gamma(bb) };
    }

    private static float Gamma(double x)
    {
        double c = Math.Clamp(x, 0, 1);
        return (float)(c <= 0.0031308 ? 12.92 * c : 1.055 * Math.Pow(c, 1 / 2.4) - 0.055);
    }
}

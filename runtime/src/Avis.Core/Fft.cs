using System;

namespace Avis.Core;

/// <summary>Allocation-free radix-2 complex FFT with precomputed twiddles.</summary>
public sealed class FftPlan
{
    public int N { get; }
    private readonly int[] _rev;
    private readonly float[] _cos;
    private readonly float[] _sin;
    public float[] Re { get; }
    public float[] Im { get; }

    public FftPlan(int n)
    {
        if (n < 2 || (n & (n - 1)) != 0) throw new ArgumentException("FFT size must be a power of two");
        N = n;
        _rev = new int[n];
        int bits = 0;
        while ((1 << bits) < n) bits++;
        for (int i = 0; i < n; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++) if ((i & (1 << b)) != 0) r |= 1 << (bits - 1 - b);
            _rev[i] = r;
        }
        _cos = new float[n / 2];
        _sin = new float[n / 2];
        for (int i = 0; i < n / 2; i++)
        {
            _cos[i] = (float)Math.Cos(2.0 * Math.PI * i / n);
            _sin[i] = (float)Math.Sin(2.0 * Math.PI * i / n);
        }
        Re = new float[n];
        Im = new float[n];
    }

    /// <summary>Real signal of length N (zero-padded if shorter) -> full complex spectrum.</summary>
    public void ForwardReal(ReadOnlySpan<float> signal)
    {
        int n = N;
        for (int i = 0; i < n; i++)
        {
            int j = _rev[i];
            Re[j] = i < signal.Length ? signal[i] : 0f;
            Im[j] = 0f;
        }
        for (int len = 2; len <= n; len <<= 1)
        {
            int half = len >> 1;
            int step = n / len;
            for (int start = 0; start < n; start += len)
            {
                int k = 0;
                for (int j = start; j < start + half; j++)
                {
                    float wr = _cos[k], wi = -_sin[k];
                    float ur = Re[j], ui = Im[j];
                    float vr = Re[j + half], vi = Im[j + half];
                    float tr = vr * wr - vi * wi;
                    float ti = vr * wi + vi * wr;
                    Re[j] = ur + tr;
                    Im[j] = ui + ti;
                    Re[j + half] = ur - tr;
                    Im[j + half] = ui - ti;
                    k += step;
                }
            }
        }
    }

    public float Power(int bin) => Re[bin] * Re[bin] + Im[bin] * Im[bin];
}

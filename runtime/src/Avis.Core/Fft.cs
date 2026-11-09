using System;
using System.Numerics;

namespace Avis.Core;

/// <summary>
/// Allocation-free radix-2 FFT for four real signals at once: the four channels are
/// packed into Vector4 lanes, so one butterfly pass serves M/L/R/S. Precomputed
/// twiddles and bit-reversal tables. Numerically identical per lane to a scalar FFT.
/// </summary>
public sealed class FftPlan
{
    public int N { get; }
    private readonly int[] _rev;
    private readonly float[] _cos;
    private readonly float[] _sin;
    public Vector4[] Re { get; }
    public Vector4[] Im { get; }

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
        Re = new Vector4[n];
        Im = new Vector4[n];
    }

    /// <summary>
    /// Window and transform four real signals: lane X=s0, Y=s1, Z=s2, W=s3.
    /// Each signal is read as [offset, offset+N) and multiplied by <paramref name="window"/>.
    /// </summary>
    public void ForwardReal4(float[] s0, float[] s1, float[] s2, float[] s3, int offset, float[] window)
    {
        int n = N;
        for (int i = 0; i < n; i++)
        {
            int j = _rev[i];
            Re[j] = new Vector4(s0[offset + i], s1[offset + i], s2[offset + i], s3[offset + i])
                    * new Vector4(window[i]);
            Im[j] = Vector4.Zero;
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
                    // Complex multiply by (wr, wi) for every lane.
                    var wr = new Vector4(_cos[k]);
                    var wi = new Vector4(-_sin[k]);
                    Vector4 ur = Re[j], ui = Im[j];
                    Vector4 vr = Re[j + half], vi = Im[j + half];
                    Vector4 tr = vr * wr - vi * wi;
                    Vector4 ti = vr * wi + vi * wr;
                    Re[j] = ur + tr;
                    Im[j] = ui + ti;
                    Re[j + half] = ur - tr;
                    Im[j + half] = ui - ti;
                    k += step;
                }
            }
        }
    }
}

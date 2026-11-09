using System;

namespace Avis.Core;

/// <summary>4-point Hermite resampler for interleaved stereo (48 kHz target).</summary>
public static class HermiteResampler
{
    public static float[] Resample(float[] source, int inRate, int outRate)
    {
        if (inRate == outRate || source.Length == 0) return source;
        int inFrames = source.Length / 2;
        int outFrames = (int)Math.Round(inFrames * (double)outRate / inRate);
        var output = new float[outFrames * 2];
        double step = (double)inRate / outRate;
        for (int i = 0; i < outFrames; i++)
        {
            double pos = i * step;
            int i1 = (int)Math.Floor(pos);
            double t = pos - i1;
            for (int c = 0; c < 2; c++)
            {
                float y0 = Sample(source, i1 - 1, c, inFrames);
                float y1 = Sample(source, i1, c, inFrames);
                float y2 = Sample(source, i1 + 1, c, inFrames);
                float y3 = Sample(source, i1 + 2, c, inFrames);
                output[2 * i + c] = Hermite(y0, y1, y2, y3, (float)t);
            }
        }
        return output;
    }

    private static float Sample(float[] src, int frame, int channel, int frames)
    {
        if (frame < 0) frame = 0;
        if (frame >= frames) frame = frames - 1;
        return src[2 * frame + channel];
    }

    public static float Hermite(float y0, float y1, float y2, float y3, float t)
    {
        float c0 = y1;
        float c1 = 0.5f * (y2 - y0);
        float c2 = y0 - 2.5f * y1 + 2f * y2 - 0.5f * y3;
        float c3 = 0.5f * (y3 - y0) + 1.5f * (y1 - y2);
        return ((c3 * t + c2) * t + c1) * t + c0;
    }
}

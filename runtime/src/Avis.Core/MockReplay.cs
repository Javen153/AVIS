using System;
using System.Collections.Generic;
using System.Linq;

namespace Avis.Core;

/// <summary>Synthetic objects for UI/render work without audio or a model.</summary>
public sealed class MockObjectSource : IObjectSource
{
    public string Name => "mock";
    private readonly int _count;
    public MockObjectSource(int count = 5) { _count = count; }
    public void Reset() { }

    public AnalysisFrame Process(in FrontendFrame f, double timeSec)
    {
        var frame = new AnalysisFrame { T = timeSec, Source = Name, Objects = new List<ObjectState>() };
        for (int i = 0; i < _count; i++)
        {
            int id = i + 1;
            double fc = 60 * Math.Pow(2.0, (i * 1.37 + 0.15 * Math.Sin(timeSec * (0.7 + 0.13 * i))) * 2.2);
            fc = Math.Clamp(fc, 40, 14000);
            float loud = (float)(-14 - 6 * i + 4 * Math.Sin(timeSec * (1.1 + 0.2 * i) + i));
            bool noise = i == _count - 1;
            var st = new ObjectState
            {
                Id = id,
                Color = ObjectParams.PaletteColor(id),
                Presence = Math.Clamp((loud + 70) / 20f, 0, 1),
                State = "active",
                LoudnessDb = loud,
                PartialLoudness = loud - 4,
                Salience = (float)(0.4 + 0.3 * Math.Sin(timeSec + i)),
                Tonality = noise ? 0.1f : 0.85f,
                CentroidHz = (float)fc,
                LowHz = (float)(fc * (noise ? 0.6 : 0.97)),
                HighHz = (float)(fc * (noise ? 2.2 : 1.04)),
                Pan = (float)(0.5 * Math.Sin(timeSec * 0.3 + i)),
                IsTransient = i == 2 && Math.Sin(timeSec * 8) > 0.93,
                AgeSec = (float)timeSec,
                Embedding = Enumerable.Range(0, 20).Select(d => (float)Math.Sin(id * 0.7 + d)).ToArray(),
            };
            if (!noise) st.Pitches.Add(new PitchEstimate { Hz = (float)fc, Strength = 0.85f });
            st.Envelope = Enumerable.Range(0, 32).Select(g => (float)Math.Exp(-Math.Abs(g - (noise ? 16 : i * 4 + 4)) / 5.0)).ToArray();
            frame.Objects.Add(st);
        }
        return frame;
    }
}

/// <summary>Replays a JSONL frame dump (used by render/check without a model).</summary>
public sealed class ReplayObjectSource : IObjectSource
{
    public string Name => "replay";
    private readonly List<AnalysisFrame> _frames;
    private int _idx;
    public ReplayObjectSource(IEnumerable<AnalysisFrame> frames) { _frames = frames.ToList(); }
    public int Count => _frames.Count;
    public void Reset() => _idx = 0;

    public AnalysisFrame Process(in FrontendFrame f, double timeSec)
    {
        if (_frames.Count == 0)
            return new AnalysisFrame { T = timeSec, Source = Name, Objects = new List<ObjectState>() };
        var fr = _frames[Math.Min(_idx, _frames.Count - 1)];
        _idx++;
        return fr;
    }
}

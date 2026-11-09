using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text.Json;
using Avis.Core;

namespace Avis.Core;

/// <summary>Frontend -> object source chain shared by the CLI and the real-time app.</summary>
public sealed class AnalysisPipeline
{
    private readonly FrontendStream _frontend;
    private IObjectSource _source;
    public AnalysisPipeline(FrontendWeights weights, IObjectSource source)
    {
        _frontend = new FrontendStream(weights);
        _source = source;
    }
    public IObjectSource Source => _source;
    public void SetSource(IObjectSource source) { _source = source; _source.Reset(); _frontend.Reset(); }
    public void Reset() { _frontend.Reset(); _source.Reset(); }
    public List<AnalysisFrame> Push(ReadOnlySpan<float> stereo)
    {
        var frames = _frontend.Push(stereo);
        var result = new List<AnalysisFrame>(frames.Count);
        foreach (var f in frames) result.Add(_source.Process(f, f.TimeSec));
        return result;
    }
}

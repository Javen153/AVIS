using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Avis.Core;

public static class FrameJson
{
    public static readonly JsonSerializerOptions Options = new()
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter() },
    };

    public static string ToLine(AnalysisFrame frame) => JsonSerializer.Serialize(frame, Options);
    public static AnalysisFrame FromLine(string line) =>
        JsonSerializer.Deserialize<AnalysisFrame>(line, Options) ?? throw new InvalidDataException("empty frame");

    public sealed class Writer : IDisposable
    {
        private readonly StreamWriter _w;
        public Writer(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            _w = new StreamWriter(path, false, new UTF8Encoding(false));
        }
        public void Write(AnalysisFrame f) => _w.WriteLine(ToLine(f));
        public void Dispose() => _w.Dispose();
    }

    public static List<AnalysisFrame> ReadAll(string path)
    {
        var list = new List<AnalysisFrame>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            list.Add(FromLine(line));
        }
        return list;
    }
}

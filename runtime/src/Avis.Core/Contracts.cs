using System.Numerics;
using System.Text.Json.Serialization;

namespace Avis.Core;

/// <summary>One FE-v1 hop: band powers in four channel combinations.</summary>
public sealed class FrontendFrame
{
    public double TimeSec { get; init; }
    public required float[] BandsDbM { get; init; }
    public required float[] PM { get; init; }
    public required float[] PL { get; init; }
    public required float[] PR { get; init; }
    /// <summary>[3,192] flatten: feat0, feat1, feat2 rows.</summary>
    public required float[] Features { get; init; }
    public int Bands => BandsDbM.Length;
}

public interface IObjectSource
{
    string Name { get; }
    void Reset();
    AnalysisFrame Process(in FrontendFrame f, double timeSec);
}

public sealed class PitchEstimate
{
    [JsonPropertyName("hz")] public float Hz { get; set; }
    [JsonPropertyName("strength")] public float Strength { get; set; }
}

public sealed class ObjectState
{
    [JsonPropertyName("id")] public int Id { get; set; }
    [JsonPropertyName("color")] public float[] Color { get; set; } = new float[3];
    [JsonPropertyName("presence")] public float Presence { get; set; }
    [JsonPropertyName("state")] public string State { get; set; } = "active";
    [JsonPropertyName("loudnessDb")] public float LoudnessDb { get; set; }
    [JsonPropertyName("partialLoudness")] public float PartialLoudness { get; set; }
    [JsonPropertyName("salience")] public float Salience { get; set; }
    [JsonPropertyName("tonality")] public float Tonality { get; set; }
    [JsonPropertyName("pitches")] public List<PitchEstimate> Pitches { get; set; } = new();
    [JsonPropertyName("centroidHz")] public float CentroidHz { get; set; }
    [JsonPropertyName("lowHz")] public float LowHz { get; set; }
    [JsonPropertyName("highHz")] public float HighHz { get; set; }
    [JsonPropertyName("envelope")] public float[] Envelope { get; set; } = new float[32];
    [JsonPropertyName("pan")] public float Pan { get; set; }
    [JsonPropertyName("isTransient")] public bool IsTransient { get; set; }
    [JsonPropertyName("ageSec")] public float AgeSec { get; set; }
    [JsonPropertyName("embedding")] public float[]? Embedding { get; set; }
}

public sealed class DebugInfo
{
    [JsonPropertyName("bandsDb")] public float[] BandsDb { get; set; } = Array.Empty<float>();
    [JsonPropertyName("ownerId")] public int[] OwnerId { get; set; } = Array.Empty<int>();
    [JsonPropertyName("ownerWeight")] public float[] OwnerWeight { get; set; } = Array.Empty<float>();
    [JsonPropertyName("embedding2d")] public float[]? Embedding2d { get; set; }
}

public sealed class AnalysisFrame
{
    public const int CurrentSchema = 1;
    [JsonPropertyName("schema")] public int Schema { get; set; } = CurrentSchema;
    [JsonPropertyName("t")] public double T { get; set; }
    [JsonPropertyName("source")] public string Source { get; set; } = "mock";
    [JsonPropertyName("objects")] public List<ObjectState> Objects { get; set; } = new();
    [JsonPropertyName("debug")] public DebugInfo? Debug { get; set; }

    public ObjectState? Find(int id) => Objects.FirstOrDefault(o => o.Id == id);
}

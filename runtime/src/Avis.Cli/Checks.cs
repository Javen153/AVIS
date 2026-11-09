using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avis.Core;

internal sealed class CheckResult
{
    public string Type { get; set; } = "";
    public bool Pass { get; set; }
    public string Detail { get; set; } = "";
}

internal sealed class ScenarioResult
{
    public string Scenario { get; set; } = "";
    public bool Pass { get; set; }
    public int Passed { get; set; }
    public int Total { get; set; }
    public int Frames { get; set; }
    public List<CheckResult> Checks { get; set; } = new();
}

internal static class Checks
{
    public static ScenarioResult Evaluate(List<AnalysisFrame> all, string expectPath)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(expectPath));
        var root = doc.RootElement;
        var win = root.GetProperty("window");
        double t0 = win[0].GetDouble(), t1 = win[1].GetDouble();
        var frames = all.Where(f => f.T >= t0 - 1e-9 && f.T <= t1 + 1e-9).ToList();
        var result = new ScenarioResult
        {
            Scenario = Path.GetFileNameWithoutExtension(expectPath).Replace(".expect", ""),
            Frames = frames.Count,
            Total = root.GetProperty("checks").GetArrayLength(),
        };
        if (frames.Count == 0)
        {
            result.Checks.Add(new CheckResult { Type = "window", Pass = false, Detail = $"no frames in [{t0},{t1}]" });
            result.Passed = 0;
            result.Total = Math.Max(1, result.Total);
            return result;
        }
        foreach (var c in root.GetProperty("checks").EnumerateArray())
        {
            var r = EvaluateOne(frames, c);
            result.Checks.Add(r);
            if (r.Pass) result.Passed++;
        }
        result.Pass = result.Passed == result.Total;
        return result;
    }

    private static CheckResult EvaluateOne(List<AnalysisFrame> frames, JsonElement c)
    {
        string type = c.GetProperty("type").GetString() ?? "";
        double frac = c.TryGetProperty("frac", out var f) ? f.GetDouble() : 1.0;
        try
        {
            switch (type)
            {
                case "count":
                {
                    int min = c.GetProperty("min").GetInt32();
                    int max = c.GetProperty("max").GetInt32();
                    double ok = frames.Count(fr => fr.Objects.Count >= min && fr.Objects.Count <= max) / (double)frames.Count;
                    return R(type, ok >= frac - 1e-9, $"in range {ok:P0} (need {frac:P0}), counts {string.Join("/", frames.Select(fr => fr.Objects.Count).GroupBy(x => x).OrderBy(g => g.Key).Select(g => $"{g.Key}x{g.Count()}"))}");
                }
                case "pitch":
                case "object_near_pitch":
                {
                    double hz = c.GetProperty("hz").GetDouble();
                    double tol = c.GetProperty("tol_cents").GetDouble();
                    double ok = frames.Count(fr => fr.Objects.Any(o => NearPitch(o, hz, tol))) / (double)frames.Count;
                    return R(type, ok >= frac - 1e-9, $"{hz:F1} Hz within {tol:F0} cents in {ok:P0} frames (need {frac:P0})");
                }
                case "pitches_present":
                {
                    var hz = c.GetProperty("hz").EnumerateArray().Select(e => e.GetDouble()).ToArray();
                    double tol = c.GetProperty("tol_cents").GetDouble();
                    bool all = true;
                    var parts = new List<string>();
                    foreach (var h in hz)
                    {
                        double ok = frames.Count(fr => fr.Objects.Any(o => NearPitch(o, h, tol))) / (double)frames.Count;
                        all &= ok >= frac - 1e-9;
                        parts.Add($"{h:F1}Hz:{ok:P0}");
                    }
                    return R(type, all, string.Join(" ", parts) + $" (need {frac:P0} each)");
                }
                case "pitch_sweep":
                case "voice_pitch":
                {
                    var curve = type == "voice_pitch" ? c.GetProperty("curve") : c;
                    double a = curve.GetProperty("f0").GetDouble(), b = curve.GetProperty("f1").GetDouble();
                    double st = curve.GetProperty("t0").GetDouble(), en = curve.GetProperty("t1").GetDouble();
                    double tol = c.GetProperty("tol_cents").GetDouble();
                    bool median = c.TryGetProperty("median", out var m) && m.GetBoolean();
                    var errs = new List<double>();
                    foreach (var fr in frames)
                    {
                        double expected = a * Math.Pow(b / a, Math.Clamp((fr.T - st) / (en - st), 0, 1));
                        double best = double.MaxValue;
                        foreach (var o in fr.Objects)
                            foreach (var p in o.Pitches)
                                best = Math.Min(best, Math.Abs(Cents(p.Hz, expected)));
                        if (best < 1e5) errs.Add(best);
                    }
                    if (errs.Count == 0) return R(type, false, "no pitches at all");
                    if (median)
                    {
                        errs.Sort();
                        double med = errs[errs.Count / 2];
                        return R(type, med <= tol, $"median error {med:F0} cents (need <= {tol:F0})");
                    }
                    double okf = errs.Count(e => e <= tol) / (double)frames.Count;
                    return R(type, okf >= frac - 1e-9, $"within {tol:F0} cents in {okf:P0} frames (need {frac:P0})");
                }
                case "id_stable":
                {
                    var perFrame = frames.Select(fr => fr.Objects.OrderByDescending(o => o.Presence).FirstOrDefault()?.Id ?? -1).ToList();
                    var dominant = perFrame.Where(i => i > 0).GroupBy(i => i).OrderByDescending(g => g.Count()).FirstOrDefault();
                    double ok = dominant == null ? 0 : perFrame.Count(i => i == dominant.Key) / (double)perFrame.Count;
                    return R(type, ok >= frac - 1e-9, $"dominant id {dominant?.Key} in {ok:P0} frames (need {frac:P0})");
                }
                case "distinct_ids":
                {
                    int min = c.TryGetProperty("min", out var mn) ? mn.GetInt32() : 0;
                    int max = c.TryGetProperty("max", out var mx) ? mx.GetInt32() : int.MaxValue;
                    int n = frames.SelectMany(fr => fr.Objects).Select(o => o.Id).Distinct().Count();
                    return R(type, n >= min && n <= max, $"{n} distinct ids (need {min}..{max})");
                }
                case "width_frac_min":
                {
                    double value = c.GetProperty("value").GetDouble();
                    double ok = frames.Count(fr => fr.Objects.Any(o => WidthFrac(o) >= value)) / (double)frames.Count;
                    return R(type, ok >= frac - 1e-9, $"width >= {value:P0} in {ok:P0} frames (need {frac:P0})");
                }
                case "tonality_max":
                case "tonality_min":
                {
                    double value = c.GetProperty("value").GetDouble();
                    bool isMax = type == "tonality_max";
                    double ok = frames.Count(fr =>
                    {
                        var objects = fr.Objects;
                        if (objects.Count == 0) return false;
                        double t = isMax ? objects.Max(o => o.Tonality) : objects.Max(o => o.Tonality);
                        return isMax ? t <= value : t >= value;
                    }) / (double)frames.Count;
                    return R(type, ok >= frac - 1e-9, $"{type.Split('_')[1]} {value:F2} in {ok:P0} frames (need {frac:P0})");
                }
                case "same_id_after_gap":
                {
                    double b0 = c.GetProperty("before")[0].GetDouble(), b1 = c.GetProperty("before")[1].GetDouble();
                    double a0 = c.GetProperty("after")[0].GetDouble(), a1 = c.GetProperty("after")[1].GetDouble();
                    var before = frames.Where(fr => fr.T >= b0 && fr.T <= b1).ToList();
                    var after = frames.Where(fr => fr.T >= a0 && fr.T <= a1).ToList();
                    if (before.Count == 0 || after.Count == 0) return R(type, false, "empty before/after window");
                    var id = before.SelectMany(fr => fr.Objects).GroupBy(o => o.Id).OrderByDescending(g => g.Sum(o => o.Presence)).First().Key;
                    double ok = after.Count(fr => fr.Objects.Any(o => o.Id == id)) / (double)after.Count;
                    return R(type, ok >= frac - 1e-9, $"id {id} returns in {ok:P0} after frames (need {frac:P0})");
                }
                case "drum_ids":
                {
                    var hits = c.GetProperty("hits").EnumerateArray().ToList();
                    var byType = new Dictionary<string, List<int>>();
                    foreach (var h in hits)
                    {
                        double t = h.GetProperty("t").GetDouble();
                        string kind = h.GetProperty("type").GetString() ?? "";
                        var fr = frames.OrderBy(f => Math.Abs(f.T - t)).First();
                        var obj = fr.Objects.Where(o => DrumMatches(o, kind)).OrderByDescending(o => o.Presence).FirstOrDefault();
                        if (obj == null) continue;
                        if (!byType.TryGetValue(kind, out var list)) byType[kind] = list = new List<int>();
                        list.Add(obj.Id);
                    }
                    bool pass = true;
                    var parts = new List<string>();
                    var dominantIds = new HashSet<int>();
                    foreach (var kind in new[] { "kick", "snare", "hihat" })
                    {
                        if (!byType.TryGetValue(kind, out var ids) || ids.Count == 0) { pass = false; parts.Add($"{kind}:none"); continue; }
                        var dom = ids.GroupBy(i => i).OrderByDescending(g => g.Count()).First();
                        double share = dom.Count() / (double)ids.Count;
                        pass &= share >= frac - 1e-9;
                        if (!dominantIds.Add(dom.Key)) pass = false;
                        parts.Add($"{kind}:id{dom.Key} {share:P0}");
                    }
                    return R(type, pass, string.Join(" ", parts) + $" (need {frac:P0} per type, distinct ids)");
                }
                default:
                    return R(type, false, "unsupported check type");
            }
        }
        catch (Exception ex)
        {
            return R(type, false, $"error: {ex.Message}");
        }
    }

    private static CheckResult R(string type, bool pass, string detail) =>
        new() { Type = type, Pass = pass, Detail = detail };

    private static double Cents(double hz, double expected) =>
        1200.0 * Math.Log2(Math.Max(1e-6, hz) / Math.Max(1e-6, expected));

    private static bool NearPitch(ObjectState o, double hz, double tol)
    {
        foreach (var p in o.Pitches)
            if (Math.Abs(Cents(p.Hz, hz)) <= tol) return true;
        // fall back to the centroid for DSP objects without an explicit pitch list
        return !o.Pitches.Any() && Math.Abs(Cents(o.CentroidHz, hz)) <= tol;
    }

    private static double WidthFrac(ObjectState o)
    {
        double lo = 30, hi = 16000;
        return Math.Clamp((o.HighHz - o.LowHz) / (hi - lo), 0, 1);
    }

    private static bool DrumMatches(ObjectState o, string kind) => kind switch
    {
        "kick" => o.CentroidHz > 0 && o.CentroidHz <= 260,
        "snare" => o.CentroidHz > 260 && o.CentroidHz <= 6000,
        "hihat" => o.CentroidHz > 6000,
        _ => false,
    };
}

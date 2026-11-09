using System;
using System.Collections.Generic;
using System.Linq;

namespace Avis.Core;

/// <summary>
/// Fallback object source: stable spectral partials are tracked over time and grouped
/// into harmonic objects; the unexplained spectrum becomes a noise object.
/// </summary>
public sealed class DspObjectSource : IObjectSource
{
    public string Name => "dsp";

    private const int Bands = 192;
    private const float FMin = 30f, FMax = 16000f;
    private const int StableHits = 12;      // ~120 ms before a partial may form an object
    private const float MatchCents = 80;    // partial association window per frame
    private const float HarmonicCents = 80; // harmonic-series membership window

    private readonly FrontendWeights _w;
    private readonly float[] _pitch = new float[Bands];
    private readonly float[] _sal = new float[Bands];
    private readonly List<Partial> _partials = new();
    private readonly List<Group> _groups = new();
    private readonly List<Group> _bank = new();
    private readonly Dictionary<string, DrumState> _drums = new();
    private int _nextId = 1;
    private long _frame;
    private float _prevLowDb = -120, _prevMidDb = -120, _prevHighDb = -120;

    private sealed class Partial
    {
        public float Freq;
        public float Db;
        public int Hits;
        public long LastFrame;
    }

    private sealed class Group
    {
        public int Id;
        public float F0;
        public long Born;
        public long LastFrame;
        public int Maturity;
        public float[] Color = Array.Empty<float>();
    }

    private sealed class DrumState
    {
        public int Id;
        public float[] Color = Array.Empty<float>();
        public float Level = -120;
        public bool Onset;
        public bool Active;
        public string Kind = "";
    }

    public DspObjectSource(FrontendWeights w) { _w = w; }

    public void Reset()
    {
        _partials.Clear(); _groups.Clear(); _bank.Clear(); _drums.Clear();
        _nextId = 1; _frame = 0;
    }

    public AnalysisFrame Process(in FrontendFrame f, double timeSec)
    {
        var db = f.BandsDbM;
        Array.Clear(_pitch); Array.Clear(_sal);
        var frame = new AnalysisFrame
        {
            T = timeSec, Source = Name, Objects = new List<ObjectState>(),
            Debug = new DebugInfo { BandsDb = db }
        };

        var peaks = DetectPeaks(f, db);
        var active = MatchPartials(peaks);

        // ---- harmonic groups from stable partials
        var stable = active.Where(p => p.Hits >= StableHits).OrderByDescending(p => p.Db).ToList();
        var assigned = new HashSet<Partial>();
        var memberships = new List<float[]>();
        var ids = new List<int>();
        var colors = new List<float[]>();
        var kinds = new List<string>();
        var transients = new List<bool>();

        foreach (var anchor in stable)
        {
            if (assigned.Contains(anchor)) continue;
            float f0 = EstimateF0(anchor, stable);
            var members = stable.Where(q => !assigned.Contains(q) && HarmonicsMatch(q.Freq, f0)).ToList();
            if (members.Count == 0) continue;
            foreach (var m in members) assigned.Add(m);

            var g = TrackGroup(f0);
            if (g.Maturity < 8) continue;
            var membership = new float[Bands];
            foreach (var m in members)
            {
                int q = BandOf(m.Freq);
                membership[q] = 1f;
                if (q > 0) membership[q - 1] = Math.Max(membership[q - 1], 0.3f);
                if (q < Bands - 1) membership[q + 1] = Math.Max(membership[q + 1], 0.3f);
                _pitch[q] = Math.Max(_pitch[q], 0.6f + 0.4f * Clamp01((m.Db + 60) / 50f));
                _sal[q] = _pitch[q];
            }
            memberships.Add(membership); ids.Add(g.Id); colors.Add(g.Color); kinds.Add("tonal"); transients.Add(false);
        }

        // ---- residual noise
        var occ = new float[Bands];
        foreach (var m in memberships) for (int k = 0; k < Bands; k++) occ[k] = Math.Min(1f, occ[k] + m[k]);
        double rtot = 0, total = 0;
        for (int k = 0; k < Bands; k++) { rtot += Math.Max(0, 1f - occ[k]) * f.PM[k]; total += f.PM[k]; }
        double residDb = 10 * Math.Log10(rtot + 1e-12);
        double totalDb = 10 * Math.Log10(total + 1e-12);

        // ---- drum classification from band energy onsets
        float lowDb = Db(BandSum(f, 40, 220)), midDb = Db(BandSum(f, 220, 2500)), highDb = Db(BandSum(f, 5000, 16000));
        bool lowOnset = lowDb - _prevLowDb > 10;
        bool highOnset = highDb > -45 && highDb - _prevHighDb > 10;
        bool midOnset = midDb > -45 && midDb - _prevMidDb > 8;
        var drumOnset = ClassifyDrums(lowOnset, midOnset, highOnset, lowDb, midDb, highDb);
        bool drumsActive = _drums.Values.Any(d => d.Active);

        int spread = 0;
        for (int k = 0; k < Bands; k++) if (Math.Max(0, 1f - occ[k]) * f.PM[k] > rtot * 0.002) spread++;
        bool noiseLike = rtot > 1e-12 && spread > Bands / 4;
        if (residDb > -55 && residDb > totalDb - 25 && noiseLike && !drumsActive)
        {
            var noise = _groups.FirstOrDefault(g => g.F0 < 0 && g.LastFrame > _frame - 10);
            if (noise == null)
            {
                int nid = _nextId++;
                noise = new Group { Id = nid, F0 = -1, Born = _frame, Color = ObjectParams.PaletteColor(nid) };
                _groups.Add(noise);
            }
            noise.LastFrame = _frame;
            var m = new float[Bands];
            for (int k = 0; k < Bands; k++) m[k] = Math.Max(0, 1f - occ[k]) * 0.9f;
            memberships.Add(m); ids.Add(noise.Id); colors.Add(noise.Color); kinds.Add("noise"); transients.Add(false);
        }

        foreach (var d in _drums.Values)
        {
            if (!d.Active) continue;
            var m = new float[Bands];
            for (int k = 0; k < Bands; k++)
            {
                float fc = _w.Fc[k];
                bool inRange = d.Kind switch
                {
                    "kick" => fc >= 40 && fc <= 260,
                    "snare" => fc >= 200 && fc <= 6500,
                    "hihat" => fc >= 5500 && fc <= 16000,
                    _ => false
                };
                if (inRange) m[k] = 1f;
            }
            memberships.Add(m); ids.Add(d.Id); colors.Add(d.Color); kinds.Add("drum"); transients.Add(d.Onset);
        }

        // ---- build objects
        for (int i = 0; i < memberships.Count; i++)
        {
            bool isDrum = kinds[i] == "drum";
            var st = ObjectParams.Build(f, memberships[i], ids[i], colors[i], null,
                (_frame - Math.Min(_frame, AgeOf(ids[i]))) / 100f, transients[i], _w, _pitch, _sal,
                isDrum ? 1f : -1f);
            if (kinds[i] == "noise")
            {
                st.Tonality = 0.05f;
                st.Pitches.Clear();
                // noise spans whatever the tonal objects did not explain
                float lo = 0, hi = 0;
                for (int k = 0; k < Bands; k++)
                {
                    if (memberships[i][k] < 0.2f) continue;
                    if (lo == 0) lo = _w.Fc[k];
                    hi = _w.Fc[k];
                }
                if (lo > 0) { st.LowHz = lo; st.HighHz = Math.Max(hi, lo); }
            }
            frame.Objects.Add(st);
        }

        // ---- lifecycle
        foreach (var p in _partials.Where(p => _frame - p.LastFrame > 10).ToList()) _partials.Remove(p);
        for (int i = _groups.Count - 1; i >= 0; i--)
        {
            var g = _groups[i];
            if (_frame - g.LastFrame > 120)
            {
                if (g.F0 > 0) { _bank.Add(g); if (_bank.Count > 64) _bank.RemoveAt(0); }
                _groups.RemoveAt(i);
            }
        }
        _prevLowDb = lowDb; _prevMidDb = midDb; _prevHighDb = highDb;
        _frame++;
        return frame;
    }

    // ------------------------------------------------------------ peaks
    private readonly List<(int Band, float Freq, float Db)> _peaks = new(32);

    private List<(int Band, float Freq, float Db)> DetectPeaks(in FrontendFrame f, float[] db)
    {
        _peaks.Clear();
        var sorted = db.OrderBy(v => v).ToArray();
        float floor = sorted[Bands / 5];
        float maxDb = sorted[Bands - 1];
        float thr = Math.Max(floor + 10f, Math.Max(-78f, maxDb - 45f));
        for (int k = 1; k < Bands - 1; k++)
        {
            if (db[k] < thr || db[k] < db[k - 1] || db[k] < db[k + 1]) continue;
            // merged plateau: keep the leftmost maximum only
            if (k > 1 && db[k] == db[k - 1]) continue;
            float y0 = db[k - 1], y1 = db[k], y2 = db[k + 1];
            float den = y0 - 2 * y1 + y2;
            float d = Math.Abs(den) > 1e-6f ? Math.Clamp(0.5f * (y0 - y2) / den, -0.5f, 0.5f) : 0f;
            float a = _w.Fc[k], b = _w.Fc[Math.Min(k + 1, Bands - 1)];
            _peaks.Add((k, a * MathF.Pow(b / a, d), y1));
            if (_peaks.Count >= 32) break;
        }
        return _peaks;
    }

    private List<Partial> MatchPartials(List<(int Band, float Freq, float Db)> peaks)
    {
        var active = new List<Partial>(peaks.Count);
        var used = new HashSet<Partial>();
        foreach (var pk in peaks)
        {
            Partial? best = null; double bd = MatchCents;
            foreach (var p in _partials)
            {
                if (used.Contains(p)) continue;
                double c = Math.Abs(Cents(p.Freq, pk.Freq));
                if (c < bd) { bd = c; best = p; }
            }
            if (best == null)
            {
                best = new Partial { Freq = pk.Freq, Hits = 1 };
                _partials.Add(best);
            }
            else
            {
                best.Freq = 0.65f * best.Freq + 0.35f * pk.Freq;
                best.Hits = Math.Min(best.Hits + 1, 1000);
            }
            best.Db = pk.Db;
            best.LastFrame = _frame;
            used.Add(best);
            active.Add(best);
        }
        return active;
    }

    private float EstimateF0(Partial anchor, List<Partial> stable)
    {
        float bestF0 = anchor.Freq;
        double bestScore = 0;
        for (int n = 1; n <= 8; n++)
        {
            float cand = anchor.Freq / n;
            if (cand < 40f || cand > 8000f) continue;
            bool present = stable.Any(q => Math.Abs(Cents(q.Freq, cand)) <= MatchCents && q.Db >= anchor.Db - 22f);
            if (!present) continue;
            double score = 0;
            foreach (var q in stable)
                if (HarmonicsMatch(q.Freq, cand)) score += Math.Pow(10, q.Db / 10.0);
            score *= 1.0 - 0.012 * n; // prefer the fundamental on ties
            if (score > bestScore * 1.0001)
            {
                bestScore = score;
                bestF0 = cand;
            }
        }
        return bestF0;
    }

    private Group TrackGroup(float f0)
    {
        var g = _groups.Where(x => x.F0 > 0 && _frame - x.LastFrame <= 3)
            .OrderBy(x => Math.Abs(Cents(x.F0, f0))).FirstOrDefault();
        if (g != null && Math.Abs(Cents(g.F0, f0)) < 120)
        {
            g.F0 = 0.8f * g.F0 + 0.2f * f0;
            g.Maturity = g.LastFrame == _frame - 1 ? Math.Min(g.Maturity + 1, 1000) : 1;
            g.LastFrame = _frame;
            return g;
        }
        var bank = _bank.OrderBy(x => Math.Abs(Cents(x.F0, f0))).FirstOrDefault();
        if (bank != null && Math.Abs(Cents(bank.F0, f0)) < 200)
        {
            _bank.Remove(bank);
            bank.F0 = f0; bank.LastFrame = _frame; bank.Born = _frame; bank.Maturity = 1;
            _groups.Add(bank);
            return bank;
        }
        int gid = _nextId++;
        g = new Group { Id = gid, F0 = f0, Born = _frame, LastFrame = _frame, Maturity = 1, Color = ObjectParams.PaletteColor(gid) };
        _groups.Add(g);
        return g;
    }

    private long AgeOf(int id)
    {
        var g = _groups.FirstOrDefault(x => x.Id == id);
        return g == null ? _frame : g.Born;
    }

    // ------------------------------------------------------------ drums
    private Dictionary<string, bool> ClassifyDrums(bool lowOnset, bool midOnset, bool highOnset,
        float lowDb, float midDb, float highDb)
    {
        if (!_drums.ContainsKey("kick"))
        {
            foreach (var kind in new[] { "kick", "snare", "hihat" })
            {
                int id = _nextId++;
                _drums[kind] = new DrumState { Id = id, Color = ObjectParams.PaletteColor(id), Kind = kind };
            }
        }
        var result = new Dictionary<string, bool>();
        foreach (var kind in new[] { "kick", "snare", "hihat" })
        {
            var d = _drums[kind];
            d.Onset = false;
            bool onset = kind switch
            {
                "kick" => lowOnset && lowDb > midDb - 2 && lowDb > highDb,
                "hihat" => highOnset && highDb > midDb + 2 && highDb > lowDb + 4,
                "snare" => midOnset && !lowOnset && !highOnset,
                _ => false
            };
            if (onset)
            {
                d.Level = Math.Max(d.Level, Math.Max(lowDb, Math.Max(midDb, highDb)));
                d.Onset = true;
                d.Active = true;
            }
            else d.Level -= 3.0f;
            if (d.Level < -60) { d.Level = -60; d.Active = false; }
            result[kind] = onset;
        }
        return result;
    }

    // ------------------------------------------------------------ helpers
    private float BandSum(in FrontendFrame f, float lo, float hi)
    {
        double s = 0;
        for (int k = 0; k < Bands; k++) if (_w.Fc[k] >= lo && _w.Fc[k] <= hi) s += f.PM[k];
        return (float)s;
    }

    private static float Db(float x) => (float)(10 * Math.Log10(x + 1e-12));
    private static float Clamp01(float x) => Math.Clamp(x, 0f, 1f);
    private static double Cents(double a, double b) => 1200.0 * Math.Log2(Math.Max(1e-6, a) / Math.Max(1e-6, b));

    private static int BandOf(float hz) =>
        Math.Clamp((int)Math.Round(Math.Log(hz / FMin) / Math.Log(FMax / FMin) * (Bands - 1)), 0, Bands - 1);

    private static bool HarmonicsMatch(float hz, float f0)
    {
        if (f0 <= 0) return false;
        double ratio = hz / f0;
        long n = (long)Math.Round(ratio);
        if (n < 1 || n > 64) return false;
        return Math.Abs(Cents(hz, f0 * n)) <= HarmonicCents;
    }
}

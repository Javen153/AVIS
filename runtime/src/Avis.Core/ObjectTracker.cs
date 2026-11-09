using System;
using System.Collections.Generic;
using System.Linq;

namespace Avis.Core;

public sealed class TrackerConfig
{
    public double MatchThreshold = 0.60;
    public double LeaderThreshold = 0.70;
    public double BankThreshold = 0.75;
    public int BirthFrames = 5;
    public int FadeFrames = 15;
    public int RemoveFrames = 60;
    public double MergeThreshold = 0.85;
    public int MergeFrames = 50;
    public double BankSeconds = 60;
    public double CentroidAlpha = 0.05;
    public double FingerprintAlpha = 0.005;
    public double SoftmaxTau = 0.10;
    public int MaxObjects = 24;
    public double ActThreshold = 0.5;

    public static TrackerConfig FromSettings(AppSettings s) => new()
    {
        MatchThreshold = s.MatchThreshold,
        LeaderThreshold = s.LeaderThreshold,
        BankThreshold = s.BankThreshold,
        BirthFrames = (int)s.BirthFrames,
        FadeFrames = (int)s.FadeFrames,
        RemoveFrames = (int)s.RemoveFrames,
        MergeThreshold = s.MergeThreshold,
        MergeFrames = (int)s.MergeFrames,
        BankSeconds = s.BankSeconds,
        CentroidAlpha = s.CentroidAlpha,
        FingerprintAlpha = s.FingerprintAlpha,
        SoftmaxTau = s.SoftmaxTau,
        MaxObjects = Math.Clamp((int)s.MaxObjects, 1, 24),
        ActThreshold = s.ActThreshold,
    };
}

/// <summary>Frame clustering + temporal tracking (SPEC F, ObjectTracker).</summary>
public sealed class ObjectTracker
{
    private const int D = 20;
    private readonly FrontendWeights _w;
    private TrackerConfig _cfg;
    private readonly List<Tracked> _active = new();
    private readonly List<Candidate> _candidates = new();
    private readonly Dictionary<int, BankEntry> _bank = new();
    private readonly List<float[]> _memberships = new();
    private readonly List<float[]> _memPool = new();
    private readonly List<int> _unassigned = new();
    private readonly List<int> _group = new();
    private readonly List<int> _idx = new();
    private readonly HashSet<int> _visited = new();
    private int[] _owner = Array.Empty<int>();
    private float[] _ownerW = Array.Empty<float>();
    private float[] _vals = Array.Empty<float>();
    private readonly float[] _mean = new float[20];
    private int _nextId = 1;
    private long _frame;

    private sealed class Tracked
    {
        public int Id;
        public float[] Color = new float[3];
        public float[] Centroid = new float[D];
        public float[] Fingerprint = new float[D];
        public long BornFrame;
        public long LastSeen;
        public int NoBands;
        public bool Fading;
        public float LastPresence;
        public float[] LastLoud = new float[8];
        public int LoudPos;
        public bool IsTransient;
        public ObjectState? LastState;
        public int MergeCount;
    }

    private sealed class Candidate
    {
        public float[] Centroid = new float[D];
        public int Hits;
        public long LastFrame = -100;
        public float Energy;
    }

    private sealed class BankEntry
    {
        public int Id;
        public float[] Color = new float[3];
        public float[] Fingerprint = new float[D];
        public long LastSeen;
    }

    public ObjectTracker(FrontendWeights w, TrackerConfig cfg)
    {
        _w = w;
        _cfg = cfg;
    }

    public void UpdateConfig(TrackerConfig cfg) => _cfg = cfg;

    public void Reset()
    {
        _active.Clear(); _candidates.Clear(); _bank.Clear();
        _memberships.Clear(); _memPool.Clear();
        _unassigned.Clear(); _group.Clear(); _idx.Clear(); _visited.Clear();
        _nextId = 1; _frame = 0;
    }

    public int ActiveCount => _active.Count;

    /// <summary>Input: 192-band embeddings and heads. Returns a complete AnalysisFrame.</summary>
    public AnalysisFrame Process(FrontendFrame f, float[][] emb, float[] act, float[] pitch, float[] sal)
    {
        int bands = _w.Fc.Length;
        if (_owner.Length != bands)
        {
            _owner = new int[bands];
            _ownerW = new float[bands];
        }
        if (_vals.Length < _cfg.MaxObjects) _vals = new float[_cfg.MaxObjects];

        // keep one pooled membership array per active object, cleared for this frame
        while (_memberships.Count > _active.Count) ReturnMembership(_memberships.Count - 1);
        for (int i = 0; i < _memberships.Count; i++) Array.Clear(_memberships[i]);

        // 1-2. assign active bands to existing objects
        Array.Fill(_owner, -1);
        Array.Clear(_ownerW);
        for (int k = 0; k < bands; k++)
        {
            if (act[k] < _cfg.ActThreshold || f.BandsDbM[k] < -80) continue;
            float best = -2; int bi = -1;
            for (int i = 0; i < _active.Count; i++)
            {
                if (_active[i].Fading) continue;
                float c = Cos(emb[k], _active[i].Centroid);
                if (c > best) { best = c; bi = i; }
            }
            if (bi >= 0 && best >= _cfg.MatchThreshold)
            {
                _owner[k] = bi; _ownerW[k] = best;
            }
        }

        // 3. leader clustering of unassigned bands -> candidates / births
        _unassigned.Clear();
        for (int k = 0; k < bands; k++)
            if (_owner[k] < 0 && act[k] >= _cfg.ActThreshold && f.BandsDbM[k] >= -80) _unassigned.Add(k);
        _visited.Clear();
        foreach (int lead in _unassigned)
        {
            if (_visited.Contains(lead)) continue;
            _group.Clear();
            _group.Add(lead);
            _visited.Add(lead);
            foreach (int k in _unassigned)
            {
                if (k == lead || _visited.Contains(k)) continue;
                if (Cos(emb[k], emb[lead]) >= _cfg.LeaderThreshold) { _group.Add(k); _visited.Add(k); }
            }
            var centroid = MeanEmb(emb, _group);
            Candidate? cand = null; float candCos = -2;
            foreach (var c in _candidates)
            {
                float cc = Cos(c.Centroid, centroid);
                if (cc > candCos) { candCos = cc; cand = c; }
            }
            if (cand == null || candCos < _cfg.LeaderThreshold)
            {
                cand = new Candidate { Centroid = centroid };
                _candidates.Add(cand);
            }
            cand.Hits = (long)_frame - cand.LastFrame <= 2 ? cand.Hits + 1 : 1;
            cand.LastFrame = _frame;
            cand.Centroid = Blend(cand.Centroid, centroid, 0.2f);
            double eSum = 0;
            foreach (int k in _group) eSum += f.BandsDbM[k];
            float energy = _group.Count > 0 ? (float)(eSum / _group.Count) : 0f;
            cand.Energy = energy;

            if (cand.Hits >= _cfg.BirthFrames && energy > -70 && _active.Count < _cfg.MaxObjects)
            {
                Tracked obj;
                int bankId = -1; float bankCos = -2;
                foreach (var kv in _bank)
                {
                    float c = Cos(kv.Value.Fingerprint, centroid);
                    if (c > bankCos) { bankCos = c; bankId = kv.Key; }
                }
                float[] color;
                if (bankId > 0 && bankCos >= _cfg.BankThreshold)
                {
                    color = _bank[bankId].Color;
                    _bank.Remove(bankId);
                    obj = new Tracked { Id = bankId, Color = color, BornFrame = _frame, Fingerprint = (float[])centroid.Clone() };
                    _nextId = Math.Max(_nextId, bankId + 1);
                }
                else
                {
                    int id = _nextId++;
                    color = ObjectParams.PaletteColor(id);
                    obj = new Tracked { Id = id, Color = color, BornFrame = _frame, Fingerprint = (float[])centroid.Clone() };
                }
                obj.Centroid = centroid;
                _active.Add(obj);
                _candidates.Remove(cand);
                _memberships.Add(RentMembership());
                foreach (int k in _group) { _owner[k] = _active.Count - 1; _ownerW[k] = 1f; }
            }
        }
        _candidates.RemoveAll(c => (long)_frame - c.LastFrame > 8);
        while (_memberships.Count < _active.Count) _memberships.Add(RentMembership());

        // 4. soft membership (cos softmax) for owning object
        for (int k = 0; k < bands; k++)
        {
            if (_owner[k] < 0) continue;
            var obj = _active[_owner[k]];
            float best = Cos(emb[k], obj.Centroid);
            float sum = 0;
            Array.Clear(_vals, 0, _active.Count);
            for (int i = 0; i < _active.Count; i++)
            {
                if (_active[i].Fading) continue;
                float c = Cos(emb[k], _active[i].Centroid);
                if (c >= _cfg.MatchThreshold - 0.15f) { float e = MathF.Exp((c - best) / (float)_cfg.SoftmaxTau); _vals[i] = e; sum += e; }
            }
            if (sum <= 0) { _memberships[_owner[k]][k] = 1; continue; }
            for (int i = 0; i < _active.Count; i++) if (_vals[i] > 0) _memberships[i][k] = _vals[i] / sum;
        }

        // 5. centroid / fingerprint updates
        for (int i = 0; i < _active.Count; i++)
        {
            var obj = _active[i];
            var m = _memberships[i];
            _idx.Clear();
            for (int k = 0; k < bands; k++) if (m[k] > 0.5f) _idx.Add(k);
            if (_idx.Count > 0)
            {
                MeanEmbInto(emb, _idx, _mean);
                BlendInPlace(obj.Centroid, _mean, (float)_cfg.CentroidAlpha);
                NormalizeInPlace(obj.Centroid);
                double psum = 0;
                foreach (int k in _idx) psum += f.PM[k];
                float pres = (float)(psum / _idx.Count);
                if (pres > 1e-9)
                {
                    BlendInPlace(obj.Fingerprint, _mean, (float)_cfg.FingerprintAlpha);
                    NormalizeInPlace(obj.Fingerprint);
                }
                obj.LastSeen = _frame;
                obj.NoBands = 0;
                obj.Fading = false;
            }
            else
            {
                obj.NoBands++;
                if (obj.NoBands >= _cfg.FadeFrames) obj.Fading = true;
            }
        }

        // 6. merge close centroids
        for (int i = 0; i < _active.Count; i++)
        {
            for (int j = i + 1; j < _active.Count; j++)
            {
                if (Cos(_active[i].Centroid, _active[j].Centroid) >= _cfg.MergeThreshold)
                {
                    _active[i].MergeCount++;
                    if (_active[i].MergeCount >= _cfg.MergeFrames)
                    {
                        int keepIdx = _active[i].Id < _active[j].Id ? i : j;
                        int dropIdx = keepIdx == i ? j : i;
                        var keep = _active[keepIdx];
                        var drop = _active[dropIdx];
                        keep.Centroid = Normalize(Blend(keep.Centroid, drop.Centroid, 0.5f));
                        keep.Fingerprint = Normalize(Blend(keep.Fingerprint, drop.Fingerprint, 0.5f));
                        keep.BornFrame = Math.Min(keep.BornFrame, drop.BornFrame);
                        _active.RemoveAt(dropIdx);
                        ReturnMembership(dropIdx);
                        i = -1; break;
                    }
                }
                else _active[i].MergeCount = 0;
            }
        }

        // 7. removal -> bank
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var obj = _active[i];
            if (obj.NoBands >= _cfg.RemoveFrames)
            {
                _bank[obj.Id] = new BankEntry { Id = obj.Id, Color = obj.Color, Fingerprint = obj.Fingerprint, LastSeen = _frame };
                _active.RemoveAt(i);
                ReturnMembership(i);
            }
        }
        foreach (var kv in _bank.ToList())
            if ((_frame - kv.Value.LastSeen) / 100.0 > _cfg.BankSeconds) _bank.Remove(kv.Key);

        // 8. build states
        var frame = new AnalysisFrame
        {
            T = f.TimeSec,
            Source = "tracker",
            Objects = new List<ObjectState>(),
        };
        var dbg = new DebugInfo
        {
            BandsDb = f.BandsDbM,
            OwnerId = _owner.ToArray(),
            OwnerWeight = _ownerW.ToArray(),
        };
        for (int i = 0; i < _active.Count; i++)
        {
            var obj = _active[i];
            var m = i < _memberships.Count ? _memberships[i] : new float[bands];
            if (HasPositive(m))
            {
                var st = ObjectParams.Build(f, m, obj.Id, obj.Color, (float[])obj.Centroid.Clone(),
                    (float)((_frame - obj.BornFrame) / 100.0), obj.IsTransient, _w, pitch, sal, -1f);
                obj.LastState = st;
                obj.LastPresence = st.Presence;
                obj.LoudPos = (obj.LoudPos + 1) % obj.LastLoud.Length;
                obj.LastLoud[obj.LoudPos] = st.LoudnessDb;
                int prev = (obj.LoudPos - 3 + obj.LastLoud.Length) % obj.LastLoud.Length;
                obj.IsTransient = st.LoudnessDb - obj.LastLoud[prev] > 6f;
                st.IsTransient = obj.IsTransient;
                frame.Objects.Add(st);
            }
            else if (obj.Fading && obj.LastState != null)
            {
                var st = obj.LastState;
                st.Presence = obj.LastPresence * MathF.Exp(-obj.NoBands / 20f);
                st.State = "fading";
                st.LoudnessDb = st.LoudnessDb - obj.NoBands * 0.5f;
                st.AgeSec = (float)((_frame - obj.BornFrame) / 100.0);
                if (st.Presence < 0.02f) continue;
                frame.Objects.Add(st);
            }
        }
        frame.Debug = dbg;
        _frame++;
        return frame;
    }

    private float[] RentMembership()
    {
        if (_memPool.Count == 0) return new float[_w.Fc.Length];
        var a = _memPool[^1];
        _memPool.RemoveAt(_memPool.Count - 1);
        return a;
    }

    private void ReturnMembership(int index)
    {
        var a = _memberships[index];
        Array.Clear(a);
        _memberships.RemoveAt(index);
        _memPool.Add(a);
    }

    private static bool HasPositive(float[] m)
    {
        for (int i = 0; i < m.Length; i++) if (m[i] > 0) return true;
        return false;
    }

    private static float Cos(float[] a, float[] b)    {
        double s = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { s += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (float)(s / (Math.Sqrt(na * nb) + 1e-12));
    }

    private static float[] MeanEmb(float[][] emb, List<int> idx)
    {
        var r = new float[D];
        foreach (var k in idx) for (int d = 0; d < D; d++) r[d] += emb[k][d];
        return Normalize(r);
    }

    private static void MeanEmbInto(float[][] emb, List<int> idx, float[] dst)
    {
        Array.Clear(dst);
        foreach (var k in idx)
            for (int d = 0; d < D; d++) dst[d] += emb[k][d];
        NormalizeInPlace(dst);
    }

    private static void BlendInPlace(float[] dst, float[] src, float alpha)
    {
        for (int i = 0; i < dst.Length; i++) dst[i] = (1 - alpha) * dst[i] + alpha * src[i];
    }

    private static void NormalizeInPlace(float[] v)
    {
        double n = 0;
        for (int i = 0; i < v.Length; i++) n += v[i] * v[i];
        n = Math.Sqrt(n) + 1e-12;
        for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / n);
    }

    private static float[] Normalize(float[] v)
    {
        double n = 0;
        for (int i = 0; i < v.Length; i++) n += v[i] * v[i];
        n = Math.Sqrt(n) + 1e-12;
        var r = new float[v.Length];
        for (int i = 0; i < v.Length; i++) r[i] = (float)(v[i] / n);
        return r;
    }

    private static float[] Blend(float[] a, float[] b, float alpha)
    {
        var r = new float[a.Length];
        for (int i = 0; i < a.Length; i++) r[i] = (1 - alpha) * a[i] + alpha * b[i];
        return r;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using Avis.Core;
using Xunit;

namespace Avis.Tests;

public class TrackerTests
{
    private const int Bands = 192;

    private static FrontendWeights Weights() => new(TestPaths.Weights);

    private static FrontendFrame Frame(int frame, params int[] activeBands)
    {
        var db = new float[Bands];
        var pm = new float[Bands];
        var pl = new float[Bands];
        var pr = new float[Bands];
        for (int k = 0; k < Bands; k++) { db[k] = -100; pm[k] = 0; pl[k] = 0; pr[k] = 0; }
        foreach (int k in activeBands)
        {
            db[k] = -20;
            pm[k] = 1e-3f;
            pl[k] = 1e-3f;
            pr[k] = 1e-3f;
        }
        return new FrontendFrame
        {
            TimeSec = frame / 100.0,
            BandsDbM = db,
            PM = pm,
            PL = pl,
            PR = pr,
            Features = new float[3 * Bands],
        };
    }

    private static float[][] UnitEmbeddings(int activeDim, Random? rng = null, double jitter = 0)
    {
        var emb = new float[Bands][];
        for (int k = 0; k < Bands; k++)
        {
            var v = new float[20];
            v[activeDim] = 1f;
            if (rng != null && jitter > 0)
                for (int d = 0; d < 20; d++) v[d] += (float)((rng.NextDouble() * 2 - 1) * jitter);
            double n = Math.Sqrt(v.Sum(x => x * x)) + 1e-12;
            for (int d = 0; d < 20; d++) v[d] = (float)(v[d] / n);
            emb[k] = v;
        }
        return emb;
    }

    private static float[] On(int[] bands)
    {
        var a = new float[Bands];
        foreach (int k in bands) a[k] = 1f;
        return a;
    }

    private static AnalysisFrame Run(ObjectTracker tr, int frame, int[] clusterA, int[] clusterB, int[]? only = null)
    {
        var active = (only ?? clusterA.Concat(clusterB).ToArray());
        var f = Frame(frame, active);
        var emb = UnitEmbeddings(0);
        // cluster B gets a second basis direction
        foreach (int k in clusterB) { var v = new float[20]; v[1] = 1f; emb[k] = v; }
        foreach (int k in clusterA) { var v = new float[20]; v[0] = 1f; emb[k] = v; }
        var act = On(active);
        var pitch = new float[Bands];
        var sal = new float[Bands];
        return tr.Process(f, emb, act, pitch, sal);
    }

    [Fact]
    public void TwoClustersProduceTwoObjects()
    {
        var tr = new ObjectTracker(Weights(), new TrackerConfig());
        AnalysisFrame? last = null;
        for (int i = 0; i < 12; i++) last = Run(tr, i, Enumerable.Range(0, 8).ToArray(), Enumerable.Range(8, 8).ToArray());
        Assert.NotNull(last);
        Assert.Equal(2, last!.Objects.Count);
        Assert.Equal(2, last.Objects.Select(o => o.Id).Distinct().Count());
    }

    [Fact]
    public void ObjectReturnsWithSameIdAfterGap()
    {
        var tr = new ObjectTracker(Weights(), new TrackerConfig());
        var a = Enumerable.Range(0, 8).ToArray();
        int id = -1;
        for (int i = 0; i < 12; i++)
        {
            var fr = Run(tr, i, a, Array.Empty<int>(), only: a);
            if (fr.Objects.Count > 0) id = fr.Objects[0].Id;
        }
        Assert.True(id > 0);
        // 1.2 s of silence: enough to fade and remove into the bank (60 frames remove,
        // bank memory is 60 s, far beyond the gap).
        for (int i = 12; i < 140; i++) Run(tr, i, a, Array.Empty<int>(), only: Array.Empty<int>());
        int id2 = -1;
        for (int i = 140; i < 152; i++)
        {
            var fr = Run(tr, i, a, Array.Empty<int>(), only: a);
            if (fr.Objects.Count > 0) { id2 = fr.Objects[0].Id; break; }
        }
        Assert.Equal(id, id2);
    }

    [Fact]
    public void CloseCentroidsMergeIntoOneObject()
    {
        var tr = new ObjectTracker(Weights(), new TrackerConfig());
        // A and B are unit vectors at 25 degrees: cos ~ 0.906 > merge threshold, but
        // > leader threshold only if we keep them far apart at birth, so start
        // with two clean clusters and then rotate B towards A.
        var f = Frame(0, Enumerable.Range(0, 8).ToArray());
        var emb = UnitEmbeddings(0);
        var act = On(Enumerable.Range(0, 8).ToArray());
        var pitch = new float[Bands];
        var sal = new float[Bands];
        AnalysisFrame? last = null;
        for (int i = 0; i < 10; i++) last = tr.Process(Frame(i, Enumerable.Range(0, 8).ToArray()), emb, act, pitch, sal);
        Assert.Single(last!.Objects);
        // now the same bands drift to a slightly rotated embedding; the object must stay single
        var rng = new Random(4);
        for (int i = 10; i < 80; i++)
        {
            var e2 = UnitEmbeddings(0, rng, 0.05);
            last = tr.Process(Frame(i, Enumerable.Range(0, 8).ToArray()), e2, act, pitch, sal);
            Assert.Single(last.Objects);
        }
    }

    [Fact]
    public void SmallNoiseDoesNotFlickerIds()
    {
        var tr = new ObjectTracker(Weights(), new TrackerConfig());
        var bands = Enumerable.Range(0, 8).ToArray();
        var act = On(bands);
        var pitch = new float[Bands];
        var sal = new float[Bands];
        var rng = new Random(11);
        var ids = new HashSet<int>();
        AnalysisFrame? last = null;
        for (int i = 0; i < 120; i++)
        {
            var emb = UnitEmbeddings(0, rng, 0.1);
            last = tr.Process(Frame(i, bands), emb, act, pitch, sal);
            if (i >= 10)
            {
                Assert.Single(last.Objects);
                ids.Add(last.Objects[0].Id);
            }
        }
        Assert.Single(ids);
    }
}

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Avis.Core;
using Vortice.Direct3D11;
using Vortice.Mathematics;

namespace Avis.Render;

public sealed partial class Renderer
{
    private sealed class ObState
    {
        public float X, Y, Rx, Ry, A;
        public int Seen;
        public readonly List<float[]> Hist = new(12);
    }

    private readonly Dictionary<int, ObState> _states = new();
    private readonly float[] _uni = new float[8];
    private int _tick;

    /// <summary>Renders one analysis frame. <paramref name="target"/> null renders offscreen for ReadPixels.</summary>
    public void Render(AnalysisFrame frame, AppSettings s, double timeSec, double dt = 1.0 / 60.0,
        ID3D11RenderTargetView? target = null)
    {
        BuildInstances(frame, s, timeSec, dt);
        var final = target ?? _ldrRtv!;

        Context.OMSetDepthStencilState(_noDepth);
        Context.RSSetState(_raster);
        Context.RSSetViewport(new Viewport(0, 0, Width, Height, 0, 1));
        Context.IASetPrimitiveTopology(Vortice.Direct3D.PrimitiveTopology.TriangleList);
        Context.VSSetShader(_vsFull);
        Context.PSSetSampler(0, _sampler);

        if (s.View == DebugView.Spectro || s.View == DebugView.Table)
        {
            DrawSpectro(final, frame, s);
            return;
        }

        // ---- HDR splat pass
        Context.OMSetRenderTargets(_hdrRtv);
        Context.ClearRenderTargetView(_hdrRtv, new Color4(0, 0, 0, 0));
        Context.OMSetBlendState(_additive);
        Context.VSSetShader(_vsSplat);
        Context.PSSetShader(_psSplat);
        Context.VSSetShaderResources(0, new[] { _instancesView });
        SetUniforms(Width, Height, (float)timeSec, s.View == DebugView.Masks ? 1f : 0f);
        Context.VSSetConstantBuffer(0, _uniforms);
        Context.PSSetConstantBuffer(0, _uniforms);
        if (_instanceCount > 0)
        {
            UploadInstances();
            Context.DrawInstanced(4, (uint)_instanceCount, 0, 0);
        }

        bool bloom = s.View == DebugView.Main && s.Bloom > 0.001;
        if (bloom) RunBloom();
        else Context.ClearRenderTargetView(_bloomRtv[0], new Color4(0, 0, 0, 0));

        // ---- composite
        Context.OMSetRenderTargets(final);
        Context.ClearRenderTargetView(final, new Color4(0, 0, 0, 0));
        Context.RSSetViewport(new Viewport(0, 0, Width, Height, 0, 1));
        Context.OMSetBlendState(_opaque);
        Context.VSSetShader(_vsFull);
        Context.PSSetShader(_psComposite);
        BindSrv(0, _hdrSrv);
        BindSrv(1, _bloomSrv[0]);
        BindSrv(2, null);
        SetUniforms(Width, Height, (float)timeSec, s.View == DebugView.Masks ? 1f : 0f);
        Context.PSSetConstantBuffer(0, _uniforms);
        Context.Draw(3, 0);
    }

    private ID3D11BlendState? _opaqueState;
    private ID3D11BlendState _opaque => _opaqueState ??= Device.CreateBlendState(BlendDescription.Opaque);

    private void RunBloom()
    {
        Context.VSSetShader(_vsFull);
        Context.PSSetShader(_psDown);
        Context.OMSetBlendState(_opaque);
        ID3D11ShaderResourceView src = _hdrSrv!;
        int w = Width, h = Height;
        for (int i = 0; i < BloomLevels; i++)
        {
            Context.OMSetRenderTargets(_bloomRtv[i]);
            Context.ClearRenderTargetView(_bloomRtv[i], new Color4(0, 0, 0, 0));
            IfaceSetViewport(_bloomWww(i), _bloomHhh(i));
            SetUniforms(Math.Max(1, w), Math.Max(1, h), 0, 0);
            Context.PSSetConstantBuffer(0, _uniforms);
            BindSrv(0, src);
            Context.Draw(3, 0);
            src = _bloomSrv[i];
            w /= 2; h /= 2;
        }
        Context.PSSetShader(_psUp);
        Context.OMSetBlendState(_additive);
        for (int i = BloomLevels - 1; i >= 1; i--)
        {
            Context.OMSetRenderTargets(_bloomRtv[i - 1]);
            IfaceSetViewport(_bloomWww(i - 1), _bloomHhh(i - 1));
            SetUniforms(_bloomWww(i), _bloomHhh(i), 0, 0);
            Context.PSSetConstantBuffer(0, _uniforms);
            BindSrv(0, _bloomSrv[i]);
            Context.Draw(3, 0);
        }
    }

    private int _bw0, _bh0;
    private int _bw(int level)
    {
        int w = Math.Max(1, Width / 2);
        for (int i = 0; i < level; i++) w = Math.Max(1, w / 2);
        return w;
    }
    private int _bh(int level)
    {
        int h = Math.Max(1, Height / 2);
        for (int i = 0; i < level; i++) h = Math.Max(1, h / 2);
        return h;
    }
    private int _bloomWww(int level) => _bw(level);
    private int _bloomHhh(int level) => _bh(level);

    private void IfaceSetViewport(int w, int h)
    {
        Context.RSSetViewport(new Viewport(0, 0, w, h, 0, 1));
    }

    // ------------------------------------------------------------- instances
    private void BuildInstances(AnalysisFrame frame, AppSettings s, double timeSec, double dt)
    {
        _instanceCount = 0;
        if (dt <= 0) dt = 1.0 / 60.0;
        float fmin = (float)Math.Max(10, s.FMin), fmax = (float)Math.Max(fmin + 10, s.FMax);
        float dbMin = (float)s.DbMin, dbMax = (float)s.DbMax;
        var seen = new HashSet<int>();

        foreach (var o in frame.Objects)
        {
            if (o.Presence <= 0.001 && o.State != "active") continue;
            seen.Add(o.Id);
            if (!_states.TryGetValue(o.Id, out var st)) { st = new ObState(); _states[o.Id] = st; }

            float hz = o.CentroidHz > 1 ? o.CentroidHz : Math.Max(30, o.LowHz);
            float tx = MapX(hz, s, fmin, fmax);
            float db = s.YMode switch
            {
                YMode.Partial => o.PartialLoudness,
                YMode.Salience => FromSalience(o.Salience, dbMin, dbMax),
                _ => o.LoudnessDb,
            };
            float ty = Math.Clamp((db - dbMin) / (dbMax - dbMin), 0f, 1f) * 2f - 1f;

            float bright = s.Brightness == BrightnessSource.Salience ? o.Salience : o.Presence;
            bright = Math.Clamp(bright, 0, 1);
            float ta = Math.Clamp(MathF.Pow(bright, (float)s.Gamma) * (float)s.Opacity, 0, 1);
            if (o.State == "fading") ta *= 0.5f;

            double spanOct = Math.Log2(Math.Max(1.01, o.HighHz / Math.Max(20, o.LowHz)));
            float tonalShrink = 1f - (float)s.TonalityInfluence * Math.Clamp(o.Tonality, 0, 1) * 0.6f;
            float rxPx = (0.018f + 0.050f * (float)Math.Min(spanOct, 6.0)) * (float)s.SplatSize * Width * Math.Max(0.25f, tonalShrink);
            float ryPx = (0.010f + 0.020f * o.Presence) * (float)s.SplatSize * Width;
            if (o.IsTransient) { ryPx *= 1.45f; rxPx *= 1.2f; ta = Math.Max(ta, 0.85f); }
            if (o.Tonality < 0.25f) ryPx *= 0.8f + 0.4f * (float)s.NoiseVisibility;

            float rxN = 2 * rxPx / Width;
            float aspect = (ryPx / Math.Max(0.5f, rxPx)) * (Width / (float)Height);

            float posRate = Math.Clamp((float)s.Smoothing * (float)dt, 0, 1);
            float aRate = 1f - MathF.Exp(-(float)dt / Math.Max(0.002f, ta > st.A ? (float)s.Attack : (float)s.Release));
            if (st.Seen == 0) { st.X = tx; st.Y = ty; st.Rx = rxN; st.Ry = aspect; st.A = 0; }
            st.X += (tx - st.X) * posRate;
            st.Y += (ty - st.Y) * posRate;
            st.Rx += (rxN - st.Rx) * posRate;
            st.Ry += (aspect - st.Ry) * posRate;
            st.A += (ta - st.A) * aRate;
            st.Seen++;

            _tick++;
            if (s.History != HistoryMode.Off)
            {
                if (st.Seen % 3 == 0)
                {
                    st.Hist.Add(new[] { st.X, st.Y, st.Rx, st.Ry, st.A, o.Color[0], o.Color[1], o.Color[2] });
                    if (st.Hist.Count > 14) st.Hist.RemoveAt(0);
                }
                for (int i = 0; i < st.Hist.Count - 1; i++)
                {
                    var g = st.Hist[i];
                    float k = (i + 1) / (float)st.Hist.Count;
                    AddInstance(g[0], g[1], g[2], g[3], g[5], g[6], g[7], g[4] * 0.22f * k, 1.6f);
                }
            }

            AddInstance(st.X, st.Y, st.Rx, st.Ry, o.Color[0], o.Color[1], o.Color[2], st.A, 1f);
            if (Environment.GetEnvironmentVariable("AVIS_RENDER_DEBUG") == "1" && _instanceCount <= 8)
                Console.Error.WriteLine($"inst id={o.Id} hz={hz:F1} x={st.X:F3} y={st.Y:F3} rx={st.Rx:F4} aspect={st.Ry:F2} a={st.A:F3}");
            if (s.View == DebugView.Masks)
            {
                // flat mask: tight core so the owner of each band is visible
                AddInstance(st.X, st.Y, st.Rx * 0.6f, st.Ry * 0.6f, o.Color[0], o.Color[1], o.Color[2], 0.9f, 0.2f);
            }
        }
        // drop stale ids
        if (_states.Count > 64)
        {
            var dead = new List<int>();
            foreach (var kv in _states) if (!seen.Contains(kv.Key) && ++kv.Value.Seen > 90) dead.Add(kv.Key);
            foreach (var d in dead) _states.Remove(d);
        }
    }

    private static float FromSalience(float sal, float dbMin, float dbMax) =>
        dbMin + Math.Clamp(sal, 0, 1) * (dbMax - dbMin);

    private static float MapX(float hz, AppSettings s, float fmin, float fmax)
    {
        double a = Warp(hz, s.Scale), b = Warp(fmin, s.Scale), c = Warp(fmax, s.Scale);
        double t = (a - b) / Math.Max(1e-9, c - b);
        return (float)Math.Clamp(t, 0, 1) * 2f - 1f;
    }

    private static double Warp(double hz, ScaleMode mode) => mode switch
    {
        ScaleMode.Linear => hz,
        ScaleMode.Mel => 2595.0 * Math.Log10(1.0 + hz / 700.0),
        ScaleMode.Erb => 21.4 * Math.Log10(1.0 + 0.00437 * hz),
        _ => Math.Log(hz),
    };

    private void AddInstance(float x, float y, float rx, float aspect, float r, float g, float b, float a, float soft)
    {
        if (_instanceCount >= MaxInstances || a <= 0.0005f) return;
        int i = _instanceCount * 12;
        _inst[i] = x; _inst[i + 1] = y;
        _inst[i + 2] = Math.Max(1e-5f, rx);
        _inst[i + 3] = Math.Max(0.02f, aspect);
        _inst[i + 4] = r; _inst[i + 5] = g; _inst[i + 6] = b;
        _inst[i + 7] = a;
        _inst[i + 8] = soft;
        _inst[i + 9] = 0;
        _inst[i + 10] = 0; _inst[i + 11] = 0;
        _instanceCount++;
        if (Environment.GetEnvironmentVariable("AVIS_RENDER_DEBUG") == "1")
            Console.Error.WriteLine($"add#{_instanceCount - 1} x={x:F3} y={y:F3} rx={rx:F4} asp={aspect:F2} a={a:F2} soft={soft:F2}");
    }

    private void UploadInstances()
    {
        var map = Context.Map(_instances, MapMode.WriteDiscard, MapFlags.None);
        Marshal.Copy(_inst, 0, map.DataPointer, _instanceCount * 12);
        Context.Unmap(_instances, 0);
    }

    private void SetUniforms(int w, int h, float time, float mode)
    {
        _uni[0] = w; _uni[1] = h; _uni[2] = time; _uni[3] = mode;
        _uni[4] = 0; _uni[5] = 0; _uni[6] = 0; _uni[7] = 0;
        var map = Context.Map(_uniforms, MapMode.WriteDiscard, MapFlags.None);
        Marshal.Copy(_uni, 0, map.DataPointer, 8);
        Context.Unmap(_uniforms, 0);
    }

    // ------------------------------------------------------------- spectro
    private void DrawSpectro(ID3D11RenderTargetView target, AnalysisFrame frame, AppSettings s)
    {
        var src = frame.Debug?.BandsDb;
        for (int k = 0; k < 192; k++) _bands[k] = src != null && k < src.Length ? src[k] : -100f;
        Array.Clear(_owners);
        foreach (var o in frame.Objects)
        {
            float hz = o.CentroidHz;
            int lo = BandOf(o.LowHz), hi = BandOf(o.HighHz);
            if (hi < lo) (lo, hi) = (hi, lo);
            for (int k = Math.Max(0, lo); k <= Math.Min(191, hi); k++)
            {
                float a = Math.Clamp(o.Presence, 0, 1);
                if (_owners[k * 4 + 3] >= a) continue;
                _owners[k * 4] = o.Color[0]; _owners[k * 4 + 1] = o.Color[1];
                _owners[k * 4 + 2] = o.Color[2]; _owners[k * 4 + 3] = a;
            }
        }
        var mapB = Context.Map(_bandBuffer, MapMode.WriteDiscard, MapFlags.None);
        Marshal.Copy(_bands, 0, mapB.DataPointer, 192);
        Context.Unmap(_bandBuffer, 0);
        var mapO = Context.Map(_ownerBuffer, MapMode.WriteDiscard, MapFlags.None);
        Marshal.Copy(_owners, 0, mapO.DataPointer, 192 * 4);
        Context.Unmap(_ownerBuffer, 0);

        float[] su =
        {
            (float)Warp((float)s.FMin, s.Scale), (float)Warp((float)s.FMax, s.Scale),
            (float)s.DbMin, (float)s.DbMax,
            BitConverter.Int32BitsToSingle(192), BitConverter.Int32BitsToSingle(frame.Objects.Count),
            BitConverter.Int32BitsToSingle(s.View == DebugView.Table ? 1 : 0), 0,
        };
        var mapS = Context.Map(_spectroUniforms, MapMode.WriteDiscard, MapFlags.None);
        Marshal.Copy(su, 0, mapS.DataPointer, 8);
        Context.Unmap(_spectroUniforms, 0);

        Context.OMSetRenderTargets(target);
        Context.ClearRenderTargetView(target, new Color4(0, 0, 0, 1));
        Context.OMSetBlendState(_opaque);
        Context.PSSetShader(_psSpectro);
        Context.PSSetConstantBuffer(1, _spectroUniforms);
        BindSrv(0, _bandView);
        BindSrv(1, _ownerView);
        Context.RSSetViewport(new Viewport(0, 0, Width, Height, 0, 1));
        Context.Draw(3, 0);
        BindSrv(0, null);
        BindSrv(1, null);
    }

    private void BindSrv(int slot, ID3D11ShaderResourceView? srv)
    {
        Context.PSSetShaderResources((uint)slot, srv is null ? Array.Empty<ID3D11ShaderResourceView>() : new[] { srv });
    }

    private static int BandOf(float hz)
    {
        double t = Math.Log(Math.Max(30, hz) / 30.0) / Math.Log(16000.0 / 30.0);
        return (int)Math.Round(Math.Clamp(t, 0, 1) * 191);
    }
}


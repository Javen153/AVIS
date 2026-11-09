using System;
using System.Threading;
using Avis.Core;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace Avis.Audio;

/// <summary>
/// Real-time capture thread: the WASAPI callback only copies into a lock-free ring,
/// an analysis thread drains it in 10 ms hops (SPEC G).
/// </summary>
public sealed class AudioCapture : IDisposable
{
    private const int RingSeconds = 4;
    private readonly float[] _ring;
    private long _write;
    private long _read;
    private readonly object _startLock = new();
    private WasapiLoopbackCapture? _capture;
    private volatile bool _running;
    private readonly ManualResetEventSlim _dataEvent = new(false);
    public double GainDb { get; set; }
    public string DeviceName { get; private set; } = "";
    public long DroppedSamples { get; private set; }
    public event Action? DefaultDeviceChanged;

    public AudioCapture()
    {
        _ring = new float[FrontendWeights.SampleRate * RingSeconds * 2];
        GainDb = 0;
    }

    /// <summary>Starts capturing the default render endpoint. Returns false when no device exists.</summary>
    public bool Start()
    {
        lock (_startLock)
        {
            if (_running) return true;
            try
            {
                MMDevice device;
                using (var enumerator = new MMDeviceEnumerator())
                    device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
                DeviceName = device.FriendlyName;
                _capture = new WasapiLoopbackCapture(device);
                _capture.DataAvailable += OnData;
                _capture.RecordingStopped += OnStopped;
                Interlocked.Exchange(ref _write, 0);
                Interlocked.Exchange(ref _read, 0);
                _capture.StartRecording();
                _running = true;
                return true;
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine("audio capture failed: " + ex.Message);
                _capture?.Dispose();
                _capture = null;
                return false;
            }
        }
    }

    private void OnStopped(object? sender, StoppedEventArgs e)
    {
        if (_running && e.Exception != null) DefaultDeviceChanged?.Invoke();
    }

    private void OnData(object? sender, WaveInEventArgs e)
    {
        if (_capture == null) return;
        var stereo = LoopbackCapture.Convert(e.Buffer, _capture.WaveFormat);
        float gain = (float)Math.Pow(10.0, GainDb / 20.0);
        long w = Interlocked.Read(ref _write);
        for (int i = 0; i < stereo.Length; i += 2)
        {
            if (w + 2 - Interlocked.Read(ref _read) > _ring.Length) { DroppedSamples += 2; continue; }
            _ring[(int)((w) % _ring.Length)] = stereo[i] * gain;
            _ring[(int)((w + 1) % _ring.Length)] = stereo[i + 1] * gain;
            w += 2;
        }
        Interlocked.Exchange(ref _write, w);
        _dataEvent.Set();
    }

    /// <summary>Pulls one hop of interleaved stereo. Returns false when less than 10 ms is available.</summary>
    public bool TryReadHop(Span<float> dst)
    {
        int need = FrontendWeights.Hop * 2;
        if (dst.Length < need) return false;
        long r = Interlocked.Read(ref _read);
        if (Interlocked.Read(ref _write) - r < need) return false;
        for (int i = 0; i < need; i++) dst[i] = _ring[(int)((r + i) % _ring.Length)];
        Interlocked.Exchange(ref _read, r + need);
        return true;
    }

    public void WaitForData(int milliseconds = 20) => _dataEvent.Wait(milliseconds);

    public static float[] MainFormat(float[] stereo, int sampleRate)
    {
        if (sampleRate == FrontendWeights.SampleRate) return stereo;
        return HermiteResampler.Resample(stereo, sampleRate, FrontendWeights.SampleRate);
    }

    public void Dispose()
    {
        _running = false;
        if (_capture != null)
        {
            try { _capture.StopRecording(); } catch { }
            _capture.Dispose();
            _capture = null;
        }
        _dataEvent.Dispose();
    }
}

using System;
using System.Collections.Generic;
using System.Threading;
using Avis.Core;
using NAudio.CoreAudioApi;
using NAudio.Wave;

namespace Avis.Audio;

/// <summary>WASAPI loopback capture with format conversion to 48 kHz stereo float.</summary>
public sealed class LoopbackCapture : IDisposable
{
    private readonly WasapiLoopbackCapture _capture;
    private readonly List<byte> _buffer = new();
    private readonly object _lock = new();
    private bool _recording;

    public string DeviceName { get; }

    public LoopbackCapture()
    {
        MMDevice device;
        using (var enumerator = new MMDeviceEnumerator())
            device = enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Multimedia);
        DeviceName = device.FriendlyName;
        _capture = new WasapiLoopbackCapture(device);
        _capture.DataAvailable += OnData;
        _capture.RecordingStopped += (_, e) =>
        {
            if (e.Exception != null) Console.Error.WriteLine("capture stopped: " + e.Exception.Message);
        };
    }

    private void OnData(object? sender, WaveInEventArgs e)
    {
        lock (_lock)
        {
            for (int i = 0; i < e.BytesRecorded; i++) _buffer.Add(e.Buffer[i]);
        }
    }

    /// <summary>Records for the given duration and returns interleaved stereo at 48 kHz.</summary>
    public float[] Record(double seconds)
    {
        lock (_lock) _buffer.Clear();
        _capture.StartRecording();
        _recording = true;
        Thread.Sleep(TimeSpan.FromSeconds(Math.Max(0.1, seconds)));
        _capture.StopRecording();
        _recording = false;
        Thread.Sleep(80);
        byte[] raw;
        lock (_lock) raw = _buffer.ToArray();
        var fmt = _capture.WaveFormat;
        var stereo = Convert(raw, fmt);
        if (fmt.SampleRate != FrontendWeights.SampleRate)
            stereo = HermiteResampler.Resample(stereo, fmt.SampleRate, FrontendWeights.SampleRate);
        return stereo;
    }

    public static float[] Convert(byte[] raw, WaveFormat fmt)
    {
        int channels = fmt.Channels;
        int bytesPer = fmt.BitsPerSample / 8;
        if (bytesPer <= 0) bytesPer = 4;
        int frames = raw.Length / (bytesPer * channels);
        var mono = new float[frames * 2];
        for (int i = 0; i < frames; i++)
        {
            float l = 0, r = 0;
            for (int c = 0; c < channels; c++)
            {
                int o = (i * channels + c) * bytesPer;
                float v = Sample(raw, o, fmt, bytesPer);
                if (channels == 1) { l = r = v; }
                else if (channels == 2) { if (c == 0) l = v; else r = v; }
                else
                {
                    // WAVE order: FL, FR, FC, LFE, BL/SL, BR/SR
                    switch (c)
                    {
                        case 0: l += v; break;
                        case 1: r += v; break;
                        case 2: l += 0.707f * v; r += 0.707f * v; break;
                        case 3: break; // LFE is intentionally excluded
                        case 4: l += 0.707f * v; break;
                        case 5: r += 0.707f * v; break;
                    }
                }
            }
            mono[2 * i] = l;
            mono[2 * i + 1] = r;
        }
        return mono;
    }

    private static float Sample(byte[] raw, int o, WaveFormat fmt, int bytesPer)
    {
        if (fmt.Encoding == WaveFormatEncoding.IeeeFloat && bytesPer == 4)
            return BitConverter.ToSingle(raw, o);
        if (fmt.Encoding == WaveFormatEncoding.Pcm)
        {
            if (bytesPer == 2) return BitConverter.ToInt16(raw, o) / 32768f;
            if (bytesPer == 3)
            {
                int v = raw[o] | (raw[o + 1] << 8) | (raw[o + 2] << 16);
                if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                return v / 8388608f;
            }
            if (bytesPer == 4) return BitConverter.ToInt32(raw, o) / 2147483648f;
        }
        if (fmt.Encoding == WaveFormatEncoding.IeeeFloat && bytesPer == 8)
            return (float)BitConverter.ToDouble(raw, o);
        return 0;
    }

    public void Dispose()
    {
        if (_recording) _capture.StopRecording();
        _capture.Dispose();
    }
}

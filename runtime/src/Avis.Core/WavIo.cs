using System;
using System.IO;
using System.Text;

namespace Avis.Core;

/// <summary>Minimal WAV reader/writer (PCM16, PCM24, IEEE float32) without NAudio.</summary>
public static class WavIo
{
    public sealed class Wav
    {
        public int SampleRate;
        public int Channels;
        public float[] Interleaved = Array.Empty<float>();
        public int Frames => Channels > 0 ? Interleaved.Length / Channels : 0;
        public double Duration => SampleRate > 0 ? Frames / (double)SampleRate : 0;
    }

    public static Wav Read(string path)
    {
        using var fs = File.OpenRead(path);
        using var br = new BinaryReader(fs);
        var riff = Encoding.ASCII.GetString(br.ReadBytes(4));
        if (riff != "RIFF") throw new InvalidDataException("not RIFF");
        br.ReadInt32();
        if (Encoding.ASCII.GetString(br.ReadBytes(4)) != "WAVE") throw new InvalidDataException("not WAVE");
        int fmt = 0, channels = 0, sr = 0, bits = 0;
        byte[]? data = null;
        while (fs.Position + 8 <= fs.Length)
        {
            var id = Encoding.ASCII.GetString(br.ReadBytes(4));
            int size = br.ReadInt32();
            if (id == "fmt ")
            {
                fmt = br.ReadUInt16();
                channels = br.ReadUInt16();
                sr = br.ReadInt32();
                br.ReadInt32();
                br.ReadUInt16();
                bits = br.ReadUInt16();
                if (size > 16) br.ReadBytes(size - 16);
            }
            else if (id == "data")
            {
                data = br.ReadBytes(size);
            }
            else br.ReadBytes(size);
            if (size % 2 == 1 && fs.Position < fs.Length) br.ReadByte();
        }
        if (data == null || channels == 0) throw new InvalidDataException("no data");
        int bytesPer = bits / 8;
        int frames = data.Length / (bytesPer * channels);
        var samples = new float[frames * channels];
        for (int i = 0; i < samples.Length; i++)
        {
            int o = i * bytesPer;
            if (fmt == 3 && bits == 32) samples[i] = BitConverter.ToSingle(data, o);
            else if (bits == 16) samples[i] = BitConverter.ToInt16(data, o) / 32768f;
            else if (bits == 24)
            {
                int v = data[o] | (data[o + 1] << 8) | (data[o + 2] << 16);
                if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                samples[i] = v / 8388608f;
            }
            else if (bits == 8) samples[i] = (data[o] - 128) / 128f;
            else throw new InvalidDataException($"unsupported {fmt}/{bits}");
        }
        return new Wav { SampleRate = sr, Channels = channels, Interleaved = samples };
    }

    public static void Write(string path, float[] interleaved, int channels, int sampleRate)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        int bytes = interleaved.Length * 4;
        using var fs = File.Create(path);
        using var bw = new BinaryWriter(fs);
        bw.Write(Encoding.ASCII.GetBytes("RIFF"));
        bw.Write(36 + bytes);
        bw.Write(Encoding.ASCII.GetBytes("WAVEfmt "));
        bw.Write(16);
        bw.Write((ushort)3);
        bw.Write((ushort)channels);
        bw.Write(sampleRate);
        bw.Write(sampleRate * channels * 4);
        bw.Write((ushort)(channels * 4));
        bw.Write((ushort)32);
        bw.Write(Encoding.ASCII.GetBytes("data"));
        bw.Write(bytes);
        foreach (var v in interleaved) bw.Write(v);
    }

    /// <summary>Downmix any channel count to stereo per SPEC G.</summary>
    public static float[] ToStereo(float[] src, int channels)
    {
        if (channels == 2) return src;
        int frames = src.Length / channels;
        var dst = new float[frames * 2];
        for (int i = 0; i < frames; i++)
        {
            float l, r;
            if (channels == 1) { l = r = src[i]; }
            else
            {
                // L=FL+0.707C+0.707SL, R=FR+0.707C+0.707SR (other channels ignored)
                float fl = src[i * channels];
                float fr = src[i * channels + 1];
                float c = channels > 2 ? src[i * channels + 2] : 0;
                float sl = channels > 4 ? src[i * channels + 4] : 0;
                float sr = channels > 5 ? src[i * channels + 5] : 0;
                l = fl + 0.707f * c + 0.707f * sl;
                r = fr + 0.707f * c + 0.707f * sr;
            }
            dst[2 * i] = l;
            dst[2 * i + 1] = r;
        }
        return dst;
    }
}

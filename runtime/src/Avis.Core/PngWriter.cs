using System;
using System.IO;
using System.IO.Compression;

namespace Avis.Core;

/// <summary>Small dependency-free PNG writer (RGBA8, filter 0).</summary>
public static class PngWriter
{
    private static readonly uint[] CrcTable = BuildCrc();

    private static uint[] BuildCrc()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }

    private static uint Crc(ReadOnlySpan<byte> data)
    {
        uint c = 0xFFFFFFFF;
        foreach (var b in data) c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
        return c ^ 0xFFFFFFFF;
    }

    private static void Chunk(Stream s, string type, ReadOnlySpan<byte> data)
    {
        Span<byte> len = stackalloc byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(len, (uint)data.Length);
        s.Write(len);
        Span<byte> t = stackalloc byte[4];
        for (int i = 0; i < 4; i++) t[i] = (byte)type[i];
        s.Write(t);
        s.Write(data);
        var crcInput = new byte[4 + data.Length];
        t.CopyTo(crcInput);
        data.CopyTo(crcInput.AsSpan(4));
        Span<byte> crc = stackalloc byte[4];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(crc, Crc(crcInput));
        s.Write(crc);
    }

    /// <summary>pixels: RGBA8, row-major, width*height*4.</summary>
    public static void Write(string path, int width, int height, ReadOnlySpan<byte> rgba)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var fs = File.Create(path);
        fs.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
        Span<byte> ihdr = stackalloc byte[13];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr, (uint)width);
        System.Buffers.Binary.BinaryPrimitives.WriteUInt32BigEndian(ihdr[4..], (uint)height);
        ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
        Chunk(fs, "IHDR", ihdr);
        using var ms = new MemoryStream();
        using (var z = new ZLibStream(ms, CompressionLevel.Fastest, leaveOpen: true))
        {
            for (int y = 0; y < height; y++)
            {
                z.WriteByte(0);
                z.Write(rgba.Slice(y * width * 4, width * 4));
            }
        }
        Chunk(fs, "IDAT", ms.ToArray());
        Chunk(fs, "IEND", ReadOnlySpan<byte>.Empty);
    }

    /// <summary>Read RGBA8 PNG produced by this writer (filter 0 only).</summary>
    public static (int width, int height, byte[] rgba) Read(string path)
    {
        var all = File.ReadAllBytes(path);
        int off = 8, w = 0, h = 0;
        using var idat = new MemoryStream();
        while (off + 12 <= all.Length)
        {
            int len = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(all.AsSpan(off));
            string type = System.Text.Encoding.ASCII.GetString(all, off + 4, 4);
            var data = all.AsSpan(off + 8, len);
            if (type == "IHDR")
            {
                w = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(data);
                h = (int)System.Buffers.Binary.BinaryPrimitives.ReadUInt32BigEndian(data[4..]);
            }
            else if (type == "IDAT") idat.Write(data);
            else if (type == "IEND") break;
            off += 12 + len;
        }
        idat.Position = 0;
        using var z = new ZLibStream(idat, CompressionMode.Decompress);
        var raw = new MemoryStream();
        z.CopyTo(raw);
        var src = raw.ToArray();
        var rgba = new byte[w * h * 4];
        int stride = w * 4 + 1;
        for (int y = 0; y < h; y++)
        {
            if (src[y * stride] != 0) throw new InvalidDataException("only filter 0 supported");
            Buffer.BlockCopy(src, y * stride + 1, rgba, y * w * 4, w * 4);
        }
        return (w, h, rgba);
    }
}

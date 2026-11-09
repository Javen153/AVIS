using System;
using System.Runtime.InteropServices;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace Avis.Render;

/// <summary>
/// D3D11 real-time renderer: instanced anisotropic splats into HDR, 5-level bloom,
/// ACES tonemap, premultiplied alpha. The same code path renders to a swapchain or
/// to an offscreen texture for PNG output (WARP when no GPU is available).
/// </summary>
public sealed partial class Renderer : IDisposable
{
    public const int BloomLevels = 5;
    private const int MaxInstances = 1024;
    private const int InstanceStride = 48;

    public ID3D11Device Device { get; }
    public ID3D11DeviceContext Context { get; }
    public bool IsWarp { get; }
    public string DeviceName { get; private set; } = "unknown";
    public int Width { get; private set; }
    public int Height { get; private set; }

    private readonly ID3D11VertexShader _vsSplat;
    private readonly ID3D11PixelShader _psSplat;
    private readonly ID3D11VertexShader _vsFull;
    private readonly ID3D11PixelShader _psDown;
    private readonly ID3D11PixelShader _psUp;
    private readonly ID3D11PixelShader _psComposite;
    private readonly ID3D11PixelShader _psSpectro;
    private readonly ID3D11Buffer _instances;
    private readonly ID3D11Buffer _uniforms;
    private readonly ID3D11Buffer _spectroUniforms;
    private readonly ID3D11Buffer _bandBuffer;
    private readonly ID3D11Buffer _ownerBuffer;
    private readonly ID3D11ShaderResourceView _instancesView;
    private readonly ID3D11ShaderResourceView _bandView;
    private readonly ID3D11ShaderResourceView _ownerView;
    private readonly ID3D11SamplerState _sampler;
    private readonly ID3D11BlendState _additive;
    private readonly ID3D11BlendState _alpha;
    private readonly ID3D11RasterizerState _raster;
    private readonly ID3D11DepthStencilState _noDepth;

    private ID3D11Texture2D? _hdr;
    private ID3D11RenderTargetView? _hdrRtv;
    private ID3D11ShaderResourceView? _hdrSrv;
    private readonly ID3D11Texture2D[] _bloom = new ID3D11Texture2D[BloomLevels];
    private readonly ID3D11RenderTargetView[] _bloomRtv = new ID3D11RenderTargetView[BloomLevels];
    private readonly ID3D11ShaderResourceView[] _bloomSrv = new ID3D11ShaderResourceView[BloomLevels];
    private ID3D11Texture2D? _ldr;
    private ID3D11RenderTargetView? _ldrRtv;
    private ID3D11Texture2D? _staging;
    private int _bloomW, _bloomH;

    private readonly float[] _inst = new float[MaxInstances * 12];
    private readonly float[] _bands = new float[192];
    private readonly float[] _owners = new float[192 * 4];
    private int _instanceCount;

    public Renderer(int width, int height, bool preferWarp = false)
    {
        Width = Math.Max(64, width);
        Height = Math.Max(64, height);
        var levels = new[] { 0xb100, 0xb000 }; // D3D_FEATURE_LEVEL_11_1, _11_0
        int hr = CreateDeviceNative(IntPtr.Zero, preferWarp ? 5 : 1, IntPtr.Zero, 0x20 /*BGRA*/, levels, (uint)levels.Length, 7,
            out IntPtr devPtr, out int level, out IntPtr ctxPtr);
        if (hr < 0 || devPtr == IntPtr.Zero)
        {
            hr = CreateDeviceNative(IntPtr.Zero, 5, IntPtr.Zero, 0x20, levels, (uint)levels.Length, 7,
                out devPtr, out level, out ctxPtr);
            IsWarp = true;
        }
        if (hr < 0 || devPtr == IntPtr.Zero) throw new InvalidOperationException($"D3D11 device creation failed: 0x{hr:X8}");
        Device = new ID3D11Device(devPtr);
        Context = new ID3D11DeviceContext(ctxPtr);
        try
        {
            using var dxgi = Device.QueryInterface<IDXGIDevice>();
            using var adapter = dxgi.GetAdapter();
            var desc = adapter.Description;
            DeviceName = desc.Description.TrimEnd('\0');
        }
        catch { DeviceName = IsWarp ? "WARP" : "GPU"; }

        _vsSplat = CompileVs(Shaders.Splat, "VSMain");
        _psSplat = CompilePs(Shaders.Splat, "PSMain");
        _vsFull = CompileVs(Shaders.Fullscreen, "VSFull");
        _psDown = CompilePs(Shaders.Down, "PSDown");
        _psUp = CompilePs(Shaders.Up, "PSUP");
        _psComposite = CompilePs(Shaders.Composite, "PSComposite");
        _psSpectro = CompilePs(Shaders.Spectro, "PSSpectro");

        _instances = Device.CreateBuffer((uint)(MaxInstances * InstanceStride), BindFlags.ShaderResource,
            ResourceUsage.Dynamic, CpuAccessFlags.Write, ResourceOptionFlags.BufferStructured, InstanceStride);
        _instancesView = Device.CreateShaderResourceView(_instances);
        _bandBuffer = Device.CreateBuffer(192u * 4u, BindFlags.ShaderResource, ResourceUsage.Dynamic,
            CpuAccessFlags.Write, ResourceOptionFlags.BufferStructured, 4);
        _bandView = Device.CreateShaderResourceView(_bandBuffer);
        _ownerBuffer = Device.CreateBuffer(192u * 16u, BindFlags.ShaderResource, ResourceUsage.Dynamic,
            CpuAccessFlags.Write, ResourceOptionFlags.BufferStructured, 16);
        _ownerView = Device.CreateShaderResourceView(_ownerBuffer);
        _uniforms = Device.CreateBuffer(256u, BindFlags.ConstantBuffer, ResourceUsage.Dynamic, CpuAccessFlags.Write);
        _spectroUniforms = Device.CreateBuffer(256u, BindFlags.ConstantBuffer, ResourceUsage.Dynamic, CpuAccessFlags.Write);

        _sampler = Device.CreateSamplerState(new SamplerDescription
        {
            Filter = Filter.MinMagMipLinear,
            AddressU = TextureAddressMode.Clamp,
            AddressV = TextureAddressMode.Clamp,
            AddressW = TextureAddressMode.Clamp,
            MipLODBias = 0,
            MaxAnisotropy = 1,
            ComparisonFunc = ComparisonFunction.Never,
            BorderColor = new Color4(0, 0, 0, 0),
            MinLOD = 0,
            MaxLOD = float.MaxValue,
        });
        _additive = Device.CreateBlendState(BlendDescription.Additive);
        _alpha = Device.CreateBlendState(BlendDescription.AlphaBlend);
        _raster = Device.CreateRasterizerState(new RasterizerDescription
        {
            FillMode = FillMode.Solid,
            CullMode = CullMode.None,
            FrontCounterClockwise = false,
            DepthBias = 0,
            DepthBiasClamp = 0,
            SlopeScaledDepthBias = 0,
            DepthClipEnable = true,
            ScissorEnable = false,
            MultisampleEnable = false,
            AntialiasedLineEnable = false,
        });
        _noDepth = Device.CreateDepthStencilState(new DepthStencilDescription
        {
            DepthEnable = false,
            DepthWriteMask = DepthWriteMask.Zero,
            DepthFunc = ComparisonFunction.Always,
            StencilEnable = false,
        });

        CreateTargets();
    }

    // ---------------------------------------------------------------- resources
    private void CreateTargets()
    {
        ReleaseTargets();
        _hdr = Device.CreateTexture2D(new Texture2DDescription
        {
            Width = (uint)Width,
            Height = (uint)Height,
            MipLevels = 1,
            ArraySize = 1,
            Format = Format.R16G16B16A16_Float,
            SampleDescription = new SampleDescription(1, 0),
            Usage = ResourceUsage.Default,
            BindFlags = BindFlags.RenderTarget | BindFlags.ShaderResource,
            CPUAccessFlags = CpuAccessFlags.None,
            MiscFlags = ResourceOptionFlags.None,
        });
        _hdrRtv = Device.CreateRenderTargetView(_hdr);
        _hdrSrv = Device.CreateShaderResourceView(_hdr);

        _ldr = Device.CreateTexture2D(new Texture2DDescription
        {
            Width = (uint)Width,
            Height = (uint)Height,
            MipLevels = 1,
            ArraySize = 1,
            Format = Format.R8G8B8A8_UNorm,
            SampleDescription = new SampleDescription(1, 0),
            Usage = ResourceUsage.Default,
            BindFlags = BindFlags.RenderTarget | BindFlags.ShaderResource,
            CPUAccessFlags = CpuAccessFlags.None,
            MiscFlags = ResourceOptionFlags.None,
        });
        _ldrRtv = Device.CreateRenderTargetView(_ldr);

        _staging = Device.CreateTexture2D(new Texture2DDescription
        {
            Width = (uint)Width,
            Height = (uint)Height,
            MipLevels = 1,
            ArraySize = 1,
            Format = Format.R8G8B8A8_UNorm,
            SampleDescription = new SampleDescription(1, 0),
            Usage = ResourceUsage.Staging,
            BindFlags = BindFlags.None,
            CPUAccessFlags = CpuAccessFlags.Read,
            MiscFlags = ResourceOptionFlags.None,
        });

        _bloomW = Math.Max(1, Width / 2);
        _bloomH = Math.Max(1, Height / 2);
        for (int i = 0; i < BloomLevels; i++)
        {
            _bloom[i] = Device.CreateTexture2D(new Texture2DDescription
            {
                Width = (uint)_bloomW,
                Height = (uint)_bloomH,
                MipLevels = 1,
                ArraySize = 1,
                Format = Format.R16G16B16A16_Float,
                SampleDescription = new SampleDescription(1, 0),
                Usage = ResourceUsage.Default,
                BindFlags = BindFlags.RenderTarget | BindFlags.ShaderResource,
                CPUAccessFlags = CpuAccessFlags.None,
                MiscFlags = ResourceOptionFlags.None,
            });
            _bloomRtv[i] = Device.CreateRenderTargetView(_bloom[i]);
            _bloomSrv[i] = Device.CreateShaderResourceView(_bloom[i]);
            _bloomW = Math.Max(1, _bloomW / 2);
            _bloomH = Math.Max(1, _bloomH / 2);
        }
    }

    private void ReleaseTargets()
    {
        _hdr?.Dispose(); _hdr = null;
        _hdrRtv?.Dispose(); _hdrRtv = null;
        _hdrSrv?.Dispose(); _hdrSrv = null;
        _ldr?.Dispose(); _ldr = null;
        _ldrRtv?.Dispose(); _ldrRtv = null;
        _staging?.Dispose(); _staging = null;
        for (int i = 0; i < BloomLevels; i++)
        {
            _bloom[i]?.Dispose(); _bloom[i] = null!;
            _bloomRtv[i]?.Dispose(); _bloomRtv[i] = null!;
            _bloomSrv[i]?.Dispose(); _bloomSrv[i] = null!;
        }
    }

    public void Resize(int width, int height)
    {
        width = Math.Max(64, width);
        height = Math.Max(64, height);
        if (width == Width && height == Height) return;
        Width = width; Height = height;
        CreateTargets();
    }

    // ---------------------------------------------------------------- shaders
    private ID3D11VertexShader CompileVs(string src, string entry) =>
        Device.CreateVertexShader(Compile(src, entry, "vs_5_0"));

    private ID3D11PixelShader CompilePs(string src, string entry) =>
        Device.CreatePixelShader(Compile(src, entry, "ps_5_0"));

    private byte[] Compile(string source, string entry, string profile)
    {
        var bytes = System.Text.Encoding.UTF8.GetBytes(source);
        IntPtr code = IntPtr.Zero, errors = IntPtr.Zero;
        int hr = D3DCompile(bytes, (UIntPtr)(uint)bytes.Length, "avis.hlsl", IntPtr.Zero, IntPtr.Zero,
            entry, profile, 0, 0, out code, out errors);
        if (hr < 0)
        {
            string msg = errors != IntPtr.Zero ? Marshal.PtrToStringAnsi(errors) ?? "" : "";
            if (errors != IntPtr.Zero) Marshal.Release(errors);
            throw new InvalidOperationException($"shader compile failed ({profile} {entry}): {msg}");
        }
        try
        {
            var vtable = Marshal.ReadIntPtr(code);
            var getPtr = Marshal.GetDelegateForFunctionPointer<GetBufferPointerFn>(Marshal.ReadIntPtr(vtable, 3 * IntPtr.Size));
            var getSize = Marshal.GetDelegateForFunctionPointer<GetBufferSizeFn>(Marshal.ReadIntPtr(vtable, 4 * IntPtr.Size));
            IntPtr data = getPtr(code);
            int size = (int)getSize(code);
            var result = new byte[size];
            Marshal.Copy(data, result, 0, size);
            return result;
        }
        finally
        {
            if (code != IntPtr.Zero) Marshal.Release(code);
            if (errors != IntPtr.Zero) Marshal.Release(errors);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate IntPtr GetBufferPointerFn(IntPtr thisPtr);

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate UIntPtr GetBufferSizeFn(IntPtr thisPtr);

    [DllImport("d3dcompiler_47.dll", CharSet = CharSet.Ansi)]
    private static extern int D3DCompile(byte[] srcData, UIntPtr srcDataSize, string sourceName,
        IntPtr defines, IntPtr include, string entryPoint, string target,
        uint flags1, uint flags2, out IntPtr code, out IntPtr errorMsgs);

    [DllImport("d3d11.dll", EntryPoint = "D3D11CreateDevice")]
    private static extern int CreateDeviceNative(IntPtr adapter, int driverType, IntPtr software,
        uint flags, int[] featureLevels, uint numLevels, uint sdkVersion,
        out IntPtr device, out int featureLevel, out IntPtr context);

    // ---------------------------------------------------------------- io
    /// <summary>Copies the last LDR result into a staging texture and returns RGBA8 rows top-down.</summary>
    public byte[] ReadPixels()
    {
        if (_ldr is null || _staging is null) throw new InvalidOperationException("renderer has no targets");
        Context.CopyResource(_staging, _ldr);
        var map = Context.Map(_staging, 0, MapMode.Read, Vortice.Direct3D11.MapFlags.None);
        var rgba = new byte[Width * Height * 4];
        try
        {
            for (int y = 0; y < Height; y++)
                Marshal.Copy(IntPtr.Add(map.DataPointer, y * (int)map.RowPitch), rgba, y * Width * 4, Width * 4);
        }
        finally
        {
            Context.Unmap(_staging, 0);
        }
        return rgba;
    }

    public void Dispose()
    {
        ReleaseTargets();
        _instancesView.Dispose(); _instances.Dispose();
        _bandView.Dispose(); _bandBuffer.Dispose();
        _ownerView.Dispose(); _ownerBuffer.Dispose();
        _uniforms.Dispose(); _spectroUniforms.Dispose();
        _sampler.Dispose(); _additive.Dispose(); _alpha.Dispose();
        _raster.Dispose(); _noDepth.Dispose();
        _vsSplat.Dispose(); _psSplat.Dispose();
        _vsFull.Dispose(); _psDown.Dispose(); _psUp.Dispose();
        _psComposite.Dispose(); _psSpectro.Dispose();
        Context.Dispose();
        Device.Dispose();
    }
}

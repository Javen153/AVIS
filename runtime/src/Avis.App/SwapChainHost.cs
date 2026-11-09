using System.Runtime.InteropServices;
using Avis.Core;
using Avis.Render;
using Microsoft.UI.Xaml.Controls;
using Vortice.Direct3D11;
using Vortice.DXGI;

namespace Avis.App;

/// <summary>Composition swap chain attached to a WinUI 3 SwapChainPanel.</summary>
public sealed class SwapChainHost : IDisposable
{
    private static readonly Guid IID_ISwapChainPanelNative = new("63aad0b8-7c24-40ff-85a8-640d944cc325");
    private static readonly Guid IID_ISwapChainPanelNativeOld = new("f92f19d2-3ade-45a6-a20c-f6f1ea90554b");

    private readonly Renderer _renderer;
    private readonly SwapChainPanel _panel;
    private IDXGISwapChain1? _swap;
    private ID3D11RenderTargetView? _rtv;
    private int _width = 1, _height = 1;

    public int Width => _width;
    public int Height => _height;
    public bool Ready => _swap != null && _rtv != null;
    public string? Error { get; private set; }

    public SwapChainHost(Renderer renderer, SwapChainPanel panel)
    {
        _renderer = renderer;
        _panel = panel;
    }

    public void Ensure(int width, int height)
    {
        width = Math.Max(64, width);
        height = Math.Max(64, height);
        if (_swap == null)
        {
            Create(width, height);
            return;
        }
        if (width == _width && height == _height) return;
        _rtv?.Dispose();
        _rtv = null;
        try
        {
            _swap.ResizeBuffers(2, (uint)width, (uint)height, Format.B8G8R8A8_UNorm, SwapChainFlags.None);
            using var back = _swap.GetBuffer<ID3D11Texture2D>(0);
            _rtv = _renderer.Device.CreateRenderTargetView(back);
            _renderer.Resize(width, height);
            _width = width;
            _height = height;
        }
        catch (Exception ex)
        {
            Error = "resize: " + ex.Message;
        }
    }

    private void Create(int width, int height)
    {
        try
        {
            using var dxgiDevice = _renderer.Device.QueryInterface<IDXGIDevice>();
            using var adapter = dxgiDevice.GetAdapter();
            using var factory = adapter.GetParent<IDXGIFactory2>();
            var desc = new SwapChainDescription1
            {
                Width = (uint)width,
                Height = (uint)height,
                Format = Format.B8G8R8A8_UNorm,
                Stereo = false,
                SampleDescription = new SampleDescription(1, 0),
                BufferUsage = Usage.RenderTargetOutput,
                BufferCount = 2,
                Scaling = Scaling.Stretch,
                SwapEffect = SwapEffect.FlipSequential,
                AlphaMode = AlphaMode.Premultiplied,
                Flags = SwapChainFlags.None,
            };
            _swap = factory.CreateSwapChainForComposition(_renderer.Device, desc);
            using var back = _swap.GetBuffer<ID3D11Texture2D>(0);
            _rtv = _renderer.Device.CreateRenderTargetView(back);
            _width = width;
            _height = height;
            Attach();
            Error = null;
        }
        catch (Exception ex)
        {
            Error = "swap chain: " + ex.Message;
            _swap = null;
            _rtv = null;
        }
    }

    private void Attach()
    {
        IntPtr unk = WinRT.MarshalInspectable<SwapChainPanel>.FromManaged(_panel);
        try
        {
            if (TrySet(unk, IID_ISwapChainPanelNative, _swap!.NativePointer)) return;
            if (TrySet(unk, IID_ISwapChainPanelNativeOld, _swap!.NativePointer)) return;
            throw new InvalidOperationException("ISwapChainPanelNative не найден");
        }
        finally
        {
            Marshal.Release(unk);
        }
    }

    private static bool TrySet(IntPtr unk, Guid iid, IntPtr swap)
    {
        int hr = Marshal.QueryInterface(unk, ref iid, out IntPtr native);
        if (hr != 0 || native == IntPtr.Zero) return false;
        try
        {
            var vtable = Marshal.ReadIntPtr(native);
            var fn = Marshal.GetDelegateForFunctionPointer<SetSwapChainFn>(
                Marshal.ReadIntPtr(vtable, 3 * IntPtr.Size));
            fn(native, swap);
            return true;
        }
        finally
        {
            Marshal.Release(native);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    private delegate int SetSwapChainFn(IntPtr thisPtr, IntPtr swapChain);

    public bool Render(AnalysisFrame frame, AppSettings settings, double t, double dt)
    {
        if (_swap == null || _rtv == null) return false;
        _renderer.Render(frame, settings, t, dt, _rtv);
        var hr = _swap.Present(1, PresentFlags.None);
        return hr >= 0 || hr == 0x087A0001; // DXGI_STATUS_OCCLUDED
    }

    public void Dispose()
    {
        _rtv?.Dispose();
        _swap?.Dispose();
        _rtv = null;
        _swap = null;
    }
}

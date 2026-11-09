using System.Runtime.InteropServices;

namespace Avis.App;

/// <summary>Small Win32 helpers: topmost, click-through overlay and a global hotkey.</summary>
public static class Win32
{
    private const int GWL_EXSTYLE = -20;
    private const long WS_EX_TRANSPARENT = 0x00000020;
    private const long WS_EX_LAYERED = 0x00080000;
    private static readonly IntPtr HWND_TOPMOST = new(-1);
    private static readonly IntPtr HWND_NOTOPMOST = new(-2);
    private const uint SWP_NOMOVE = 0x0002, SWP_NOSIZE = 0x0001, SWP_NOACTIVATE = 0x0010;

    public static void SetTopmost(IntPtr hwnd, bool on) =>
        SetWindowPos(hwnd, on ? HWND_TOPMOST : HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

    public static void SetClickThrough(IntPtr hwnd, bool on)
    {
        long style = GetWindowLongPtr(hwnd, GWL_EXSTYLE).ToInt64();
        if (on) style |= WS_EX_TRANSPARENT | WS_EX_LAYERED;
        else style &= ~(WS_EX_TRANSPARENT | WS_EX_LAYERED);
        SetWindowLongPtr(hwnd, GWL_EXSTYLE, new IntPtr(style));
    }

    /// <summary>Ctrl+Shift+O works globally, including while the overlay is click-through.</summary>
    public static void StartHotkey(Action action)
    {
        var thread = new Thread(() =>
        {
            const int id = 0xA715;
            if (!RegisterHotKey(IntPtr.Zero, id, 0x0002 | 0x0004, 0x4F)) return;
            try
            {
                while (GetMessage(out var msg, IntPtr.Zero, 0, 0) > 0)
                {
                    if (msg.message == 0x0312 && msg.wParam == id) action();
                }
            }
            finally { UnregisterHotKey(IntPtr.Zero, id); }
        })
        { IsBackground = true, Name = "avis-hotkey" };
        thread.Start();
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MSG
    {
        public IntPtr hwnd;
        public uint message;
        public IntPtr wParam;
        public IntPtr lParam;
        public uint time;
        public int x;
        public int y;
    }

    [DllImport("user32.dll")] private static extern bool SetWindowPos(IntPtr hwnd, IntPtr after, int x, int y, int cx, int cy, uint flags);
    [DllImport("user32.dll", EntryPoint = "GetWindowLongPtrW")] private static extern IntPtr GetWindowLongPtr(IntPtr hwnd, int index);
    [DllImport("user32.dll", EntryPoint = "SetWindowLongPtrW")] private static extern IntPtr SetWindowLongPtr(IntPtr hwnd, int index, IntPtr value);
    [DllImport("user32.dll")] private static extern bool RegisterHotKey(IntPtr hwnd, int id, uint modifiers, uint key);
    [DllImport("user32.dll")] private static extern bool UnregisterHotKey(IntPtr hwnd, int id);
    [DllImport("user32.dll")] private static extern int GetMessage(out MSG msg, IntPtr hwnd, uint min, uint max);
}

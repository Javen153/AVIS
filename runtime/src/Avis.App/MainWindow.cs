using System.Diagnostics;
using System.Text.Json;
using Avis.Core;
using Avis.Render;
using Microsoft.UI;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Imaging;
using Windows.UI;

namespace Avis.App;

public sealed class MainWindow : Window
{
    private readonly AppOptions _options;
    private readonly AppSettings _settings;
    private readonly SwapChainPanel _panel = new();
    private readonly TextBlock _stats = new();
    private readonly TextBlock _status = new();
    private readonly CommandBar _bar = new();
    private ComboBox _sourceCombo = null!;
    private ScrollViewer _settingsView = null!;
    private Renderer? _renderer;
    private SwapChainHost? _swap;
    private AvisEngine? _engine;
    private DispatcherQueueTimer? _timer;
    private readonly Stopwatch _clock = Stopwatch.StartNew();
    private readonly List<double> _frameTimes = new();
    private readonly List<double> _renderTimes = new();
    private double _lastTick;
    private double _lastStats;
    private bool _overlay;
    private bool _initialised;
    private bool _shotTaken;
    private bool _shotPending;
    private bool _benchWritten;
    private bool _recording;
    private string _lastProvider = "", _lastModel = "";
    private string? _initError;

    public MainWindow(AppOptions options)
    {
        _options = options;
        Title = "AVIS";
        _settings = AppSettings.Load(Path.Combine(AvisPaths.LocalAppData(), "settings.json"));
        _settings.ModelDirOverride = ResolveModelDir(options.ModelDir);
        foreach (var (path, value) in options.Sets) _settings.SetByPath(path, value);
        if (options.Overlay) _settings.Overlay = true;
        if (options.Topmost) _settings.AlwaysOnTop = true;

        try { SystemBackdrop = new MicaBackdrop(); } catch { }
        ExtendsContentIntoTitleBar = true;
        Content = BuildRoot();
        SetTitleBar(BuildTitleBar());

        Closed += (_, _) => Shutdown();
        Activated += (_, e) => { if (e.WindowActivationState != WindowActivationState.Deactivated) EnsureStarted(); };
        _panel.Loaded += (_, _) => EnsureStarted();
        _panel.SizeChanged += (_, _) => ResizeSwapChain();
    }

    // ------------------------------------------------------------------ UI
    private static string ResolveModelDir(string arg)
    {
        if (string.IsNullOrWhiteSpace(arg)) return "";
        if (Directory.Exists(arg)) return Path.GetFullPath(arg);
        return AvisPaths.ModelDir(arg);
    }

    private FrameworkElement BuildRoot()
    {
        var root = new Grid();
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        var body = new Grid();
        body.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        body.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        body.Children.Add(_panel);

        var overlay = new Grid { VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Stretch };
        BuildCommandBar(overlay);
        body.Children.Add(overlay);

        _stats.HorizontalAlignment = HorizontalAlignment.Left;
        _stats.VerticalAlignment = VerticalAlignment.Bottom;
        _stats.Margin = new Thickness(14, 0, 0, 10);
        _stats.FontFamily = new FontFamily("Consolas");
        _stats.FontSize = 12;
        _stats.Foreground = new SolidColorBrush(Colors.White);
        _stats.IsHitTestVisible = false;
        _stats.Visibility = _settings.ShowStats ? Visibility.Visible : Visibility.Collapsed;
        body.Children.Add(_stats);

        _shotPending = _options.Screenshot.Length > 0;
        _settingsView = SettingsPanel.Build(_settings, OnSettingsChanged);
        _settingsView.Width = 360;
        _settingsView.Visibility = _options.ShowSettings ? Visibility.Visible : Visibility.Collapsed;
        body.Children.Add(_settingsView);
        Grid.SetColumn(_settingsView, 1);

        root.Children.Add(body);
        return root;
    }

    private FrameworkElement BuildTitleBar()
    {
        var grid = new Grid { Height = 32, Padding = new Thickness(12, 0, 12, 0) };
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var title = new TextBlock
        {
            Text = "AVIS",
            VerticalAlignment = VerticalAlignment.Center,
            FontWeight = FontWeights.SemiBold,
            Margin = new Thickness(0, 0, 12, 0),
        };
        _status.VerticalAlignment = VerticalAlignment.Center;
        _status.FontSize = 12;
        _status.Opacity = 0.75;
        _status.Text = "запуск…";
        grid.Children.Add(title);
        grid.Children.Add(_status);
        Grid.SetColumn(_status, 1);
        var hint = new TextBlock
        {
            Text = "Ctrl+Shift+O — оверлей",
            FontSize = 11,
            Opacity = 0.5,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 12, 0),
        };
        grid.Children.Add(hint);
        Grid.SetColumn(hint, 2);
        return grid;
    }

    private void BuildCommandBar(Grid host)
    {
        _bar.DefaultLabelPosition = CommandBarDefaultLabelPosition.Right;
        _bar.IsOpen = false;
        _bar.Background = new SolidColorBrush(Color.FromArgb(190, 18, 18, 24));
        _bar.HorizontalAlignment = HorizontalAlignment.Stretch;

        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10, VerticalAlignment = VerticalAlignment.Center };
        var label = new TextBlock { Text = "Источник", VerticalAlignment = VerticalAlignment.Center, FontSize = 12 };
        _sourceCombo = new ComboBox
        {
            ItemsSource = new[] { "Нейро", "DSP", "Mock" },
            SelectedIndex = 0,
            Width = 120,
            VerticalAlignment = VerticalAlignment.Center,
        };
        _sourceCombo.SelectionChanged += (_, _) =>
        {
            if (!_initialised) return;
            string kind = _sourceCombo.SelectedIndex switch { 0 => "neural", 1 => "dsp", _ => "mock" };
            _engine?.SetSource(kind);
            UpdateStatus();
        };
        var presets = new ComboBox
        {
            ItemsSource = AppSettings.BuiltInPresets().Keys.ToArray(),
            PlaceholderText = "Пресет",
            Width = 130,
            VerticalAlignment = VerticalAlignment.Center,
        };
        presets.SelectionChanged += (_, _) =>
        {
            if (_initialised == false || presets.SelectedItem is not string name) return;
            var built = AppSettings.BuiltInPresets();
            if (!built.TryGetValue(name, out var p)) return;
            p.ModelDirOverride = _settings.ModelDirOverride;
            ApplyPreset(p);
        };
        content.Children.Add(label);
        content.Children.Add(_sourceCombo);
        content.Children.Add(presets);
        _bar.Content = content;

        _bar.PrimaryCommands.Add(Button("\uE713", "Настройки", ToggleSettings));
        _bar.PrimaryCommands.Add(Button("\uE740", "Оверлей", () => SetOverlay(!_overlay)));
        _bar.PrimaryCommands.Add(Button("\uE722", "Снимок", TakeScreenshot));
        host.Children.Add(_bar);
    }

    private static AppBarButton Button(string glyph, string label, Action action)
    {
        var b = new AppBarButton { Icon = new FontIcon { Glyph = glyph }, Label = label };
        b.Click += (_, _) => action();
        return b;
    }

    // ------------------------------------------------------------- startup
    private void EnsureStarted()
    {
        if (_initialised || _initError != null) return;
        if (_panel.ActualWidth < 8 || _panel.ActualHeight < 8) return;
        try
        {
            var (w, h) = PanelPixels();
            _renderer = new Renderer(w, h);
            _swap = new SwapChainHost(_renderer, _panel);
            _swap.Ensure(w, h);
            if (_swap.Error != null) throw new InvalidOperationException(_swap.Error);

            _engine = new AvisEngine(_settings) { WavPath = _options.WavPath };
            string kind = _options.Source.Length > 0 ? _options.Source : (NeuralAvailable() ? "neural" : "dsp");
            _engine.SetSource(kind);
            _engine.Start();
            _lastProvider = _settings.Provider.ToString();
            _lastModel = _settings.ModelName;
            if (_settings.RecordJsonl) StartRecording();
            ApplyBackdropSettings();
            _sourceCombo.SelectedIndex = _engine.Kind switch { "neural" => 0, "dsp" => 1, _ => 2 };
            _initialised = true;
            UpdateStatus();
            _timer = DispatcherQueue.CreateTimer();
            _timer.Interval = TimeSpan.FromMilliseconds(1000.0 / 144.0);
            _timer.Tick += OnTick;
            _timer.Start();
            if (_options.Overlay) SetOverlay(true);
        }
        catch (Exception ex)
        {
            _initError = ex.Message;
            _status.Text = "ошибка: " + ex.Message;
        }
    }

    private bool NeuralAvailable()
    {
        string dir = string.IsNullOrEmpty(_settings.ModelDirOverride) ? AvisPaths.ModelDir(_settings.ModelName) : _settings.ModelDirOverride;
        return AvisEngine.ModelAvailable(dir);
    }

    private (int, int) PanelPixels()
    {
        double scale = _panel.XamlRoot?.RasterizationScale ?? 1.0;
        return (Math.Max(64, (int)Math.Round(_panel.ActualWidth * scale)),
                Math.Max(64, (int)Math.Round(_panel.ActualHeight * scale)));
    }

    private void ResizeSwapChain()
    {
        if (!_initialised || _swap == null) return;
        var (w, h) = PanelPixels();
        _swap.Ensure(w, h);
    }

    // --------------------------------------------------------------- frame
    private void OnTick(DispatcherQueueTimer sender, object args)
    {
        double now = _clock.Elapsed.TotalSeconds;
        double dt = _lastTick > 0 ? Math.Min(0.1, now - _lastTick) : 1.0 / 60.0;
        _lastTick = now;
        if (_engine == null || _swap == null || _renderer == null) return;
        var frame = _engine.Latest;
        if (frame == null) return;
        try
        {
            var sw = Stopwatch.StartNew();
            bool ok = _swap.Render(frame, _settings, now, dt);
            sw.Stop();
            if (ok)
            {
                _renderTimes.Add(sw.Elapsed.TotalMilliseconds);
                if (_renderTimes.Count > 2400) _renderTimes.RemoveRange(0, 1200);
                if (_lastTick > 0) { _frameTimes.Add(dt * 1000.0); if (_frameTimes.Count > 2400) _frameTimes.RemoveRange(0, 1200); }
            }
        }
        catch (Exception ex)
        {
            _initError = ex.Message;
            _status.Text = "рендер: " + ex.Message;
            _timer?.Stop();
            return;
        }
        double up = now - _lastStats;
        if (up >= 0.35)
        {
            _lastStats = now;
            UpdateStats(now);
        }
        if (!_shotTaken && _options.Screenshot.Length > 0 && now >= _options.ScreenshotAt)
        {
            _shotTaken = true;
            TakeScreenshot();
            if (_options.Autoclose && _options.BenchSeconds <= 0) CloseSoon();
        }
        if (_options.BenchSeconds > 0 && !_benchWritten && now >= _options.BenchSeconds)
        {
            _benchWritten = true;
            WriteBench();
            if (_options.Autoclose) CloseSoon();
        }
        else if (_options.Autoclose && _options.BenchSeconds <= 0 && _options.Screenshot.Length == 0 && now >= 1.2)
        {
            CloseSoon();
        }
    }

    private void CloseSoon()
    {
        _timer?.Stop();
        DispatcherQueue.TryEnqueue(Microsoft.UI.Dispatching.DispatcherQueuePriority.Low, () => Close());
    }

    private void UpdateStats(double now)
    {
        double fps = _frameTimes.Count > 0 ? 1000.0 / Median(_frameTimes) : 0;
        double rt = _renderTimes.Count > 0 ? Median(_renderTimes) : 0;
        var at = _engine?.AnalysisTimes() ?? Array.Empty<double>();
        double an = at.Length > 0 ? Median(at) : 0;
        _stats.Text = $"FPS {fps,5:0.0}   рендер {rt,5:1.0} мс   анализ {an,5:1.0} мс   объектов {_engine?.ObjectsAvg ?? 0:0.0}";
        _stats.Visibility = _settings.ShowStats ? Visibility.Visible : Visibility.Collapsed;
    }

    private void UpdateStatus()
    {
        _status.Text = _engine == null ? "запуск…" : $"{_engine.Status} · {_renderer?.DeviceName}";
    }

    // ----------------------------------------------------------- settings
    private void ToggleSettings()
    {
        _settingsView.Visibility = _settingsView.Visibility == Visibility.Visible ? Visibility.Collapsed : Visibility.Visible;
    }

    private void OnSettingsChanged()
    {
        _engine?.ApplySettings(_settings);
        ApplyBackdropSettings();
        if (_settings.Overlay != _overlay) SetOverlay(_settings.Overlay);
        if (_settings.RecordJsonl && !_recording) StartRecording();
        else if (!_settings.RecordJsonl && _recording) { _recording = false; _engine?.StopRecording(); }
        if (_engine != null && _engine.Kind == "neural" &&
            (_lastProvider != _settings.Provider.ToString() || _lastModel != _settings.ModelName))
        {
            _lastProvider = _settings.Provider.ToString();
            _lastModel = _settings.ModelName;
            _engine.SetSource("neural");
        }
        UpdateStatus();
    }

    private void ApplyPreset(AppSettings preset)
    {
        foreach (var (prop, _) in AppSettings.Describe())
        {
            var value = prop.GetValue(preset);
            if (value != null) prop.SetValue(_settings, value);
        }
        // rebuild the panel so controls show the preset values
        var parent = _settingsView.Parent as Grid;
        if (parent != null)
        {
            int index = parent.Children.IndexOf(_settingsView);
            parent.Children.Remove(_settingsView);
            _settingsView = SettingsPanel.Build(_settings, OnSettingsChanged);
            _settingsView.Width = 360;
            parent.Children.Insert(index, _settingsView);
            Grid.SetColumn(_settingsView, 1);
        }
        OnSettingsChanged();
    }

    private void StartRecording()
    {
        string path = Path.Combine(AvisPaths.Root, "artifacts", "app", $"session_{DateTime.Now:yyyyMMdd_HHmmss}.jsonl");
        _engine?.StartRecording(path);
        _recording = true;
    }

    private void ApplyBackdropSettings()
    {
        try { SystemBackdrop = _settings.Material == WindowMaterial.Acrylic ? new DesktopAcrylicBackdrop() : new MicaBackdrop(); }
        catch { }
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        Win32.SetTopmost(hwnd, _settings.AlwaysOnTop && !_overlay);
    }

    // ------------------------------------------------------------ overlay
    private void SetOverlay(bool on)
    {
        _overlay = on;
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
        try
        {
            if (on)
            {
                this.AppWindow.SetPresenter(Microsoft.UI.Windowing.AppWindowPresenterKind.FullScreen);
                Win32.SetClickThrough(hwnd, true);
                Win32.SetTopmost(hwnd, true);
                _bar.Visibility = Visibility.Collapsed;
                _settingsView.Visibility = Visibility.Collapsed;
            }
            else
            {
                this.AppWindow.SetPresenter(Microsoft.UI.Windowing.AppWindowPresenterKind.Default);
                Win32.SetClickThrough(hwnd, false);
                Win32.SetTopmost(hwnd, _settings.AlwaysOnTop);
                _bar.Visibility = Visibility.Visible;
            }
        }
        catch (Exception ex) { _status.Text = "оверлей: " + ex.Message; }
        _settings.Overlay = on;
    }

    // ---------------------------------------------------------- screenshot
    private void TakeScreenshot()
    {
        if (_renderer == null || _engine?.Latest == null) return;
        string path = _shotPending && _options.Screenshot.Length > 0
            ? _options.Screenshot
            : Path.Combine(AvisPaths.Root, "artifacts", "app", $"shot_{DateTime.Now:HHmmss}.png");
        _shotPending = false;
        try
        {
            _renderer.Render(_engine.Latest, _settings, _clock.Elapsed.TotalSeconds, 1.0 / 60.0, null);
            var rgba = _renderer.ReadPixels();
            for (int i = 3; i < rgba.Length; i += 4) rgba[i] = 255; // premultiplied over black
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            PngWriter.Write(Path.GetFullPath(path), _renderer.Width, _renderer.Height, rgba);
            _status.Text = "снимок: " + Path.GetFileName(path);
        }
        catch (Exception ex) { _status.Text = "снимок: " + ex.Message; }
    }

    // --------------------------------------------------------------- bench
    private void WriteBench()
    {
        var at = _engine?.AnalysisTimes() ?? Array.Empty<double>();
        var fps = _frameTimes.Select(ms => 1000.0 / ms).ToArray();
        var bench = new
        {
            source = _engine?.Kind ?? "?",
            seconds = _clock.Elapsed.TotalSeconds,
            frames = _frameTimes.Count,
            fps_p50 = fps.Length > 0 ? Percent(fps, 50) : 0,
            fps_p05 = fps.Length > 0 ? Percent(fps, 5) : 0,
            fps_p95 = fps.Length > 0 ? Percent(fps, 95) : 0,
            render_ms_p50 = _renderTimes.Count > 0 ? Percent(_renderTimes.ToArray(), 50) : 0,
            render_ms_p95 = _renderTimes.Count > 0 ? Percent(_renderTimes.ToArray(), 95) : 0,
            analysis_ms_p50 = at.Length > 0 ? Percent(at, 50) : 0,
            analysis_ms_p99 = at.Length > 0 ? Percent(at, 99) : 0,
            objects_avg = _engine?.ObjectsAvg ?? 0,
            device = _renderer?.DeviceName ?? "?",
            resolution = _renderer == null ? "" : $"{_renderer.Width}x{_renderer.Height}",
            status = _engine?.Status ?? "",
        };
        string path = _options.BenchOut.Length > 0
            ? _options.BenchOut
            : Path.Combine(AvisPaths.Root, "artifacts", "app", "bench.json");
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, JsonSerializer.Serialize(bench, new JsonSerializerOptions { WriteIndented = true }));
        _status.Text = "bench: " + Path.GetFileName(path);
    }

    // ------------------------------------------------------------- helpers
    private static double Median(IReadOnlyList<double> v)
    {
        if (v.Count == 0) return 0;
        var a = v.ToArray();
        Array.Sort(a);
        return a[a.Length / 2];
    }

    private static double Percent(double[] v, double p)
    {
        if (v.Length == 0) return 0;
        var a = (double[])v.Clone();
        Array.Sort(a);
        int i = Math.Clamp((int)Math.Round(a.Length * p / 100.0), 0, a.Length - 1);
        return a[i];
    }

    private void Shutdown()
    {
        _timer?.Stop();
        try { _settings.Save(Path.Combine(AvisPaths.LocalAppData(), "settings.json")); } catch { }
        _swap?.Dispose();
        _renderer?.Dispose();
        _engine?.Dispose();
    }
}

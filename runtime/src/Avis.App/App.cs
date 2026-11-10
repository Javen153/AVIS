using Microsoft.UI.Xaml;

namespace Avis.App;

public sealed partial class App : Application
{
    private readonly AppOptions _options;
    private Window? _window;

    public App(AppOptions options)
    {
        _options = options;
        // Loads App.xaml, which merges the WinUI control styles through
        // XamlControlsResources. Merging them from code does not work on
        // Windows App SDK 2.6: it throws inside the framework (D-023).
        InitializeComponent();
        UnhandledException += OnUnhandledException;
        AppDomain.CurrentDomain.UnhandledException += (_, e) => Log("appdomain", e.ExceptionObject as Exception);
    }

    /// <summary>WinUI would otherwise tear the process down with a stowed exception
    /// (0xC000027B) and nothing would reach the log. Keep the window alive.</summary>
    private void OnUnhandledException(object sender, Microsoft.UI.Xaml.UnhandledExceptionEventArgs e)
    {
        Log("xaml", e.Exception);
        e.Handled = true;
    }

    private static void Log(string origin, Exception? ex)
    {
        try
        {
            string path = Path.Combine(Avis.Core.AvisPaths.LocalAppData(), "crash.log");
            var info = new FileInfo(path);
            if (info.Exists && info.Length > 256 * 1024) info.Delete();
            File.AppendAllText(path, $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] {origin}{Environment.NewLine}{ex}{Environment.NewLine}{Environment.NewLine}");
        }
        catch { }
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow(_options);
        _window.Activate();
    }
}

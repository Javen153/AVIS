using Microsoft.UI.Xaml;

namespace Avis.App;

public sealed class App : Application
{
    private readonly AppOptions _options;
    private Window? _window;

    public App(AppOptions options)
    {
        _options = options;
    }

    protected override void OnLaunched(LaunchActivatedEventArgs args)
    {
        _window = new MainWindow(_options);
        _window.Activate();
    }
}

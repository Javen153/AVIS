using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;

namespace Avis.App;

public static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        var options = AppOptions.Parse(args);
        WinRT.ComWrappersSupport.InitializeComWrappers();
        Application.Start(p =>
        {
            var context = new DispatcherQueueSynchronizationContext(DispatcherQueue.GetForCurrentThread());
            SynchronizationContext.SetSynchronizationContext(context);
            new App(options);
        });
    }
}

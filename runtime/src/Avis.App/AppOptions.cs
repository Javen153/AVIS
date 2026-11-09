using System.Globalization;

namespace Avis.App;

/// <summary>Command line surface required by the spec (section C).</summary>
public sealed class AppOptions
{
    public string Source { get; private set; } = "";
    public double BenchSeconds { get; private set; }
    public string BenchOut { get; private set; } = "";
    public bool Autoclose { get; private set; }
    public string Screenshot { get; private set; } = "";
    public double ScreenshotAt { get; private set; } = 1.2;
    public bool Overlay { get; private set; }
    public bool ShowSettings { get; private set; }
    public bool Topmost { get; private set; }
    public List<(string Path, string Value)> Sets { get; } = new();
    public string ModelDir { get; private set; } = "";
    public string WavPath { get; private set; } = "";

    public static AppOptions Parse(string[] args)
    {
        var o = new AppOptions();
        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i].ToLowerInvariant();
            string Next() => i + 1 < args.Length ? args[++i] : "";
            switch (a)
            {
                case "--source": o.Source = Next(); break;
                case "--bench": o.BenchSeconds = Num(Next()); break;
                case "--bench-out": o.BenchOut = Next(); break;
                case "--autoclose": o.Autoclose = true; break;
                case "--screenshot": o.Screenshot = Next(); break;
                case "--screenshot-at": o.ScreenshotAt = Num(Next()); break;
                case "--overlay": o.Overlay = true; break;
                case "--settings": o.ShowSettings = true; break;
                case "--topmost": o.Topmost = true; break;
                case "--model": o.ModelDir = Next(); break;
                case "--wav": o.WavPath = Next(); break;
                case "--set":
                {
                    string kv = Next();
                    int eq = kv.IndexOf('=');
                    if (eq > 0) o.Sets.Add((kv[..eq], kv[(eq + 1)..]));
                    break;
                }
            }
        }
        return o;
    }

    private static double Num(string s) =>
        double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : 0;
}

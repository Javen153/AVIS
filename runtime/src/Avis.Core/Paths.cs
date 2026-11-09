using System;
using System.IO;

namespace Avis.Core;

/// <summary>Locates shared assets both in the repository and in a published dist folder.</summary>
public static class AvisPaths
{
    public static string Root { get; } = FindRoot();

    private static string FindRoot()
    {
        var env = Environment.GetEnvironmentVariable("AVIS_ROOT");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env)) return Path.GetFullPath(env);
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "shared")) &&
                Directory.Exists(Path.Combine(dir.FullName, "runtime")))
                return dir.FullName;
            dir = dir.Parent;
        }
        // Published layout: <root>/AVIS.App.exe, <root>/shared, <root>/models
        return Path.GetFullPath(AppContext.BaseDirectory);
    }

    public static string Weights => Path.Combine(Root, "shared", "frontend", "v1", "weights.bin");
    public static string Golden => Path.Combine(Root, "shared", "golden");
    public static string Models => Path.Combine(Root, "models");
    public static string Scenarios => Path.Combine(Root, "testdata", "scenarios");

    public static string ModelDir(string name)
    {
        if (string.IsNullOrWhiteSpace(name)) name = "v1";
        return Path.IsPathRooted(name) ? name : Path.Combine(Models, name);
    }

    public static string LocalAppData()
    {
        var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AVIS");
        Directory.CreateDirectory(dir);
        return dir;
    }
}

using System;
using System.IO;

namespace Avis.Tests;

internal static class TestPaths
{
    public static string Root { get; } = FindRoot();

    private static string FindRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "shared")) &&
                Directory.Exists(Path.Combine(dir.FullName, "runtime")))
                return dir.FullName;
            dir = dir.Parent;
        }
        // Fallback: AVIS_ROOT environment variable
        var env = Environment.GetEnvironmentVariable("AVIS_ROOT");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env)) return env;
        throw new DirectoryNotFoundException("repository root not found from " + AppContext.BaseDirectory);
    }

    public static string Weights => Path.Combine(Root, "shared", "frontend", "v1", "weights.bin");
    public static string Golden => Path.Combine(Root, "shared", "golden");
    public static string Scenarios => Path.Combine(Root, "testdata", "scenarios");
}

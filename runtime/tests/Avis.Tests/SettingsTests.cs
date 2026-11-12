using System.Linq;
using Avis.Core;
using Xunit;

namespace Avis.Tests;

public sealed class SettingsTests
{
    [Fact]
    public void ModelNameOffersV1AndV21()
    {
        var attr = AppSettings.Describe()
            .Single(x => x.Prop.Name == nameof(AppSettings.ModelName)).Attr;
        Assert.NotNull(attr.Choices);
        Assert.Contains("v1", attr.Choices!);
        Assert.Contains("v2.1", attr.Choices!);
    }

    [Fact]
    public void ModelNameRoundTripsThroughCliAndDefaults()
    {
        var s = new AppSettings();
        Assert.True(s.SetByPath("neural.ModelName", "v2.1"));
        Assert.Equal("v2.1", s.ModelName);
        Assert.Equal("v2.1", AppSettings.FromJson(s.ToJson()).ModelName);
        s.ResetToDefaults();
        Assert.Equal("v1", s.ModelName);
    }
}

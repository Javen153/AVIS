using System.Globalization;
using System.Reflection;
using Avis.Core;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Avis.App;

/// <summary>Builds the auto-generated settings panel from <see cref="AppSettings.Describe"/>.</summary>
public static class SettingsPanel
{
    public static ScrollViewer Build(AppSettings settings, Action changed)
    {
        var root = new StackPanel { Spacing = 10, Padding = new Thickness(12, 8, 12, 24) };
        root.Children.Add(new TextBlock { Text = "Настройки", FontSize = 18, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold });
        var groups = AppSettings.Describe()
            .GroupBy(x => x.Attr.Group)
            .OrderBy(g => g.Key);
        foreach (var g in groups)
        {
            var inner = new StackPanel { Spacing = 10, Padding = new Thickness(0, 6, 0, 10) };
            var expander = new Expander
            {
                Header = g.First().Attr.GroupLabel,
                IsExpanded = g.Key == "visual",
                Content = inner,
                HorizontalAlignment = HorizontalAlignment.Stretch,
                HorizontalContentAlignment = HorizontalAlignment.Stretch,
            };
            foreach (var (prop, attr) in g)
                inner.Children.Add(Control(settings, prop, attr, changed));
            root.Children.Add(expander);
        }
        return new ScrollViewer
        {
            Content = root,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Microsoft.UI.Colors.Transparent),
        };
    }

    private static FrameworkElement Control(AppSettings settings, PropertyInfo prop, SettingAttribute attr, Action changed)
    {
        var panel = new StackPanel { Spacing = 2 };
        var t = prop.PropertyType;
        if (t == typeof(double))
        {
            var value = (double)prop.GetValue(settings)!;
            var head = new TextBlock { Text = $"{attr.Label}: {value:0.###}", FontSize = 12 };
            var slider = new Slider
            {
                Minimum = attr.Min,
                Maximum = attr.Max,
                StepFrequency = attr.Step <= 0 ? 0.01 : attr.Step,
                Value = Math.Clamp(value, attr.Min, attr.Max),
                IsThumbToolTipEnabled = false,
                Margin = new Thickness(0, -6, 0, 0),
            };
            slider.ValueChanged += (_, e) =>
            {
                prop.SetValue(settings, e.NewValue);
                head.Text = $"{attr.Label}: {e.NewValue:0.###}";
                changed();
            };
            panel.Children.Add(head);
            panel.Children.Add(slider);
            ToolTipService.SetToolTip(slider, attr.Description);
        }
        else if (t == typeof(bool))
        {
            var toggle = new ToggleSwitch
            {
                Header = attr.Label,
                IsOn = (bool)prop.GetValue(settings)!,
                Margin = new Thickness(0, -4, 0, 0),
            };
            toggle.Toggled += (_, _) => { prop.SetValue(settings, toggle.IsOn); changed(); };
            ToolTipService.SetToolTip(toggle, attr.Description);
            panel.Children.Add(toggle);
        }
        else if (t.IsEnum)
        {
            var names = Enum.GetNames(t);
            var combo = new ComboBox
            {
                Header = attr.Label,
                ItemsSource = names,
                SelectedIndex = Math.Max(0, Array.IndexOf(names, prop.GetValue(settings)!.ToString())),
                HorizontalAlignment = HorizontalAlignment.Stretch,
            };
            combo.SelectionChanged += (_, _) =>
            {
                if (combo.SelectedItem is string s) { prop.SetValue(settings, Enum.Parse(t, s, true)); changed(); }
            };
            ToolTipService.SetToolTip(combo, attr.Description);
            panel.Children.Add(combo);
        }
        else
        {
            var box = new TextBox
            {
                Header = attr.Label,
                Text = prop.GetValue(settings)?.ToString() ?? "",
                PlaceholderText = attr.Description,
            };
            box.TextChanged += (_, _) => { prop.SetValue(settings, box.Text); changed(); };
            panel.Children.Add(box);
        }
        return panel;
    }

    public static string Format(double v) => v.ToString("0.###", CultureInfo.InvariantCulture);
}

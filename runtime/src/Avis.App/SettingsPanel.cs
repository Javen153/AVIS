using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using Avis.Core;
using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace Avis.App;

/// <summary>Builds the auto-generated settings panel from <see cref="AppSettings.Describe"/>.</summary>
public static class SettingsPanel
{
    public static ScrollViewer Build(AppSettings settings, Action changed, Action? reset = null)
    {
        var defaults = new AppSettings();
        var root = new StackPanel { Spacing = 10, Padding = new Thickness(12, 8, 12, 24) };
        root.Children.Add(Header(reset));
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
                inner.Children.Add(Control(settings, defaults, prop, attr, changed));
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

    private static Grid Header(Action? reset)
    {
        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var title = new TextBlock
        {
            Text = "Настройки",
            FontSize = 18,
            FontWeight = FontWeights.SemiBold,
            VerticalAlignment = VerticalAlignment.Center,
        };
        grid.Children.Add(title);
        if (reset != null)
        {
            var button = new Button
            {
                Content = "По умолчанию",
                FontSize = 12,
                Padding = new Thickness(10, 4, 10, 4),
                VerticalAlignment = VerticalAlignment.Center,
            };
            button.Click += (_, _) => reset();
            ToolTipService.SetToolTip(button,
                "Вернуть все параметры к значениям по умолчанию.\nПуть к модели и машинные пути не меняются.");
            grid.Children.Add(button);
            Grid.SetColumn(button, 1);
        }
        return grid;
    }

    private static FrameworkElement Control(AppSettings settings, AppSettings defaults,
        PropertyInfo prop, SettingAttribute attr, Action changed)
    {
        var panel = new StackPanel { Spacing = 2 };
        var t = prop.PropertyType;
        string tip = Tip(attr, prop.GetValue(defaults));
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
            ToolTipService.SetToolTip(slider, tip);
            ToolTipService.SetToolTip(head, tip);
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
            ToolTipService.SetToolTip(toggle, tip);
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
            ToolTipService.SetToolTip(combo, tip);
            panel.Children.Add(combo);
        }
        else if (t == typeof(string) && attr.Choices is { Length: > 0 })
        {
            var current = prop.GetValue(settings)?.ToString() ?? "";
            var items = ModelChoices(attr.Choices, current);
            var combo = new ComboBox
            {
                Header = attr.Label,
                ItemsSource = items,
                SelectedIndex = Math.Max(0, items.IndexOf(current)),
                HorizontalAlignment = HorizontalAlignment.Stretch,
            };
            combo.SelectionChanged += (_, _) =>
            {
                if (combo.SelectedItem is string s && s != current)
                {
                    prop.SetValue(settings, s);
                    current = s;
                    changed();
                }
            };
            ToolTipService.SetToolTip(combo, tip);
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
            ToolTipService.SetToolTip(box, tip);
            panel.Children.Add(box);
        }
        return panel;
    }

    /// <summary>Declared versions first (v1, v2.1), then every exported model found in /models.</summary>
    private static List<string> ModelChoices(IEnumerable<string> declared, string? current)
    {
        var items = new List<string>(declared);
        void Add(string? name)
        {
            if (!string.IsNullOrWhiteSpace(name) && !items.Contains(name)) items.Add(name);
        }
        try
        {
            if (Directory.Exists(AvisPaths.Models))
            {
                var found = Directory.GetDirectories(AvisPaths.Models)
                    .Where(d => File.Exists(Path.Combine(d, "model.onnx")))
                    .Select(d => Path.GetFileName(d))
                    .OrderBy(n => n, StringComparer.OrdinalIgnoreCase);
                foreach (var name in found) Add(name);
            }
        }
        catch
        {
            // A missing or unreadable models folder must not break the settings panel.
        }
        Add(current);
        return items;
    }

    /// <summary>Two-line tooltip: the short description, then the range/default hint.</summary>
    private static string Tip(SettingAttribute attr, object? def)
    {
        var sb = new StringBuilder(attr.Description);
        if (def is double d)
            sb.Append('\n').Append("Диапазон ").Append(Text(attr.Min)).Append('…').Append(Text(attr.Max))
              .Append(", шаг ").Append(Text(attr.Step)).Append(". По умолчанию: ").Append(Text(d)).Append('.');
        else if (def is bool b)
            sb.Append('\n').Append("По умолчанию: ").Append(b ? "включено" : "выключено").Append('.');
        else if (def is Enum)
            sb.Append('\n').Append("По умолчанию: ").Append(def).Append('.');
        else if (def is string s)
            sb.Append('\n').Append("По умолчанию: ").Append(s.Length == 0 ? "пусто" : s).Append('.');
        return sb.ToString();
    }

    public static string Format(double v) => v.ToString("0.###", CultureInfo.InvariantCulture);

    /// <summary>Number for UI text: follows the current culture like the slider labels do.</summary>
    private static string Text(double v) => v.ToString("0.###", CultureInfo.CurrentCulture);
}

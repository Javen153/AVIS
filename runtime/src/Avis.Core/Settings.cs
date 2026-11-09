using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Avis.Core;

[AttributeUsage(AttributeTargets.Property)]
public sealed class SettingAttribute : Attribute
{
    public string Group { get; }
    public string GroupLabel { get; }
    public string Label { get; }
    public string Description { get; }
    public double Min { get; set; } = 0;
    public double Max { get; set; } = 1;
    public double Step { get; set; } = 0.01;
    public SettingAttribute(string group, string groupLabel, string label, string description)
    {
        Group = group; GroupLabel = groupLabel; Label = label; Description = description;
    }
}

public enum ScaleMode { Log, Erb, Mel, Linear }
public enum YMode { Absolute, Partial, Salience }
public enum BrightnessSource { Loudness, Salience }
public enum ChordMode { Elongated, PerNote }
public enum HistoryMode { Off, Trails, Waterfall }
public enum ColorMode { Palette, Timbre }
public enum WindowMaterial { Mica, Acrylic }
public enum DebugView { Main, Spectro, Masks, Table, Embedding }
public enum InferenceProvider { Cpu, DirectML }

/// <summary>All user-tunable values. Visual settings never affect analysis.</summary>
public sealed class AppSettings
{
    // ---------------------------------------------------------------- capture
    [Setting("capture", "Захват", "Устройство", "Имя устройства вывода; пусто = устройство по умолчанию")]
    public string CaptureDevice { get; set; } = "";

    [Setting("capture", "Захват", "Усиление, дБ", "Цифровое усиление входа", Min = -24, Max = 24, Step = 0.5)]
    public double GainDb { get; set; } = 0;

    // --------------------------------------------------------------- analysis
    [Setting("analysis", "Анализ", "Шкала X", "Отображение частотной оси")]
    public ScaleMode Scale { get; set; } = ScaleMode.Log;

    [Setting("analysis", "Анализ", "Нижняя частота, Гц", "Левая граница оси", Min = 20, Max = 200, Step = 1)]
    public double FMin { get; set; } = 30;

    [Setting("analysis", "Анализ", "Верхняя частота, Гц", "Правая граница оси", Min = 4000, Max = 20000, Step = 100)]
    public double FMax { get; set; } = 16000;

    // ---------------------------------------------------------------- neural
    [Setting("neural", "Нейросеть", "Модель", "Папка модели в /models")]
    public string ModelName { get; set; } = "v1";

    [Setting("neural", "Нейросеть", "Провайдер", "CPU надёжнее; DirectML быстрее на GPU")]
    public InferenceProvider Provider { get; set; } = InferenceProvider.Cpu;

    [Setting("neural", "Нейросеть", "Порог активности", "Минимальная уверенность полосы", Min = 0.1, Max = 0.95, Step = 0.01)]
    public double ActThreshold { get; set; } = 0.5;

    [Setting("neural", "Нейросеть", "Макс. объектов", "Ограничение числа одновременных объектов", Min = 1, Max = 24, Step = 1)]
    public double MaxObjects { get; set; } = 24;

    // --------------------------------------------------------------- tracking
    [Setting("tracking", "Трекинг", "Порог привязки", "cos(emb, centroid) для существующего объекта", Min = 0.3, Max = 0.95, Step = 0.01)]
    public double MatchThreshold { get; set; } = 0.60;

    [Setting("tracking", "Трекинг", "Порог лидера", "cos для объединения полос в кандидата", Min = 0.3, Max = 0.95, Step = 0.01)]
    public double LeaderThreshold { get; set; } = 0.70;

    [Setting("tracking", "Трекинг", "Порог банка", "cos для возврата старого ID", Min = 0.3, Max = 0.99, Step = 0.01)]
    public double BankThreshold { get; set; } = 0.75;

    [Setting("tracking", "Трекинг", "Кадров до рождения", "Сколько кадров кандидат должен прожить", Min = 1, Max = 30, Step = 1)]
    public double BirthFrames { get; set; } = 5;

    [Setting("tracking", "Трекинг", "Кадров до затухания", "Нет полос — объект fading", Min = 2, Max = 60, Step = 1)]
    public double FadeFrames { get; set; } = 15;

    [Setting("tracking", "Трекинг", "Кадров до удаления", "Скрыть fading-объект", Min = 10, Max = 200, Step = 1)]
    public double RemoveFrames { get; set; } = 60;

    [Setting("tracking", "Трекинг", "Порог слияния", "cos между центроидами", Min = 0.5, Max = 0.99, Step = 0.01)]
    public double MergeThreshold { get; set; } = 0.85;

    [Setting("tracking", "Трекинг", "Кадров для слияния", "Сколько кадров близки центроиды", Min = 10, Max = 300, Step = 1)]
    public double MergeFrames { get; set; } = 50;

    [Setting("tracking", "Трекинг", "Память банка, с", "Сколько помнить исчезнувшие объекты", Min = 10, Max = 300, Step = 1)]
    public double BankSeconds { get; set; } = 60;

    [Setting("tracking", "Трекинг", "α центроида", "Скорость обновления центроида", Min = 0.005, Max = 0.3, Step = 0.005)]
    public double CentroidAlpha { get; set; } = 0.05;

    [Setting("tracking", "Трекинг", "α отпечатка", "Скорость обновления банка", Min = 0.0005, Max = 0.05, Step = 0.0005)]
    public double FingerprintAlpha { get; set; } = 0.005;

    [Setting("tracking", "Трекинг", "τ softmax", "Мягкость принадлежности полос", Min = 0.02, Max = 0.5, Step = 0.01)]
    public double SoftmaxTau { get; set; } = 0.10;

    // --------------------------------------------------------------- loudness
    [Setting("loudness", "Громкость", "Режим Y", "Что показывает вертикаль")]
    public YMode YMode { get; set; } = YMode.Absolute;

    [Setting("loudness", "Громкость", "Нижний дБ", "Низ экрана", Min = -90, Max = -30, Step = 1)]
    public double DbMin { get; set; } = -70;

    [Setting("loudness", "Громкость", "Верхний дБ", "Верх экрана", Min = -20, Max = 6, Step = 1)]
    public double DbMax { get; set; } = 0;

    [Setting("loudness", "Громкость", "Гамма яркости", "Больше — тихие объекты ярче", Min = 0.4, Max = 2.5, Step = 0.05)]
    public double Gamma { get; set; } = 1.0;

    [Setting("loudness", "Громкость", "Яркость от", "Громкость или заметность")]
    public BrightnessSource Brightness { get; set; } = BrightnessSource.Loudness;

    // ----------------------------------------------------------------- visual
    [Setting("visual", "Визуал", "Размер пятна", "Базовый радиус сплэта", Min = 0.35, Max = 4, Step = 0.05)]
    public double SplatSize { get; set; } = 1.0;

    [Setting("visual", "Визуал", "Мягкость", "Мягкость края", Min = 0.2, Max = 4, Step = 0.05)]
    public double SplatSoftness { get; set; } = 1.0;

    [Setting("visual", "Визуал", "Влияние тональности", "Уже пятно у тональных звуков", Min = 0, Max = 1, Step = 0.01)]
    public double TonalityInfluence { get; set; } = 0.8;

    [Setting("visual", "Визуал", "Bloom", "Свечение ярких мест", Min = 0, Max = 2, Step = 0.05)]
    public double Bloom { get; set; } = 0.6;

    [Setting("visual", "Визуал", "Сглаживание", "Жёсткость пружины", Min = 4, Max = 60, Step = 1)]
    public double Smoothing { get; set; } = 18;

    [Setting("visual", "Визуал", "Атака, с", "Скорость появления", Min = 0.005, Max = 0.3, Step = 0.005)]
    public double Attack { get; set; } = 0.02;

    [Setting("visual", "Визуал", "Затухание, с", "Скорость исчезновения", Min = 0.02, Max = 2, Step = 0.01)]
    public double Release { get; set; } = 0.25;

    [Setting("visual", "Визуал", "Атака ударных, с", "Вспышка барабана", Min = 0.002, Max = 0.1, Step = 0.002)]
    public double DrumAttack { get; set; } = 0.006;

    [Setting("visual", "Визуал", "Затухание ударных, с", "Хвост барабана", Min = 0.02, Max = 2, Step = 0.01)]
    public double DrumRelease { get; set; } = 0.35;

    [Setting("visual", "Визуал", "Аккорды", "Одно вытянутое пятно или отдельные")]
    public ChordMode Chords { get; set; } = ChordMode.PerNote;

    [Setting("visual", "Визуал", "История", "Следы прошлого на экране")]
    public HistoryMode History { get; set; } = HistoryMode.Off;

    [Setting("visual", "Визуал", "Видимость шума", "Яркость шумовых объектов", Min = 0, Max = 1, Step = 0.01)]
    public double NoiseVisibility { get; set; } = 0.8;

    [Setting("visual", "Визуал", "Цвет", "Палитра или тембровый эмбеддинг")]
    public ColorMode Color { get; set; } = ColorMode.Palette;

    [Setting("visual", "Визуал", "Прозрачность", "Общая непрозрачность объектов", Min = 0.2, Max = 1, Step = 0.01)]
    public double Opacity { get; set; } = 1.0;

    // ----------------------------------------------------------------- window
    [Setting("window", "Окно", "Материал", "Mica или Acrylic")]
    public WindowMaterial Material { get; set; } = WindowMaterial.Mica;

    [Setting("window", "Окно", "Поверх всех", "Держать окно сверху")]
    public bool AlwaysOnTop { get; set; } = false;

    [Setting("window", "Окно", "Оверлей", "Без рамки, клик-сквозь, Ctrl+Shift+O")]
    public bool Overlay { get; set; } = false;

    // ------------------------------------------------------------------ debug
    [Setting("debug", "Отладка", "Вид", "Основной или отладочные виды")]
    public DebugView View { get; set; } = DebugView.Main;

    [Setting("debug", "Отладка", "Статистика", "FPS и задержка в углу")]
    public bool ShowStats { get; set; } = true;

    [Setting("debug", "Отладка", "Запись JSONL", "Писать кадры в artifacts")]
    public bool RecordJsonl { get; set; } = false;

    // ------------------------------------------------------------- plumbing
    [JsonIgnore] public string BaseDirectory { get; set; } = "";
    [JsonIgnore] public string ModelDirOverride { get; set; } = "";

    public string ToJson() => JsonSerializer.Serialize(this, JsonOpts);
    public static AppSettings FromJson(string json) =>
        JsonSerializer.Deserialize<AppSettings>(json, JsonOpts) ?? new AppSettings();

    public AppSettings Clone() => FromJson(ToJson());

    public static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter() },
    };

    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, ToJson());
    }

    public static AppSettings Load(string path)
    {
        if (!File.Exists(path)) return new AppSettings();
        try { return FromJson(File.ReadAllText(path)); }
        catch { return new AppSettings(); }
    }

    public static IEnumerable<(PropertyInfo Prop, SettingAttribute Attr)> Describe() =>
        typeof(AppSettings).GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Select(p => (p, p.GetCustomAttribute<SettingAttribute>()))
            .Where(x => x.Item2 != null)
            .Select(x => (x.p, x.Item2!));

    /// <summary>Apply "group.name=value" from CLI. Returns false if unknown.</summary>
    public bool SetByPath(string path, string value)
    {
        var parts = path.Split('.');
        if (parts.Length != 2) return false;
        foreach (var (prop, attr) in Describe())
        {
            if (!string.Equals(attr.Group, parts[0], StringComparison.OrdinalIgnoreCase)) continue;
            if (!string.Equals(prop.Name, parts[1], StringComparison.OrdinalIgnoreCase)) continue;
            var t = prop.PropertyType;
            object? v;
            if (t == typeof(double)) v = double.Parse(value, System.Globalization.CultureInfo.InvariantCulture);
            else if (t == typeof(bool)) v = bool.Parse(value);
            else if (t == typeof(string)) v = value;
            else if (t.IsEnum) v = Enum.Parse(t, value, true);
            else return false;
            prop.SetValue(this, v);
            return true;
        }
        return false;
    }

    public static Dictionary<string, AppSettings> BuiltInPresets()
    {
        var precise = new AppSettings
        {
            YMode = YMode.Absolute, SplatSize = 0.8, SplatSoftness = 0.8, Bloom = 0.3,
            History = HistoryMode.Off, NoiseVisibility = 0.6, Chords = ChordMode.PerNote,
            Smoothing = 24, Release = 0.18,
        };
        var beautiful = new AppSettings
        {
            YMode = YMode.Partial, SplatSize = 1.5, SplatSoftness = 1.6, Bloom = 1.1,
            History = HistoryMode.Trails, NoiseVisibility = 0.9, Chords = ChordMode.Elongated,
            Smoothing = 12, Release = 0.4,
        };
        var debug = new AppSettings
        {
            View = DebugView.Table, ShowStats = true, History = HistoryMode.Off, Bloom = 0.2,
        };
        return new Dictionary<string, AppSettings>(StringComparer.OrdinalIgnoreCase)
        {
            ["Точный"] = precise, ["Красивый"] = beautiful, ["Отладка"] = debug,
        };
    }
}

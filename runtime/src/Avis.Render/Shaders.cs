namespace Avis.Render;

/// <summary>HLSL sources compiled at build time with D3DCompile.</summary>
internal static class Shaders
{
    public const string Common = """
        cbuffer Uniforms : register(b0)
        {
            float2 viewport;
            float time;
            float mode;
        };
        SamplerState LinearSampler : register(s0);
        """;

    public const string Splat = Common + """
        struct Inst
        {
            float2 pos;
            float radius;
            float aspect;
            float4 color;
            float4 params; // x=softness, y=rotation
        };
        StructuredBuffer<Inst> Instances : register(t0);

        struct VSOut
        {
            float4 pos : SV_Position;
            float2 uv : TEXCOORD0;
            float4 col : COLOR0;
            float2 soft : TEXCOORD1;
        };

        VSOut VSMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
        {
            Inst inst = Instances[iid];
            float2 corner = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? 1.0 : -1.0);
            float c = cos(inst.params.y), s = sin(inst.params.y);
            float2 q = float2(corner.x * inst.radius, corner.y * inst.radius * inst.aspect);
            q = float2(c * q.x - s * q.y, s * q.x + c * q.y);
            VSOut o;
            o.pos = float4(inst.pos + q, 0.0, 1.0);
            o.uv = corner;
            o.col = inst.color;
            o.soft = float2(inst.params.x, 0.0);
            return o;
        }

        float4 PSMain(VSOut i) : SV_Target
        {
            float r2 = dot(i.uv, i.uv);
            if (r2 > 1.0) discard;
            float fade = smoothstep(1.0, 0.35, r2);
            float g = exp(-r2 * 3.2 * i.soft) * fade;
            return float4(i.col.rgb * i.col.a * g, i.col.a * g);
        }
        """;

    public const string Fullscreen = Common + """
        struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
        VSOut VSFull(uint vid : SV_VertexID)
        {
            float2 p = float2((vid == 1) ? 3.0 : -1.0, (vid == 2) ? 3.0 : -1.0);
            VSOut o;
            o.pos = float4(p, 0.0, 1.0);
            o.uv = float2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
            return o;
        }
        """;

    public const string Down = Fullscreen + """
        Texture2D Source : register(t0);
        float4 PSDown(VSOut i) : SV_Target
        {
            float2 texel = 1.0 / viewport;
            float3 c = Source.Sample(LinearSampler, i.uv + texel * float2(-0.5, -0.5)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(0.5, -0.5)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(-0.5, 0.5)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(0.5, 0.5)).rgb;
            return float4(c * 0.25, 1.0);
        }
        """;

    public const string Up = Fullscreen + """
        Texture2D Source : register(t0);
        float4 PSUP(VSOut i) : SV_Target
        {
            float2 texel = 1.0 / viewport;
            float3 c = Source.Sample(LinearSampler, i.uv).rgb * 4.0;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(-1, 0)).rgb * 2.0;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(1, 0)).rgb * 2.0;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(0, -1)).rgb * 2.0;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(0, 1)).rgb * 2.0;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(-1, -1)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(1, -1)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(-1, 1)).rgb;
            c += Source.Sample(LinearSampler, i.uv + texel * float2(1, 1)).rgb;
            return float4(c / 16.0, 1.0);
        }
        """;

    public const string Composite = Fullscreen + """
        Texture2D Hdr : register(t0);
        Texture2D BloomTex : register(t1);
        Texture2D Spectro : register(t2);

        float3 aces(float3 x)
        {
            const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
            return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
        }

        float4 PSComposite(VSOut i) : SV_Target
        {
            if (mode > 1.5)
            {
                float3 s = Spectro.Sample(LinearSampler, i.uv).rgb;
                return float4(s, saturate(max(s.r, max(s.g, s.b))));
            }
            float3 c = Hdr.Sample(LinearSampler, i.uv).rgb;
            if (mode > 0.5)
                return float4(c, saturate(max(c.r, max(c.g, c.b))));
            float3 b = BloomTex.Sample(LinearSampler, i.uv).rgb;
            c = aces(c + b);
            float a = saturate(max(c.r, max(c.g, c.b)));
            return float4(c * a, a);
        }
        """;

    public const string Spectro = Fullscreen + """
        StructuredBuffer<float> BandsDb : register(t0);
        StructuredBuffer<float4> Owners : register(t1);
        cbuffer SpectroU : register(b1)
        {
            float fminLog;
            float fmaxLog;
            float dbMin;
            float dbMax;
            int bandCount;
            int ownerCount;
            int showOwners;
            float pad;
        };

        float3 heat(float t)
        {
            t = saturate(t);
            return float3(smoothstep(0.25, 1.0, t),
                          smoothstep(0.0, 0.85, t) * 0.85,
                          smoothstep(0.05, 0.5, t) * (1.0 - t) + 0.05);
        }

        float4 PSSpectro(VSOut i) : SV_Target
        {
            float x = saturate(i.uv.x);
            float bar = saturate(1.0 - i.uv.y);
            float lf = x * (bandCount - 1);
            int b = (int)floor(lf);
            float frac = lf - b;
            b = clamp(b, 0, bandCount - 2);
            float v = lerp(BandsDb[b], BandsDb[b + 1], frac);
            float h = saturate((v - dbMin) / (dbMax - dbMin));
            float3 c = heat(h) * step(bar, h) * (0.65 + 0.35 * (1.0 - bar));
            if (showOwners > 0)
            {
                float4 o = Owners[b];
                float4 o2 = Owners[b + 1];
                float4 ow = lerp(o, o2, frac);
                if (ow.a > 0.5) c = lerp(c, ow.rgb, 0.5 * step(bar, h));
            }
            return float4(c, 1.0);
        }
        """;
}

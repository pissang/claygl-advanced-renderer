@export car.dof.coc

uniform sampler2D depth;

uniform float zNear = 0.1;
uniform float zFar = 2000;

uniform float focalDistance = 10;
// 50mm
uniform float focalLength = 50;
// f/5.6
uniform float aperture = 5.6;

uniform float maxCoc;

// Height of the 35mm full-frame format (36mm x 24mm)
// TODO: Should be set by a physical camera
uniform float _filmHeight = 0.024;

varying vec2 v_Texcoord;

@import clay.util.encode_float

void main()
{
    float z = texture2D(depth, v_Texcoord).r * 2.0 - 1.0;

    float dist = 2.0 * zNear * zFar / (zFar + zNear - z * (zFar - zNear));

    // From https://github.com/Unity-Technologies/PostProcessing
    float f = focalLength / 1000.0;
    float s1 = max(f, focalDistance);
    float coeff = f * f / (aperture * (s1 - f) * _filmHeight * 2.0);

    float coc = (dist - focalDistance) * coeff / max(dist, 1e-5);
    coc /= maxCoc;

    gl_FragColor = vec4(clamp(coc * 0.5 + 0.5, 0.0, 1.0), 0.0, 0.0, 1.0);
}
@end

@export car.dof.composite

#define DEBUG 0

uniform sampler2D sharpTex;
uniform sampler2D nearTex;
uniform sampler2D farTex;
uniform sampler2D cocTex;
uniform float maxCoc;
uniform float minCoc;

varying vec2 v_Texcoord;

@import clay.util.rgbm

void main()
{
    float coc = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
    vec4 nearTexel = decodeHDR(texture2D(nearTex, v_Texcoord));
    vec4 farTexel = decodeHDR(texture2D(farTex, v_Texcoord));
    vec4 sharpTexel = decodeHDR(texture2D(sharpTex, v_Texcoord));

    float nfa = clamp(nearTexel.a, 0.0, 1.0);

    // Convert CoC to far field alpha value.
    float ffa = smoothstep(minCoc / maxCoc, 0.2, coc);
    // ffa = smoothstep(0.0, 1.0, ffa);
    // Also weghted by alpha to avoid black pixels.
    ffa *= clamp(farTexel.a, 0.0, 1.0);
    gl_FragColor.rgb = mix(mix(sharpTexel.rgb, farTexel.rgb, ffa), nearTexel.rgb, nfa);

    gl_FragColor.a = max(max(sharpTexel.a, nfa), clamp(farTexel.a, 0.0, 1.0));

    // gl_FragColor.rgb = mix(sharpTexel.rgb, nearTexel.rgb, nfa);
    // gl_FragColor = vec4(vec3(nearTexel.a), 1.0);
    // gl_FragColor = nearTexel;
}

@end

@export car.dof.separate

uniform sampler2D mainTex;
uniform sampler2D cocTex;
uniform float minCoc;

varying vec2 v_Texcoord;

@import clay.util.rgbm

void main()
{
    vec4 color = decodeHDR(texture2D(mainTex, v_Texcoord));
    float coc = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
#ifdef FARFIELD
    color *= step(0.0, coc);
#else
    // Will have a dark halo on the edge after blurred if set whole color black.
    // Only set alpha to zero.
    color.a *= step(minCoc, -coc);
#endif

    gl_FragColor = encodeHDR(color);
}
@end


@export car.dof.dilateCoc

#define SHADER_NAME dilateCoc

uniform sampler2D cocTex;
uniform vec2 textureSize;

varying vec2 v_Texcoord;

void main()
{
#ifdef VERTICAL
    vec2 offset = vec2(0.0, 1.0 / textureSize.y);
#else
    vec2 offset = vec2(1.0 / textureSize.x, 0.0);
#endif

    float coc0 = 1.0;
    for (int i = 0; i < 17; i++) {
        vec2 duv = (float(i) - 8.0) * offset * 1.5;
        float coc = texture2D(cocTex, v_Texcoord + duv).r * 2.0 - 1.0;
        coc *= pow(1.0 - abs(float(i) - 8.0) / 10.0, 2.0);
        coc0 = min(coc0, coc);
    }
    gl_FragColor = vec4(coc0 * 0.5 + 0.5, 0.0, 0.0, 1.0);
}
@end



@export car.dof.blur
// https://www.shadertoy.com/view/Xd2BWc
// https://bartwronski.com/2017/08/06/separable-bokeh/
// https://www.ea.com/frostbite/news/circular-separable-convolution-depth-of-field

#define KERNEL_SIZE 17

// const vec4 Kernel0BracketsRealXY_ImZW = vec4(-0.038708,0.943062,-0.025574,0.660892);
const vec2 kernel1Weight = vec2(0.411259,-0.548794);

// const vec4 Kernel1BracketsRealXY_ImZW = vec4(0.000115,0.559524,0.000000,0.178226);
const vec2 kernel2Weight = vec2(0.513282,4.561110);

uniform vec4 kernel1[KERNEL_SIZE];
uniform vec4 kernel2[KERNEL_SIZE];

#ifdef FINAL_PASS
uniform sampler2D rTex;
uniform sampler2D gTex;
uniform sampler2D bTex;
uniform sampler2D aTex;
#endif
uniform sampler2D mainTex;

uniform sampler2D cocTex;
uniform sampler2D dilateCocTex;

uniform float maxCoc;
uniform float minCoc;
uniform vec2 textureSize;

varying vec2 v_Texcoord;

vec2 multComplex(vec2 p, vec2 q)
{
    return vec2(p.x*q.x-p.y*q.y, p.x*q.y+p.y*q.x);
}

float GetSmallestCoc(vec2 uv)
{
    vec2 k = 1.0 / textureSize;

    float coc = texture2D(cocTex, uv).r;

    vec4 around = vec4(
        texture2D(cocTex, uv - k).r,
        texture2D(cocTex, uv + vec2(k.x, -k.y)).r,
        texture2D(cocTex, uv + vec2(-k.x, k.y)).r,
        texture2D(cocTex, uv + k).r
    );

    return min(min(min(min(around.x, around.y), around.z), around.w), coc);
}

@import clay.util.rgbm
@import clay.util.float

void main()
{
    float halfKernelSize = float(KERNEL_SIZE / 2);

    vec2 texelSize = 1.0 / textureSize;

    float weight = 0.0;

#ifdef FARFIELD
    float coc0 = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
#else
    // Try to gathering the texel from nearfield in pixel of farfield.
    // To achieve bleeding from nearfield.
    float coc0 = -(texture2D(dilateCocTex, v_Texcoord).r * 2.0 - 1.0);
#endif
    if (coc0 <= 0.0) {
        // Write black color. DON'T discard because the color of previous pass won't be cleared.
        gl_FragColor = vec4(0.0);
        return;
    }
    coc0 *= maxCoc;

// TODO Nearfield use one component.

#ifdef FINAL_PASS
    vec4 valR = vec4(0.0);
    vec4 valG = vec4(0.0);
    vec4 valB = vec4(0.0);
    vec4 valA = vec4(0.0);

    vec2 offset = vec2(0.0, abs(coc0) / halfKernelSize);
#else
    vec4 val = vec4(0.0);

    vec2 offset = vec2(texelSize.x / texelSize.y * abs(coc0) / halfKernelSize, 0.0);
#endif

    for (int i = 0; i < KERNEL_SIZE; i++) {
        vec2 duv = (float(i) - halfKernelSize) * offset;
        float dist = length(duv);
        vec2 uv = clamp(v_Texcoord + duv, vec2(0.0), vec2(1.0));
#ifdef FARFIELD
        // Use the smallest coc around. remove coc aliasing.
        // Avoid gather the wrong pixel when source image is anti-alised.
        // Which will caused infocus pixels leak to the farfield.
        float coc = GetSmallestCoc(uv) * 2.0 - 1.0;
#else
        float coc = texture2D(cocTex, uv).r * 2.0 - 1.0;
#endif
        coc *= maxCoc;

        float w = 1.0;
#ifdef FARFIELD
        // Reject pixels in focus
        // w = step(minCoc, coc);
        // w *= smoothstep(0.0, 1.0, coc);

        // Gather as scatter. Reject pixels out of coc.
        // PENDING May have problem in separable filter, add a threshold.
        w *= step(dist, coc);
#endif
        weight += w;

        vec4 c0c1 = vec4(kernel1[i].xy, kernel2[i].xy);

#ifdef FINAL_PASS
        vec4 rTexel = texture2D(rTex, uv) * w;
        vec4 gTexel = texture2D(gTex, uv) * w;
        vec4 bTexel = texture2D(bTex, uv) * w;
        vec4 aTexel = texture2D(aTex, uv) * w;

        valR.xy += multComplex(rTexel.xy,c0c1.xy);
        valR.zw += multComplex(rTexel.zw,c0c1.zw);

        valG.xy += multComplex(gTexel.xy,c0c1.xy);
        valG.zw += multComplex(gTexel.zw,c0c1.zw);

        valB.xy += multComplex(bTexel.xy,c0c1.xy);
        valB.zw += multComplex(bTexel.zw,c0c1.zw);

        valA.xy += multComplex(aTexel.xy,c0c1.xy);
        valA.zw += multComplex(aTexel.zw,c0c1.zw);

#else
        vec4 color = texture2D(mainTex, uv);
        float tmp;
    #if defined(R_PASS)
        tmp = color.r;
    #elif defined(G_PASS)
        tmp = color.g;
    #elif defined(B_PASS)
        tmp = color.b;
    #elif defined(A_PASS)
        tmp = color.a;
    #endif
        val += tmp * c0c1 * w;
        // val.xy += tmp * c0c1.xy;
        // val.zw += tmp * c0c1.zw;
#endif
    }

    weight /= float(KERNEL_SIZE);
    weight = max(weight, 0.0001);

#ifdef FINAL_PASS
    valR /= weight;
    valG /= weight;
    valB /= weight;
    valA /= weight;
    float r = dot(valR.xy,kernel1Weight)+dot(valR.zw,kernel2Weight);
    float g = dot(valG.xy,kernel1Weight)+dot(valG.zw,kernel2Weight);
    float b = dot(valB.xy,kernel1Weight)+dot(valB.zw,kernel2Weight);
    float a = dot(valA.xy,kernel1Weight)+dot(valA.zw,kernel2Weight);
    gl_FragColor = vec4(r, g, b, a);
#else
    val /= weight;
    gl_FragColor = val;
#endif
}

@end


// @export car.dof.blurNearAlpha

// #define SHADER_NAME blurNearAlpha

// uniform sampler2D mainTex;
// uniform sampler2D cocTex;

// uniform float maxCoc;
// uniform vec2 textureSize;
// // 0.0 is horizontal, 1.0 is vertical
// uniform float blurDir;

// varying vec2 v_Texcoord;

// @import clay.util.rgbm
// @import clay.util.clamp_sample

// void main (void)
// {
//     @import clay.compositor.kernel.gaussian_13

//     float coc0 = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
//     coc0 *= -maxCoc;
//     coc0 = max(0.0, coc0);

//     vec2 texelSize = 1.0 / textureSize;
//     vec2 off = vec2(texelSize.x / texelSize.y * coc0 / 6.0, coc0 / 6.0);

//     off *= vec2(1.0 - blurDir, blurDir);

//     float sum = 0.0;
//     float weightAll = 0.0;

//     // blur in y (horizontal)
//     for (int i = 0; i < 13; i++) {
//         float w = gaussianKernel[i];
//         float a = texture2D(mainTex, v_Texcoord + float(i - 6) * off).a;
//         sum += a * w;
//         weightAll += w;
//     }
//     gl_FragColor = texture2D(mainTex, v_Texcoord);
//     gl_FragColor.a = sum / weightAll;
// }

// @end
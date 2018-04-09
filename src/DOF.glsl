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

@end

@export car.dof.composite

#define DEBUG 0

uniform sampler2D sharp;
uniform sampler2D blur;
uniform sampler2D cocTex;
uniform float maxCoc;

varying vec2 v_Texcoord;

@import clay.util.rgbm
@import clay.util.float

void main()
{
    float coc = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
    vec4 blurTexel = decodeHDR(texture2D(blur, v_Texcoord));
    vec4 sharpTexel = decodeHDR(texture2D(sharp, v_Texcoord));

    // float tmp = floor(blurTexel.a * 65535.0);
    // float alpha = floor(tmp / 256.0);
    // float nfa = (tmp - alpha * 256.0) / 255.0;
    // blurTexel.a = alpha / 255.0;
    float nfa = blurTexel.a;
    blurTexel.a = 1.0;

    // Convert CoC to far field alpha value.
    float ffa = smoothstep(0.0, 0.2, coc);
    // TODO
    gl_FragColor = mix(mix(sharpTexel, blurTexel, ffa), blurTexel, nfa);

    // gl_FragColor = vec4(vec3(abs(coc)), 1.0);
    // gl_FragColor = vec4(blurTexel.rgb, 1.0);
}

@end

@export car.dof.maxCoc

uniform sampler2D cocTex;
uniform vec2 textureSize;

varying vec2 v_Texcoord;

float tap(vec2 off) {
    return texture2D(cocTex, v_Texcoord + off).r * 2.0 - 1.0;
}

void main()
{
    vec4 d = vec4(-1.0, -1.0, +1.0, +1.0) / textureSize.xyxy;

    float coc = tap(vec2(0.0));
    float lt = tap(d.xy);
    float rt = tap(d.zy);
    float lb = tap(d.xw);
    float rb = tap(d.zw);

    coc = abs(lt) > abs(coc) ? lt : coc;
    coc = abs(rt) > abs(coc) ? rt : coc;
    coc = abs(lb) > abs(coc) ? lb : coc;
    coc = abs(rb) > abs(coc) ? rb : coc;

    gl_FragColor = vec4(coc * 0.5 + 0.5, 0.0,0.0,1.0);
}
@end



@export car.dof.diskBlur

#define POISSON_KERNEL_SIZE 16;

uniform sampler2D mainTex;
uniform sampler2D cocTex;
uniform sampler2D maxCocTex;

uniform float maxCoc;
uniform vec2 textureSize;

uniform vec2 poissonKernel[POISSON_KERNEL_SIZE];

uniform float percent;

varying vec2 v_Texcoord;

float nrand(const in vec2 n) {
    return fract(sin(dot(n.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

@import clay.util.rgbm
@import clay.util.float


void main()
{
    vec2 texelSize = 1.0 / textureSize;
    float maxCocInTile = abs(texture2D(maxCocTex, v_Texcoord).r * 2.0 - 1.0);
    vec2 offset = vec2(maxCoc * texelSize.x / texelSize.y, maxCoc) * maxCocInTile;

    float rnd = 6.28318 * nrand(v_Texcoord + 0.07 * percent);
    float cosa = cos(rnd);
    float sina = sin(rnd);
    vec4 basis = vec4(cosa, -sina, sina, cosa);

    vec4 fgColor = vec4(0.0);
    vec4 bgColor = vec4(0.0);

    float weightFg = 0.0;
    float weightBg = 0.0;

    float coc0 = texture2D(cocTex, v_Texcoord).r * 2.0 - 1.0;
    coc0 *= maxCoc;

    float margin = texelSize.y * 2.0;
    for (int i = 0; i < POISSON_KERNEL_SIZE; i++) {
        // TODO Use min/max tile
        vec2 duv = poissonKernel[i];
        duv = vec2(dot(duv, basis.xy), dot(duv, basis.zw));
        duv = offset * duv;
        float dist = length(duv);

        vec2 uv = v_Texcoord + duv;
        vec4 texel = decodeHDR(texture2D(mainTex, uv));
        float coc = texture2D(cocTex, uv).r * 2.0 - 1.0;
        coc *= maxCoc;

        // BG: Select the small coc to avoid color bleeding.
        float bgCoc = max(min(coc0, coc), 0.0);

        // Compare the CoC to the sample distance
        // Discard the pixels out of coc(scatter as gather). Add a small margin to smooth out.
        float bgw = clamp((bgCoc - dist + margin) / margin, 0.0, 1.0);
        float fgw = clamp((-coc  - dist + margin) / margin, 0.0, 1.0);

        // Cut influence from focused areas because they're darkened by CoC
        // premultiplying. This is only needed for near field.
        // fgw *= step(texelSize.y, -coc);

        bgColor += bgw * texel;
        fgColor += fgw * texel;

        weightFg += fgw;
        weightBg += bgw;
    }

    fgColor /= max(weightFg, 0.0001);
    bgColor /= max(weightBg, 0.0001);

    weightFg = clamp(weightFg * 3.1415 / float(POISSON_KERNEL_SIZE), 0.0, 1.0);

    gl_FragColor = encodeHDR(mix(bgColor, fgColor, weightFg));
    float alpha = clamp(gl_FragColor.a, 0.0, 1.0);
    alpha = floor(alpha * 255.0);

    gl_FragColor.a = (alpha * 256.0 + floor(weightFg * 255.0)) / 65535.0;
    gl_FragColor.a = weightFg;
}

@end
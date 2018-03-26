@export car.taa

// Modified from http://casual-effects.com/g3d/G3D10/data-files/shader/Film/Film_temporalAA.pix

#define SHADER_NAME TAA2

uniform sampler2D prevTex;
uniform sampler2D currTex;
uniform sampler2D velocityTex;

uniform vec2 texelSize;
uniform vec2 velocityTexelSize;

uniform vec2 jitterOffset;

// Between 0 to 1
// http://casual-effects.com/g3d/G3D10/build/manual/class_g3_d_1_1_temporal_filter_1_1_settings.html#aa3f2ea1bc948d770437aedfd1e921c32
uniform float hysteresis = 0.9;

uniform bool still;

varying vec2 v_Texcoord;

vec4 slideTowardsAABB(in vec4 oldColor, in vec4 newColor, in vec4 minimum, in vec4 maximum, in float maxVel) {
    if (all(greaterThanEqual(oldColor, minimum)) && all(lessThanEqual(oldColor, maximum))) {
        // In the bounding box, ok to use the old color
        return oldColor;
    }
    else {
        // Accelerate the history towards the new color.
        // Making the lerp value too large leaves ghosting.
        // Making it too small leaves aliasing. Two good tests
        // are the thin ridges in Holodeck and the sky in Sponza.
        float ghost = 0.4;// (maxVel < 10) ? 0.9 : 0.4;
        return mix(newColor, oldColor, ghost);
    }
}

void main () {

    if (still) {
        gl_FragColor = mix(texture2D(currTex, v_Texcoord), texture2D(prevTex, v_Texcoord), 0.9);
        return;
    }
    float sharpen = 0.01 * pow(hysteresis, 3.0);

    // Compute source neighborhood statistics
    vec4 source = texture2D(currTex, v_Texcoord);
    vec4 motionTexel = texture2D(velocityTex, v_Texcoord - jitterOffset);
    vec2 motion = motionTexel.rg - 0.5;
    // Remove pixels moved too far.
    if (length(motion) > 0.5 || motionTexel.a < 0.1) {
        gl_FragColor = source;
        return;
    }

    // Compute source neighborhood statistics
    vec4 localMin = source, localMax = source;
    float maxVel = dot(motion, motion);

    for (int y = -1; y <= +1; ++y) {
        for (int x = -1; x <= +1; ++x) {
            vec2 off = vec2(float(x), float(y));
            vec4 c = texture2D(currTex, v_Texcoord + off * texelSize);
            localMin = min(localMin, c);
            localMax = max(localMax, c);

            vec4 mTexel = texture2D(velocityTex, v_Texcoord + off * velocityTexelSize);
            vec2 m = mTexel.xy - 0.5;
            if (length(m) > 0.5 || mTexel.a < 0.1) {
                continue;
            }
            maxVel = max(dot(m, m), maxVel);
        }
    }
    vec4 history = texture2D(prevTex, v_Texcoord - motion);

    if (sharpen > 0.0) {
        history =
            history * (1.0 + sharpen) -
            (texture2D(prevTex, v_Texcoord + texelSize) +
             texture2D(prevTex, v_Texcoord + vec2(-1.0,1.0) * texelSize) +
             texture2D(prevTex, v_Texcoord + vec2(1.0,-1.0) * texelSize) +
             texture2D(prevTex, v_Texcoord + -texelSize)) * (sharpen * 0.25);
    }
    history = slideTowardsAABB(history, source, localMin, localMax, maxVel);
    // Back off hysteresis when under significant motion
    gl_FragColor = mix(source, history, hysteresis * clamp(1.0 - length(motion) * 0.2, 0.85, 1.0));
}
@end
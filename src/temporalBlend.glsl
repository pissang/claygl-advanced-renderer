@export car.temporalBlend

uniform sampler2D prevTex;
uniform sampler2D currTex;
uniform sampler2D velocityTex;

uniform float stillBlending = 0.95;
uniform float motionBlending = 0.5;

varying vec2 v_Texcoord;

void main() {
    vec4 vel = texture2D(velocityTex, v_Texcoord);
    vec2 motion = vel.rg - 0.5;
    vec4 curr = texture2D(currTex, v_Texcoord);
    vec4 prev = texture2D(prevTex, v_Texcoord - motion);
    if (vel.a < 0.01) {
        gl_FragColor = curr;
    }
    else {
        float motionLength = length(motion);
        // TODO velocity weighting.
        float weight = clamp(
            mix(stillBlending, motionBlending, motionLength * 1000.0),
            motionBlending, stillBlending
        );

        gl_FragColor = mix(curr, prev, weight);
    }
}

@end
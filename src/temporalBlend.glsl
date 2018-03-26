@export car.temporalBlend

uniform sampler2D prevTex;
uniform sampler2D currTex;
uniform sampler2D velocityTex;

uniform float weight = 0.1;

varying vec2 v_Texcoord;

void main() {
    vec4 vel = texture2D(velocityTex, v_Texcoord);
    vec4 curr = texture2D(currTex, v_Texcoord);
    vec4 prev = texture2D(prevTex, v_Texcoord - vel.rg + 0.5);
    if (length(vel.rg - 0.5) > 0.1 || vel.a < 0.01) {
        gl_FragColor = curr;
    }
    else {
        gl_FragColor = mix(prev, curr, weight);
    }
}

@end
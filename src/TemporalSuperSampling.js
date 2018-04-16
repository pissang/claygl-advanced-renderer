// Temporal Super Sample for static Scene
import { compositor, FrameBuffer, Texture2D, Texture, Shader, Matrix4 } from 'claygl';

var Pass = compositor.Pass;

import halton from './halton';

import TAAGLSLCode from './TAA3.glsl.js';

Shader.import(TAAGLSLCode);

function TemporalSuperSampling (opt) {
    opt = opt || {};
    var haltonSequence = [];

    for (var i = 0; i < 30; i++) {
        haltonSequence.push([
            halton(i, 2), halton(i, 3)
        ]);
    }

    this._haltonSequence = haltonSequence;

    this._frame = 0;

    // Frame texture before temporal supersampling
    this._prevFrameTex = new Texture2D({
        type: Texture.HALF_FLOAT
    });
    this._outputTex = new Texture2D({
        type: Texture.HALF_FLOAT
    });

    this._taaPass = new Pass({
        fragment: Shader.source('car.taa')
    });

    this._velocityTex = opt.velocityTexture;

    this._depthTex = opt.depthTexture;

    this._taaFb = new FrameBuffer({
        depthBuffer: false
    });

    this._outputPass = new Pass({
        fragment: Shader.source('clay.compositor.output'),
        // TODO, alpha is premultiplied?
        blendWithPrevious: true
    });
    this._outputPass.material.define('fragment', 'OUTPUT_ALPHA');
    this._outputPass.material.blend = function (_gl) {
        // FIXME.
        // Output is premultiplied alpha when BLEND is enabled ?
        // http://stackoverflow.com/questions/2171085/opengl-blending-with-previous-contents-of-framebuffer
        _gl.blendEquationSeparate(_gl.FUNC_ADD, _gl.FUNC_ADD);
        _gl.blendFuncSeparate(_gl.ONE, _gl.ONE_MINUS_SRC_ALPHA, _gl.ONE, _gl.ONE_MINUS_SRC_ALPHA);
    };
}

TemporalSuperSampling.prototype = {

    constructor: TemporalSuperSampling,

    /**
     * Jitter camera projectionMatrix
     * @parma {clay.Renderer} renderer
     * @param {clay.Camera} camera
     */
    jitterProjection: function (renderer, camera) {
        var offset = this._haltonSequence[this._frame % this._haltonSequence.length];
        var viewport = renderer.viewport;
        var dpr = viewport.devicePixelRatio || renderer.getDevicePixelRatio();
        var width = viewport.width * dpr;
        var height = viewport.height * dpr;

        var translationMat = new Matrix4();
        translationMat.array[12] = (offset[0] * 2.0 - 1.0) / width;
        translationMat.array[13] = (offset[1] * 2.0 - 1.0) / height;

        Matrix4.mul(camera.projectionMatrix, translationMat, camera.projectionMatrix);

        Matrix4.invert(camera.invProjectionMatrix, camera.projectionMatrix);
    },

    getJitterOffset: function (renderer) {
        var offset = this._haltonSequence[this._frame % this._haltonSequence.length];
        var viewport = renderer.viewport;
        var dpr = viewport.devicePixelRatio || renderer.getDevicePixelRatio();
        var width = viewport.width * dpr;
        var height = viewport.height * dpr;

        return [
            offset[0] / width,
            offset[1] / height
        ];
    },

    /**
     * Reset accumulating frame
     */
    resetFrame: function () {
        this._frame = 0;
    },

    /**
     * Return current frame
     */
    getFrame: function () {
        return this._frame;
    },

    getTargetTexture: function () {
        return this._prevFrameTex;
    },

    resize: function (width, height) {
        this._prevFrameTex.width = width;
        this._prevFrameTex.height = height;

        this._outputTex.width = width;
        this._outputTex.height = height;
    },

    isFinished: function () {
        return this._frame >= this._haltonSequence.length;
    },

    render: function (renderer, camera, sourceTex, still, output) {
        var taaPass = this._taaPass;

        taaPass.setUniform('jitterOffset', this.getJitterOffset(renderer));
        taaPass.setUniform('velocityTex', this._velocityTex);
        taaPass.setUniform('prevTex', this._prevFrameTex);
        taaPass.setUniform('currTex', sourceTex);
        taaPass.setUniform('depthTex', this._depthTex);
        taaPass.setUniform('texelSize', [1 / sourceTex.width, 1 / sourceTex.height]);
        taaPass.setUniform('velocityTexelSize', [1 / this._depthTex.width, 1 / this._depthTex.height]);

        taaPass.setUniform('still', !!still);

        this._taaFb.attach(this._outputTex);
        this._taaFb.bind(renderer);
        taaPass.render(renderer);
        this._taaFb.unbind(renderer);

        if (output) {
            this._outputPass.setUniform('texture', this._outputTex);
            this._outputPass.render(renderer);
        }

        // Swap texture
        var tmp = this._prevFrameTex;
        this._prevFrameTex = this._outputTex;
        this._outputTex = tmp;

        this._frame++;
    },

    dispose: function (renderer) {
        this._taaFb.dispose(renderer);
        this._prevFrameTex.dispose(renderer);
        this._outputTex.dispose(renderer);
        this._outputPass.dispose(renderer);
        this._taaPass.dispose(renderer);
    }
};

export default TemporalSuperSampling;
// Temporal Super Sample for static Scene
import { compositor, FrameBuffer, Texture2D, Shader, Matrix4 } from 'claygl';

var Pass = compositor.Pass;

import halton from './halton';

import TAAGLSLCode from './TAA.glsl.js';

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

    this._sourceTex = new Texture2D();
    this._sourceFb = new FrameBuffer();
    this._sourceFb.attach(this._sourceTex);

    // Frame texture before temporal supersampling
    this._prevFrameTex = new Texture2D();
    this._outputTex = new Texture2D();

    var taaPass = this._taaPass = new Pass({
        fragment: Shader.source('car.taa')
    });
    taaPass.setUniform('velocityTex', opt.velocityTexture);
    taaPass.setUniform('depthTex', opt.depthTexture);

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
        var viewport = renderer.viewport;
        var dpr = viewport.devicePixelRatio || renderer.getDevicePixelRatio();
        var width = viewport.width * dpr;
        var height = viewport.height * dpr;

        var offset = this._haltonSequence[this._frame % this._haltonSequence.length];

        var translationMat = new Matrix4();
        translationMat.array[12] = (offset[0] * 2.0 - 1.0) / width;
        translationMat.array[13] = (offset[1] * 2.0 - 1.0) / height;

        Matrix4.mul(camera.projectionMatrix, translationMat, camera.projectionMatrix);

        Matrix4.invert(camera.invProjectionMatrix, camera.projectionMatrix);
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

    /**
     * Get source framebuffer for usage
     */
    getSourceFrameBuffer: function () {
        return this._sourceFb;
    },

    resize: function (width, height) {
        if (this._sourceTex.width !== width || this._sourceTex.height !== height) {

            this._prevFrameTex.width = width;
            this._prevFrameTex.height = height;

            this._outputTex.width = width;
            this._outputTex.height = height;

            this._sourceTex.width = width;
            this._sourceTex.height = height;

            this._prevFrameTex.dirty();
            this._outputTex.dirty();
            this._sourceTex.dirty();
        }
    },

    isFinished: function () {
        return this._frame >= this._haltonSequence.length;
    },

    render: function (renderer, camera, still) {
        var taaPass = this._taaPass;
        // if (this._frame === 0) {
        //     // Direct output
        //     taaPass.setUniform('weight1', 0);
        //     taaPass.setUniform('weight2', 1);
        // }
        // else {
        // taaPass.setUniform('weight1', 0.9);
        // taaPass.setUniform('weight2', 0.1);
        // }
        taaPass.setUniform('prevTex', this._prevFrameTex);
        taaPass.setUniform('currTex', this._sourceTex);
        taaPass.setUniform('texelSize', [1 / this._sourceTex.width, 1 / this._sourceTex.height]);
        taaPass.setUniform('depthTexelSize', [1 / this._depthTex.width, 1 / this._depthTex.height]);
        taaPass.setUniform('sinTime', Math.sin(+(new Date()) / 8));
        taaPass.setUniform('projection', camera.projectionMatrix.array);

        taaPass.setUniform('still', !!still);

        this._taaFb.attach(this._outputTex);
        this._taaFb.bind(renderer);
        taaPass.render(renderer);
        this._taaFb.unbind(renderer);

        this._outputPass.setUniform('texture', this._outputTex);
        this._outputPass.render(renderer);

        // Swap texture
        var tmp = this._prevFrameTex;
        this._prevFrameTex = this._outputTex;
        this._outputTex = tmp;

        this._frame++;
    },

    dispose: function (renderer) {
        this._sourceFb.dispose(renderer);
        this._taaFb.dispose(renderer);
        this._prevFrameTex.dispose(renderer);
        this._outputTex.dispose(renderer);
        this._sourceTex.dispose(renderer);
        this._outputPass.dispose(renderer);
        this._taaPass.dispose(renderer);
    }
};

export default TemporalSuperSampling;
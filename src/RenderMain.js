import { prePass, Vector2, FrameBuffer, Texture2D, Texture } from 'claygl';

var ShadowMapPass = prePass.ShadowMap;

import EffectCompositor from './EffectCompositor';
import TemporalSuperSampling from './TemporalSuperSampling';
import halton from './halton';

function RenderMain(renderer, scene, enableShadow) {

    this.renderer = renderer;
    this.scene = scene;

    this.preZ = true;

    this._compositor = new EffectCompositor();

    this._temporalSS = new TemporalSuperSampling({
        velocityTexture: this._compositor.getVelocityTexture(),
        depthTexture: this._compositor.getDepthTexture()
    });

    if (enableShadow) {
        this._shadowMapPass = new ShadowMapPass({
            lightFrustumBias: 20
        });
    }

    this._enableTemporalSS = 'auto';

    scene.on('beforerender', function (renderer, scene, camera) {
        if (this.needsTemporalSS()) {
            this._temporalSS.jitterProjection(renderer, camera);
        }
    }, this);


    this._framebuffer = new FrameBuffer();
    this._sourceTex = new Texture2D({
        type: Texture.HALF_FLOAT
    });
    this._depthTex = new Texture2D({
        format: Texture.DEPTH_COMPONENT,
        type: Texture.UNSIGNED_INT
    });
}

/**
 * Cast a ray
 * @param {number} x offsetX
 * @param {number} y offsetY
 * @param {clay.math.Ray} out
 * @return {clay.math.Ray}
 */
var ndc = new Vector2();
RenderMain.prototype.castRay = function (x, y, out) {
    var renderer = this.layer.renderer;

    var oldViewport = renderer.viewport;
    renderer.viewport = this.viewport;
    renderer.screenToNDC(x, y, ndc);
    this.camera.castRay(ndc, out);
    renderer.viewport = oldViewport;

    return out;
};

/**
 * Prepare and update scene before render
 */
RenderMain.prototype.prepareRender = function () {
    var scene = this.scene;
    var camera = scene.getMainCamera();
    var renderer = this.renderer;

    camera.aspect = renderer.getViewportAspect();

    scene.update();
    scene.updateLights();
    var renderList = scene.updateRenderList(camera);

    this._updateSRGBOfList(renderList.opaque);
    this._updateSRGBOfList(renderList.transparent);

    this._frame = 0;
    if (!this._temporalSupportDynamic) {
        this._temporalSS.resetFrame();
    }

    // var lights = scene.getLights();
    // for (var i = 0; i < lights.length; i++) {
    //     if (lights[i].cubemap) {
    //         if (this._compositor && this._compositor.isSSREnabled()) {
    //             lights[i].invisible = true;
    //         }
    //         else {
    //             lights[i].invisible = false;
    //         }
    //     }
    // }

    if (this._enablePostEffect) {
        this._compositor.resize(renderer.getWidth(), renderer.getHeight(), renderer.getDevicePixelRatio());
    }
    if (this._temporalSS) {
        this._temporalSS.resize(renderer.getWidth(), renderer.getHeight(), renderer.getDevicePixelRatio());
    }
};

RenderMain.prototype.render = function (accumulating) {
    var scene = this.scene;
    var camera = scene.getMainCamera();
    this._doRender(scene, camera, accumulating, this._frame);
    this._frame++;
};

RenderMain.prototype.needsAccumulate = function () {
    return this.needsTemporalSS();
};

RenderMain.prototype.needsTemporalSS = function () {
    var enableTemporalSS = this._enableTemporalSS;
    if (enableTemporalSS === 'auto') {
        enableTemporalSS = this._enablePostEffect;
    }
    return enableTemporalSS;
};

RenderMain.prototype.hasDOF = function () {
    return this._enableDOF;
};

RenderMain.prototype.isAccumulateFinished = function () {
    var frame = this._frame;
    return !(this.needsTemporalSS() && !this._temporalSS.isFinished(frame))
        && !(this._compositor && !this._compositor.isSSAOFinished(frame))
        && !(this._compositor && !this._compositor.isSSRFinished(frame))
        && !(this._compositor && frame < 30);
};

RenderMain.prototype._doRender = function (scene, camera, accumulating, accumFrame) {

    var renderer = this.renderer;

    accumFrame = accumFrame || 0;

    if (!accumulating && this._shadowMapPass) {
        this._shadowMapPass.kernelPCF = this._pcfKernels[0];
        // Not render shadowmap pass in accumulating frame.
        this._shadowMapPass.render(renderer, scene, camera, true);
    }

    this._updateShadowPCFKernel(scene, camera, accumFrame);

    // Shadowmap will set clearColor.
    renderer.gl.clearColor(0.0, 0.0, 0.0, 0.0);

    if (this._enablePostEffect) {
        // normal render also needs to be jittered when have edge pass.
        if (this.needsTemporalSS()) {
            this._temporalSS.jitterProjection(renderer, camera);
        }
        this._compositor.updateGBuffer(renderer, scene, camera, this._temporalSS.getFrame());
    }

    // Always update SSAO to make sure have correct ssaoMap status
    // TODO TRANSPARENT OBJECTS.
    this._updateSSAO(renderer, scene, camera, accumulating ? this._temporalSS.getFrame() : 0);

    var frameBuffer;

    var needTemporalPass = this.needsTemporalSS() && (this._temporalSupportDynamic || accumulating);
    var needPostEffect = this._enablePostEffect;

    if (!needTemporalPass && !needPostEffect) {
        renderer.render(scene, camera, true, this.preZ);
        this.afterRenderScene(renderer, scene, camera);
    }
    else {
        var isSSREnabled = this._compositor.isSSREnabled();

        var sourceTex = this._sourceTex;
        var depthTex = this._depthTex;
        var frameBuffer = this._framebuffer;
        var dpr = renderer.getDevicePixelRatio();
        depthTex.width = sourceTex.width = renderer.getWidth() * dpr;
        depthTex.height = sourceTex.height = renderer.getHeight() * dpr;

        frameBuffer.attach(sourceTex);
        frameBuffer.attach(depthTex, FrameBuffer.DEPTH_ATTACHMENT);
        frameBuffer.bind(renderer);
        renderer.gl.clear(renderer.gl.DEPTH_BUFFER_BIT | renderer.gl.COLOR_BUFFER_BIT);
        renderer.render(scene, camera, true, this.preZ);
        this.afterRenderScene(renderer, scene, camera);
        frameBuffer.unbind(renderer);

        if (isSSREnabled && needPostEffect) {
            this._compositor.updateSSR(
                renderer, scene, camera,
                sourceTex,
                // TODO reprojection
                sourceTex,
                // needTemporalPass ? this._temporalSS.getTargetTexture() : sourceTex,
                this._temporalSS.getFrame()
            );
            sourceTex = this._compositor.getSSRTexture();
        }

        if (needTemporalPass) {
            var directOutput = !needPostEffect;
            this._temporalSS.render(renderer, camera, sourceTex, accumulating, directOutput);
            sourceTex = this._temporalSS.getTargetTexture();
        }
        if (needPostEffect) {
            this._compositor.composite(
                renderer, scene, camera, sourceTex, depthTex,
                needTemporalPass ? this._temporalSS.getFrame() : 0,
                accumulating
            );
        }
    }

    this.afterRenderAll(renderer, scene, camera);
};

RenderMain.prototype._updateSRGBOfList = function (list) {
    var isLinearSpace = this.isLinearSpace();
    for (var i = 0; i < list.length; i++) {
        list[i].material[isLinearSpace ? 'define' : 'undefine']('fragment', 'SRGB_DECODE');
    };
};

RenderMain.prototype.afterRenderScene = function (renderer, scene, camera) {};
RenderMain.prototype.afterRenderAll = function (renderer, scene, camera) {};

RenderMain.prototype._updateSSAO = function (renderer, scene, camera, frame) {
    var ifEnableSSAO = this._enableSSAO && this._enablePostEffect;
    var compositor = this._compositor;
    if (ifEnableSSAO) {
        this._compositor.updateSSAO(renderer, scene, camera, this._temporalSS.getFrame());
    }

    function updateQueue(queue) {
        for (var i = 0; i < queue.length; i++) {
            var renderable = queue[i];
            renderable.material[ifEnableSSAO ? 'enableTexture' : 'disableTexture']('ssaoMap');
            if (ifEnableSSAO) {
                renderable.material.set('ssaoMap', compositor.getSSAOTexture());
            }
        }
    }
    updateQueue(scene.getRenderList(camera).opaque);
    updateQueue(scene.getRenderList(camera).transparent);
};

RenderMain.prototype._updateShadowPCFKernel = function (scene, camera, frame) {
    var pcfKernel = this._pcfKernels[frame % this._pcfKernels.length];
    function updateQueue(queue) {
        for (var i = 0; i < queue.length; i++) {
            if (queue[i].receiveShadow) {
                queue[i].material.set('pcfKernel', pcfKernel);
                if (queue[i].material) {
                    queue[i].material.define('fragment', 'PCF_KERNEL_SIZE', pcfKernel.length / 2);
                }
            }
        }
    }
    updateQueue(scene.getRenderList(camera).opaque);
    updateQueue(scene.getRenderList(camera).transparent);
};

RenderMain.prototype.dispose = function () {
    var renderer = this.renderer;
    this._compositor.dispose(renderer);
    this._temporalSS.dispose(renderer);
    if (this._shadowMapPass) {
        this._shadowMapPass.dispose(renderer);
    }
    renderer.dispose();
};

RenderMain.prototype.setPostEffect = function (opts, api) {
    var compositor = this._compositor;
    opts = opts || {};
    this._enablePostEffect = !!opts.enable;
    var bloomOpts = opts.bloom || {};
    var edgeOpts = opts.edge || {};
    var dofOpts = opts.depthOfField || {};
    var ssaoOpts = opts.screenSpaceAmbientOcclusion || {};
    var ssrOpts = opts.screenSpaceReflection || {};
    var fxaaOpts = opts.FXAA || {};
    var colorCorrOpts = opts.colorCorrection || {};
    bloomOpts.enable ? compositor.enableBloom() : compositor.disableBloom();
    dofOpts.enable ? compositor.enableDOF() : compositor.disableDOF();
    ssrOpts.enable ? compositor.enableSSR() : compositor.disableSSR();
    colorCorrOpts.enable ? compositor.enableColorCorrection() : compositor.disableColorCorrection();
    edgeOpts.enable ? compositor.enableEdge() : compositor.disableEdge();
    fxaaOpts.enable ? compositor.enableFXAA() : compositor.disableFXAA();

    this._enableDOF = dofOpts.enable;
    this._enableSSAO = ssaoOpts.enable;

    this._enableSSAO ? compositor.enableSSAO() : compositor.disableSSAO();

    compositor.setBloomIntensity(bloomOpts.intensity);
    compositor.setEdgeColor(edgeOpts.color);
    compositor.setColorLookupTexture(colorCorrOpts.lookupTexture, api);
    compositor.setExposure(colorCorrOpts.exposure);

    ['radius', 'quality', 'intensity', 'temporalFilter'].forEach(function (name) {
        compositor.setSSAOParameter(name, ssaoOpts[name]);
    });
    ['quality', 'maxRoughness'].forEach(function (name) {
        compositor.setSSRParameter(name, ssrOpts[name]);
    });
    ['quality', 'focalDistance', 'focalRange', 'blurRadius', 'aperture'].forEach(function (name) {
        compositor.setDOFParameter(name, dofOpts[name]);
    });
    ['brightness', 'contrast', 'saturation'].forEach(function (name) {
        compositor.setColorCorrection(name, colorCorrOpts[name]);
    });
};

RenderMain.prototype.setShadow = function (opts) {
    var pcfKernels = [];
    var off = 0;
    for (var i = 0; i < 30; i++) {
        var pcfKernel = [];
        for (var k = 0; k < opts.kernelSize; k++) {
            pcfKernel.push((halton(off, 2) * 2.0 - 1.0) * opts.blurSize);
            pcfKernel.push((halton(off, 3) * 2.0 - 1.0) * opts.blurSize);
            off++;
        }
        pcfKernels.push(pcfKernel);
    }
    this._pcfKernels = pcfKernels;
};

RenderMain.prototype.isDOFEnabled = function () {
    return this._enablePostEffect && this._enableDOF;
};

RenderMain.prototype.setDOFFocusOnPoint = function (depth) {
    if (this._enablePostEffect) {

        if (depth > this.camera.far || depth < this.camera.near) {
            return;
        }

        this._compositor.setDOFParameter('focalDistance', depth);
        return true;
    }
};

RenderMain.prototype.setTemporalSuperSampling = function (temporalSuperSamplingOpt) {
    temporalSuperSamplingOpt = temporalSuperSamplingOpt || {};
    this._enableTemporalSS = temporalSuperSamplingOpt.enable;
    this._temporalSupportDynamic = temporalSuperSamplingOpt.dynamic;

    if (this._enableTemporalSS && this._temporalSupportDynamic) {
        this._compositor.enableVelocityBuffer();
    }
    else {
        this._compositor.disableVelocityBuffer();
    }
};

RenderMain.prototype.isLinearSpace = function () {
    return this._enablePostEffect;
};

export default RenderMain;
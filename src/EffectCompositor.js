import { Shader, Texture2D, Texture, FrameBuffer, createCompositor, deferred, shader } from 'claygl';

import SSAOPass from './SSAOPass';
import SSRPass from './SSRPass';
import circularSeparateKernel from './circularSeparateKernel';

import effectJson from './composite.js';

import dofCode from './DOF.glsl.js';
import temporalBlendCode from './temporalBlend.glsl.js';

var GBuffer = deferred.GBuffer;

Shader.import(dofCode);

Shader.import(temporalBlendCode);

var commonOutputs = {
    color: {
        parameters: {
            width: function (renderer) {
                return renderer.getWidth();
            },
            height: function (renderer) {
                return renderer.getHeight();
            }
        }
    }
};

var FINAL_NODES_CHAIN = ['composite', 'FXAA'];

function EffectCompositor() {

    this._gBufferPass = new GBuffer({
        renderTransparent: true,
        enableTargetTexture3: false,
        enableTargetTexture4: true
    });

    this._compositor = createCompositor(effectJson);

    var sourceNode = this._compositor.getNodeByName('source');
    var cocNode = this._compositor.getNodeByName('coc');

    this._sourceNode = sourceNode;
    this._cocNode = cocNode;
    this._compositeNode = this._compositor.getNodeByName('composite');
    this._fxaaNode = this._compositor.getNodeByName('FXAA');

    this._dofBlurNodes = [
        'dof_blur_far_1', 'dof_blur_far_2', 'dof_blur_far_3', 'dof_blur_far_4', 'dof_blur_far_final',
        'dof_blur_near_1', 'dof_blur_near_2', 'dof_blur_near_3', 'dof_blur_near_4', 'dof_blur_near_final'
    ].map(function (name) {
        return this._compositor.getNodeByName(name);
    }, this);

    this._dofFarFieldNode = this._compositor.getNodeByName('dof_separate_far');
    this._dofNearFieldNode = this._compositor.getNodeByName('dof_separate_near');
    this._dofCompositeNode = this._compositor.getNodeByName('dof_composite');

    this._dofBlurKernel = null;
    this._dofBlurKernelSize = new Float32Array(0);

    this._finalNodesChain = FINAL_NODES_CHAIN.map(function (name) {
        return this._compositor.getNodeByName(name);
    }, this);

    var gBufferObj = {
        normalTexture: this._gBufferPass.getTargetTexture1(),
        depthTexture: this._gBufferPass.getTargetTexture2(),
        albedoTexture: this._gBufferPass.getTargetTexture3(),
        velocityTexture: this._gBufferPass.getTargetTexture4()
    };
    this._ssaoPass = new SSAOPass(gBufferObj);
    this._ssrPass = new SSRPass(gBufferObj);
}

EffectCompositor.prototype.resize = function (width, height, dpr) {
    dpr = dpr || 1;
    width = width * dpr;
    height = height * dpr;
    this._gBufferPass.resize(width, height);
};

EffectCompositor.prototype._ifRenderNormalPass = function () {
    // return this._enableSSAO || this._enableEdge || this._enableSSR;
    return true;
};

EffectCompositor.prototype._getPrevNode = function (node) {
    var idx = FINAL_NODES_CHAIN.indexOf(node.name) - 1;
    var prevNode = this._finalNodesChain[idx];
    while (prevNode && !this._compositor.getNodeByName(prevNode.name)) {
        idx -= 1;
        prevNode = this._finalNodesChain[idx];
    }
    return prevNode;
};
EffectCompositor.prototype._getNextNode = function (node) {
    var idx = FINAL_NODES_CHAIN.indexOf(node.name) + 1;
    var nextNode = this._finalNodesChain[idx];
    while (nextNode && !this._compositor.getNodeByName(nextNode.name)) {
        idx += 1;
        nextNode = this._finalNodesChain[idx];
    }
    return nextNode;
};
EffectCompositor.prototype._addChainNode = function (node) {
    var prevNode = this._getPrevNode(node);
    var nextNode = this._getNextNode(node);
    if (!prevNode) {
        return;
    }

    prevNode.outputs = commonOutputs;
    node.inputs.texture = prevNode.name;
    if (nextNode) {
        node.outputs = commonOutputs;
        nextNode.inputs.texture = node.name;
    }
    else {
        node.outputs = null;
    }
    this._compositor.addNode(node);
};
EffectCompositor.prototype._removeChainNode = function (node) {
    var prevNode = this._getPrevNode(node);
    var nextNode = this._getNextNode(node);
    if (!prevNode) {
        return;
    }

    if (nextNode) {
        prevNode.outputs = commonOutputs;
        nextNode.inputs.texture = prevNode.name;
    }
    else {
        prevNode.outputs = null;
    }
    this._compositor.removeNode(node);
};
/**
 * Update normal
 */
EffectCompositor.prototype.updateGBuffer = function (renderer, scene, camera, frame) {
    if (this._ifRenderNormalPass()) {
        this._gBufferPass.update(renderer, scene, camera);
    }
};

/**
 * Render SSAO after render the scene, before compositing
 */
EffectCompositor.prototype.updateSSAO = function (renderer, scene, camera, frame) {
    this._ssaoPass.update(renderer, camera, frame);
};

EffectCompositor.prototype.updateSSR = function (renderer, scene, camera, sourceTexture, reflectionSourceTexture, frame) {
    this._ssrPass.setSSAOTexture(
        this._enableSSAO ? this._ssaoPass.getTargetTexture() : null
    );
    this._ssrPass.update(renderer, camera, sourceTexture, reflectionSourceTexture, frame);
};

/**
 * Enable SSAO effect
 */
EffectCompositor.prototype.enableSSAO = function () {
    this._enableSSAO = true;
};

/**
 * Disable SSAO effect
 */
EffectCompositor.prototype.disableSSAO = function () {
    this._enableSSAO = false;
};

EffectCompositor.prototype.enableVelocityBuffer = function () {
    this._gBufferPass.enableTargetTexture4 = true;
};
EffectCompositor.prototype.disableVelocityBuffer = function () {
    this._gBufferPass.enableTargetTexture4 = false;
};

/**
 * Enable SSR effect
 */
EffectCompositor.prototype.enableSSR = function () {
    this._enableSSR = true;
};
/**
 * Disable SSR effect
 */
EffectCompositor.prototype.disableSSR = function () {
    this._enableSSR = false;
};

/**
 * Render SSAO after render the scene, before compositing
 */
EffectCompositor.prototype.getSSAOTexture = function () {
    return this._ssaoPass.getTargetTexture();
};

EffectCompositor.prototype.getSSRTexture = function () {
    return this._ssrPass.getTargetTexture();
};


EffectCompositor.prototype.getVelocityTexture = function () {
    return this._gBufferPass.getTargetTexture4();
};
EffectCompositor.prototype.getDepthTexture = function () {
    return this._gBufferPass.getTargetTexture2();
};

/**
 * Disable fxaa effect
 */
EffectCompositor.prototype.disableFXAA = function () {
    this._removeChainNode(this._fxaaNode);
};

/**
 * Enable fxaa effect
 */
EffectCompositor.prototype.enableFXAA = function () {
    this._addChainNode(this._fxaaNode);
};

/**
 * Enable bloom effect
 */
EffectCompositor.prototype.enableBloom = function () {
    this._compositeNode.inputs.bloom = 'bloom_composite';
    this._compositor.dirty();
};

/**
 * Disable bloom effect
 */
EffectCompositor.prototype.disableBloom = function () {
    this._compositeNode.inputs.bloom = null;
    this._compositor.dirty();
};

/**
 * Enable depth of field effect
 */
EffectCompositor.prototype.enableDOF = function () {
    this._compositeNode.inputs.texture = 'dof_composite';
    this._compositor.dirty();
};
/**
 * Disable depth of field effect
 */
EffectCompositor.prototype.disableDOF = function () {
    this._compositeNode.inputs.texture = 'source';
    this._compositor.dirty();
};

/**
 * Enable color correction
 */
EffectCompositor.prototype.enableColorCorrection = function () {
    this._compositeNode.define('COLOR_CORRECTION');
    this._enableColorCorrection = true;
};
/**
 * Disable color correction
 */
EffectCompositor.prototype.disableColorCorrection = function () {
    this._compositeNode.undefine('COLOR_CORRECTION');
    this._enableColorCorrection = false;
};

/**
 * Enable edge detection
 */
EffectCompositor.prototype.enableEdge = function () {
    this._enableEdge = true;
};

/**
 * Disable edge detection
 */
EffectCompositor.prototype.disableEdge = function () {
    this._enableEdge = false;
};

/**
 * Set bloom intensity
 * @param {number} value
 */
EffectCompositor.prototype.setBloomIntensity = function (value) {
    if (value == null) {
        return;
    }
    this._compositeNode.setParameter('bloomIntensity', value);
};

EffectCompositor.prototype.setSSAOParameter = function (name, value) {
    if (value == null) {
        return;
    }
    switch (name) {
        case 'quality':
            // PENDING
            var kernelSize = ({
                low: 6,
                medium: 12,
                high: 32,
                ultra: 62
            })[value] || 12;
            this._ssaoPass.setParameter('kernelSize', kernelSize);
            break;
        case 'radius':
            this._ssaoPass.setParameter(name, value);
            this._ssaoPass.setParameter('bias', value / 50);
            break;
        case 'intensity':
        case 'temporalFilter':
            this._ssaoPass.setParameter(name, value);
            break;
    }
};

EffectCompositor.prototype.setDOFParameter = function (name, value) {
    if (value == null) {
        return;
    }
    switch (name) {
        case 'focalDistance':
        case 'focalRange':
        case 'aperture':
            this._cocNode.setParameter(name, value);
            break;
        case 'blurRadius':
            this._dofBlurRadius = value;
            break;
        // case 'quality':
        //     this._dofBlurKernel = poissonKernel[value] || poissonKernel.medium;
        //     var kernelSize = this._dofBlurKernel.length / 2;
        //     for (var i = 0; i < this._dofBlurNodes.length; i++) {
        //         this._dofBlurNodes[i].define('POISSON_KERNEL_SIZE', kernelSize);
        //     }
        //     break;
    }
};

EffectCompositor.prototype.setSSRParameter = function (name, value) {
    if (value == null) {
        return;
    }
    switch (name) {
        case 'quality':
            // PENDING
            var maxIteration = ({
                low: 10,
                medium: 15,
                high: 30,
                ultra: 80
            })[value] || 20;
            var pixelStride = ({
                low: 32,
                medium: 16,
                high: 8,
                ultra: 4
            })[value] || 16;
            this._ssrPass.setParameter('maxIteration', maxIteration);
            this._ssrPass.setParameter('pixelStride', pixelStride);
            break;
        case 'maxRoughness':
            this._ssrPass.setParameter('minGlossiness', Math.max(Math.min(1.0 - value, 1.0), 0.0));
            break;
        default:
            console.warn('Unkown SSR parameter ' + name);
    }
};

/**
 * Set color of edge
 */
EffectCompositor.prototype.setEdgeColor = function (value) {
    // if (value == null) {
    //     return;
    // }
    // this._edgePass.setParameter('edgeColor', value);
};

EffectCompositor.prototype.setExposure = function (value) {
    if (value == null) {
        return;
    }
    this._compositeNode.setParameter('exposure', Math.pow(2, value));
};

EffectCompositor.prototype.setColorLookupTexture = function (image, api) {
    // this._compositeNode.pass.material.setTextureImage('lut', this._enableColorCorrection ? image : 'none', api, {
    //     minFilter: Texture.NEAREST,
    //     magFilter: Texture.NEAREST,
    //     flipY: false
    // });
};
EffectCompositor.prototype.setColorCorrection = function (type, value) {
    this._compositeNode.setParameter(type, value);
};

EffectCompositor.prototype.composite = function (renderer, scene, camera, sourceTexture, depthTexture, frame) {
    this._sourceNode.texture = sourceTexture;

    this._cocNode.setParameter('depth', depthTexture);

    // var blurKernel = this._dofBlurKernel;

    var maxCoc = this._dofBlurRadius || 10;
    maxCoc /= renderer.getHeight();
    // var minCoc = 1 / renderer.getHeight();
    var minCoc = 0;
    // var jitter = Math.random();
    for (var i = 0; i < this._dofBlurNodes.length; i++) {
        var blurNode = this._dofBlurNodes[i];
        blurNode.setParameter('kernel1', circularSeparateKernel.component1);
        blurNode.setParameter('kernel2', circularSeparateKernel.component2);
        blurNode.setParameter('maxCoc', maxCoc);
        blurNode.setParameter('minCoc', minCoc);
    }
    this._cocNode.setParameter('maxCoc', maxCoc);
    this._dofCompositeNode.setParameter('maxCoc', maxCoc);
    this._dofCompositeNode.setParameter('minCoc', minCoc);
    this._dofFarFieldNode.setParameter('minCoc', minCoc / maxCoc);
    this._dofNearFieldNode.setParameter('minCoc', minCoc / maxCoc);

    this._cocNode.setParameter('zNear', camera.near);
    this._cocNode.setParameter('zFar', camera.far);

    this._compositor.render(renderer);
};

EffectCompositor.prototype.isSSRFinished = function (frame) {
    return this._ssrPass ? this._ssrPass.isFinished(frame) : true;
};

EffectCompositor.prototype.isSSAOFinished = function (frame) {
    return this._ssaoPass ? this._ssaoPass.isFinished(frame) : true;
};

EffectCompositor.prototype.isSSREnabled = function () {
    return this._enableSSR;
};

EffectCompositor.prototype.dispose = function (renderer) {
    this._compositor.dispose(renderer);

    this._gBufferPass.dispose(renderer);
    this._ssaoPass.dispose(renderer);
};

export default EffectCompositor;
import nodeResolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';

export default {
    input: __dirname + '/index.js',
    plugins: [
        nodeResolve(),
        commonjs()
    ],
    // sourceMap: true,
    output: {
        format: 'umd',
        name: 'ClayAdvancedRenderer',
        file: 'dist/claygl-advanced-renderer.js',
        globals: { 'claygl': 'clay' }
    },
    external: ['claygl']
};
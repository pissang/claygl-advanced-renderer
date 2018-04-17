var DOF_BLUR_OUTPUTS = {
    'color': {
        'parameters': {
            'width': 'expr(width / 2.0 * 1.0)',
            'height': 'expr(height / 2.0 * 1.0)',
            'type': 'HALF_FLOAT'
        }
    }
};

var DOF_BLUR_PARAMETERS = {
    'textureSize': 'expr( [width / 2.0 * 1.0, height / 2.0 * 1.0] )'
};

export default {
    'type' : 'compositor',
    'nodes' : [
        {
            'name': 'source',
            'type': 'texture',
            'outputs': {
                'color': {}
            }
        },
        {
            'name': 'source_half',
            'shader': '#source(clay.compositor.downsample)',
            'inputs': {
                'texture': 'source'
            },
            'outputs': {
                'color': {
                    'parameters': {
                        'width': 'expr(width * 1.0 / 2)',
                        'height': 'expr(height * 1.0 / 2)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'textureSize': 'expr( [width * 1.0, height * 1.0] )'
            }
        },


        {
            'name' : 'bright',
            'shader' : '#source(clay.compositor.bright)',
            'inputs' : {
                'texture' : 'source_half'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 2)',
                        'height' : 'expr(height * 1.0 / 2)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'threshold' : 2,
                'scale': 4,
                'textureSize': 'expr([width * 1.0 / 2, height / 2])'
            }
        },

        {
            'name': 'bright_downsample_4',
            'shader' : '#source(clay.compositor.downsample)',
            'inputs' : {
                'texture' : 'bright'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 4)',
                        'height' : 'expr(height * 1.0 / 4)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'textureSize': 'expr( [width * 1.0 / 2, height / 2] )'
            }
        },
        {
            'name': 'bright_downsample_8',
            'shader' : '#source(clay.compositor.downsample)',
            'inputs' : {
                'texture' : 'bright_downsample_4'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 8)',
                        'height' : 'expr(height * 1.0 / 8)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'textureSize': 'expr( [width * 1.0 / 4, height / 4] )'
            }
        },
        {
            'name': 'bright_downsample_16',
            'shader' : '#source(clay.compositor.downsample)',
            'inputs' : {
                'texture' : 'bright_downsample_8'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 16)',
                        'height' : 'expr(height * 1.0 / 16)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'textureSize': 'expr( [width * 1.0 / 8, height / 8] )'
            }
        },
        {
            'name': 'bright_downsample_32',
            'shader' : '#source(clay.compositor.downsample)',
            'inputs' : {
                'texture' : 'bright_downsample_16'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 32)',
                        'height' : 'expr(height * 1.0 / 32)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'textureSize': 'expr( [width * 1.0 / 16, height / 16] )'
            }
        },


        {
            'name' : 'bright_upsample_16_blur_h',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_downsample_32'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 16)',
                        'height' : 'expr(height * 1.0 / 16)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 0.0,
                'textureSize': 'expr( [width * 1.0 / 32, height / 32] )'
            }
        },
        {
            'name' : 'bright_upsample_16_blur_v',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_upsample_16_blur_h'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 16)',
                        'height' : 'expr(height * 1.0 / 16)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 1.0,
                'textureSize': 'expr( [width * 1.0 / 32, height * 1.0 / 32] )'
            }
        },



        {
            'name' : 'bright_upsample_8_blur_h',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_downsample_16'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 8)',
                        'height' : 'expr(height * 1.0 / 8)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 0.0,
                'textureSize': 'expr( [width * 1.0 / 16, height * 1.0 / 16] )'
            }
        },
        {
            'name' : 'bright_upsample_8_blur_v',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_upsample_8_blur_h'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 8)',
                        'height' : 'expr(height * 1.0 / 8)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 1.0,
                'textureSize': 'expr( [width * 1.0 / 16, height * 1.0 / 16] )'
            }
        },
        {
            'name' : 'bright_upsample_8_blend',
            'shader' : '#source(clay.compositor.blend)',
            'inputs' : {
                'texture1' : 'bright_upsample_8_blur_v',
                'texture2' : 'bright_upsample_16_blur_v'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 8)',
                        'height' : 'expr(height * 1.0 / 8)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'weight1' : 0.3,
                'weight2' : 0.7
            }
        },


        {
            'name' : 'bright_upsample_4_blur_h',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_downsample_8'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 4)',
                        'height' : 'expr(height * 1.0 / 4)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 0.0,
                'textureSize': 'expr( [width * 1.0 / 8, height * 1.0 / 8] )'
            }
        },
        {
            'name' : 'bright_upsample_4_blur_v',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_upsample_4_blur_h'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 4)',
                        'height' : 'expr(height * 1.0 / 4)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 1.0,
                'textureSize': 'expr( [width * 1.0 / 8, height * 1.0 / 8] )'
            }
        },
        {
            'name' : 'bright_upsample_4_blend',
            'shader' : '#source(clay.compositor.blend)',
            'inputs' : {
                'texture1' : 'bright_upsample_4_blur_v',
                'texture2' : 'bright_upsample_8_blend'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 4)',
                        'height' : 'expr(height * 1.0 / 4)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'weight1' : 0.3,
                'weight2' : 0.7
            }
        },





        {
            'name' : 'bright_upsample_2_blur_h',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_downsample_4'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 2)',
                        'height' : 'expr(height * 1.0 / 2)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 0.0,
                'textureSize': 'expr( [width * 1.0 / 4, height * 1.0 / 4] )'
            }
        },
        {
            'name' : 'bright_upsample_2_blur_v',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_upsample_2_blur_h'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 2)',
                        'height' : 'expr(height * 1.0 / 2)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 1.0,
                'textureSize': 'expr( [width * 1.0 / 4, height * 1.0 / 4] )'
            }
        },
        {
            'name' : 'bright_upsample_2_blend',
            'shader' : '#source(clay.compositor.blend)',
            'inputs' : {
                'texture1' : 'bright_upsample_2_blur_v',
                'texture2' : 'bright_upsample_4_blend'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0 / 2)',
                        'height' : 'expr(height * 1.0 / 2)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'weight1' : 0.3,
                'weight2' : 0.7
            }
        },



        {
            'name' : 'bright_upsample_full_blur_h',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0)',
                        'height' : 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 0.0,
                'textureSize': 'expr( [width * 1.0 / 2, height * 1.0 / 2] )'
            }
        },
        {
            'name' : 'bright_upsample_full_blur_v',
            'shader' : '#source(clay.compositor.gaussian_blur)',
            'inputs' : {
                'texture' : 'bright_upsample_full_blur_h'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0)',
                        'height' : 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'blurSize' : 1,
                'blurDir': 1.0,
                'textureSize': 'expr( [width * 1.0 / 2, height * 1.0 / 2] )'
            }
        },
        {
            'name' : 'bloom_composite',
            'shader' : '#source(clay.compositor.blend)',
            'inputs' : {
                'texture1' : 'bright_upsample_full_blur_v',
                'texture2' : 'bright_upsample_2_blend'
            },
            'outputs' : {
                'color' : {
                    'parameters' : {
                        'width' : 'expr(width * 1.0)',
                        'height' : 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters' : {
                'weight1' : 0.3,
                'weight2' : 0.7
            }
        },


        {
            'name': 'coc',
            'shader': '#source(car.dof.coc)',
            'outputs': {
                'color': {
                    'parameters': {
                        'width': 'expr(width * 1.0)',
                        'height': 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            }
        },

        {
            'name': 'coc_dilate_1',
            'shader': '#source(car.dof.dilateCoc)',
            'inputs': {
                'cocTex': 'coc'
            },
            'outputs': {
                'color': {
                    'parameters': {
                        'width': 'expr(width * 1.0)',
                        'height': 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters': {
                'textureSize': 'expr( [width / 1.0 * 1.0, height / 1.0 * 1.0] )'
            }
        },

        {
            'name': 'coc_dilate_2',
            'shader': '#source(car.dof.dilateCoc)',
            'inputs': {
                'cocTex': 'coc_dilate_1'
            },
            'outputs': {
                'color': {
                    'parameters': {
                        'width': 'expr(width * 1.0)',
                        'height': 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'parameters': {
                'textureSize': 'expr( [width / 1.0 * 1.0, height / 1.0 * 1.0] )'
            },
            'defines': {
                'VERTICAL': null
            }
        },

        {
            'name': 'dof_separate_far',
            'shader': '#source(car.dof.separate)',
            'inputs': {
                'mainTex': 'source',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'defines': {
                'FARFIELD': null
            }
        },

        {
            'name': 'dof_separate_near',
            'shader': '#source(car.dof.separate)',
            'inputs': {
                'mainTex': 'source',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS
        },

        {
            'name': 'dof_blur_far_1',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_far',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'R_PASS': null,
                'FARFIELD': null
            }
        },

        {
            'name': 'dof_blur_far_2',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_far',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'G_PASS': null,
                'FARFIELD': null
            }
        },


        {
            'name': 'dof_blur_far_3',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_far',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'B_PASS': null,
                'FARFIELD': null
            }
        },


        {
            'name': 'dof_blur_far_4',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_far',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'A_PASS': null,
                'FARFIELD': null
            }
        },

        {
            'name': 'dof_blur_far_final',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'rTex': 'dof_blur_far_1',
                'gTex': 'dof_blur_far_2',
                'bTex': 'dof_blur_far_3',
                'aTex': 'dof_blur_far_4',
                'cocTex': 'coc'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'FINAL_PASS': null,
                'FARFIELD': null
            }
        },

        {
            'name': 'dof_blur_near_1',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_near',
                'cocTex': 'coc',
                'dilateCocTex': 'coc_dilate_2'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'R_PASS': null
            }
        },

        {
            'name': 'dof_blur_near_2',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_near',
                'cocTex': 'coc',
                'dilateCocTex': 'coc_dilate_2'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'G_PASS': null
            }
        },


        {
            'name': 'dof_blur_near_3',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_near',
                'cocTex': 'coc',
                'dilateCocTex': 'coc_dilate_2'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'B_PASS': null
            }
        },

        {
            'name': 'dof_blur_near_4',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'mainTex': 'dof_separate_near',
                'cocTex': 'coc',
                'dilateCocTex': 'coc_dilate_2'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'A_PASS': null
            }
        },

        {
            'name': 'dof_blur_near_final',
            'shader': '#source(car.dof.blur)',
            'inputs': {
                'rTex': 'dof_blur_near_1',
                'gTex': 'dof_blur_near_2',
                'bTex': 'dof_blur_near_3',
                'aTex': 'dof_blur_near_4',
                'cocTex': 'coc',
                'dilateCocTex': 'coc_dilate_2'
            },
            'outputs': DOF_BLUR_OUTPUTS,
            'parameters': DOF_BLUR_PARAMETERS,
            'defines': {
                'FINAL_PASS': null
            }
        },


        // {
        //     'name': 'dof_blur_upsample',
        //     'shader': '#source(car.dof.extraBlur)',
        //     'inputs': {
        //         'blur': 'dof_blur',
        //         'cocTex': 'coc'
        //     },
        //     'outputs': {
        //         'color': {
        //             'parameters': {
        //                 'width': 'expr(width * 1.0)',
        //                 'height': 'expr(height * 1.0)',
        //                 'type': 'HALF_FLOAT'
        //             }
        //         }
        //     },
        //     'parameters': {
        //         'textureSize': 'expr( [width / 2.0 * 1.0, height / 2.0 * 1.0] )'
        //     }
        // },

        {
            'name': 'dof_composite',
            'shader': '#source(car.dof.composite)',
            'inputs': {
                'sharpTex': 'source',
                'farTex': 'dof_blur_far_final',
                'nearTex': 'dof_blur_near_final',
                'cocTex': 'coc'
            },
            'outputs': {
                'color': {
                    'parameters': {
                        'width': 'expr(width * 1.0)',
                        'height': 'expr(height * 1.0)',
                        'type': 'HALF_FLOAT'
                    }
                }
            },
            'defines': {
                // DEBUG: 4
            }
        },
        {
            'name' : 'composite',
            'shader' : '#source(clay.compositor.hdr.composite)',
            'inputs' : {
                'texture': 'source',
                'bloom' : 'bloom_composite'
            },
            'defines': {
                // Images are all premultiplied alpha before composite because of blending.
                // 'PREMULTIPLY_ALPHA': null,
                // 'DEBUG': 1
            }
        },
        {
            'name' : 'FXAA',
            'shader' : '#source(clay.compositor.fxaa)',
            'inputs' : {
                'texture' : 'composite'
            }
        }
    ]
};
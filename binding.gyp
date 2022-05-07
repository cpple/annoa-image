{
    'conditions': [
        [
            'OS=="win"', {'variables': {'obj': 'obj'}},
            {
                'variables': {
                    'obj': 'o'
                }
            }
        ]
    ],
    "targets": [
        {
            "target_name": "annoa_image",
            'variables': {
                'runtime_link%': 'annoa_image'
            },
            "sources": [
                "./src/cuda/tool.cu",
                "./src/cuda/cuda.cpp",
                "./src/tools/tool.cpp",
                "./src/tools/CudaDevice.cpp",
                "./src/wrap/Image.cpp",
                "./src/wrap/MateData.cpp",
                "./src/image.cc"
            ],
            'cflags!': [ '-fno-exceptions' ],
            'cflags_cc!': [ '-fno-exceptions' ],
            'include_dirs' : [
                "<!@(node -p \"require('node-addon-api').include\")"
            ],
            'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
            'rules': [
                {
                    'extension': 'cu',
                    'inputs': ['<(RULE_INPUT_PATH)'],
                    'outputs':['<(INTERMEDIATE_DIR)/<(RULE_INPUT_ROOT).<(obj)'],
                    'conditions': [
                        [
                            'OS=="win"',
                            {
                                'rule_name': 'cuda on windows',
                                'message': "compile cuda file on windows",
                                'process_outputs_as_sources': 0,
                                'action': ['nvcc --use-local-env -c <(_inputs) -o <(_outputs)'],
                            },
                            {
                                'rule_name': 'cuda on linux',
                                'message': "compile cuda file on linux",
                                'process_outputs_as_sources': 1,
                                'action': [
                                    'nvcc', '-Xcompiler', '-fpic', '-c',
                                    '<@(_inputs)','-o', '<@(_outputs)'
                                ],
                            }
                        ]
                    ]
                }
            ],
            'defines': [
                '_GNU_SOURCE',
            ],
            'conditions': [
                ['OS=="win"', {
                    "msvs_settings": {
                        "VCCLCompilerTool": {
                            "ExceptionHandling": 1
                        }
                    }
                }],
                ['OS=="mac"', {
                    'libraries': [
                        '-framework CUDA'
                    ],
                    'include_dirs': [
                        '/usr/local/include'
                    ],
                    'library_dirs': [
                        '/usr/local/lib'
                    ],
                    'cflags+': ['-fvisibility=hidden'],
                    "xcode_settings": {
                        "CLANG_CXX_LIBRARY": "libc++",
                        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
                        'GCC_SYMBOLS_PRIVATE_EXTERN': 'YES', # -fvisibility=hidden
                    }
                }],
                [ 'OS=="zos"', {
                    'cflags': [
                        '-qascii',
                    ],
                }],
                [
                    'OS=="win"',
                    {
                        'conditions': [
                            [
                                'target_arch=="x64"',
                                {
                                    'variables': {
                                        'arch': 'x64'
                                    }
                                },
                                {
                                    'variables': {
                                        'arch': 'Win32'
                                    }
                                }
                            ]
                        ],
                        'variables': {
                            'cuda_root%': '<!(echo %CUDA_PATH%)'
                        },
                        'libraries': [
                            "<!@(node utils/find-cudnn.js --libs)"
                        ],
                        "include_dirs": [
                            "<!@(node utils/find-cudnn.js --cflags)"
                        ]
                    },
                    {
                        "include_dirs": [
                            "/usr/local/cuda-5.0/include",
                            "/usr/local/cuda/include"
                        ]
                    }
                ]
            ],
            'configurations': {
                'Release': {
                    'conditions': [
                        ['OS == "linux"', {
                            'cflags_cc': [
                                '-Wno-cast-function-type'
                            ]
                        }],
                        ['target_arch == "arm"', {
                            'cflags_cc': [
                                '-Wno-psabi'
                            ]
                        }],
                        ['OS == "win"', {
                            'msvs_settings': {
                                'VCCLCompilerTool': {
                                    'ExceptionHandling': 1,
                                    'WholeProgramOptimization': 'true'
                                },
                                'VCLibrarianTool': {
                                    'AdditionalOptions': [
                                        '/LTCG:INCREMENTAL'
                                    ]
                                },
                                'VCLinkerTool': {
                                    'ImageHasSafeExceptionHandlers': 'false',
                                    'OptimizeReferences': 2,
                                    'EnableCOMDATFolding': 2,
                                    'LinkIncremental': 1,
                                    'AdditionalOptions': [
                                        '/LTCG:INCREMENTAL'
                                    ]
                                }
                            },
                            'msvs_disabled_warnings': [
                                4275
                            ]
                        }]
                    ]
                }
            },
        },
        {
            "target_name": "action_after_build",
            "type": "none",
            "dependencies": [ "<(module_name)" ],
            "copies": [
                {
                    "files": [ "<(PRODUCT_DIR)/<(module_name).node" ],
                    "destination": "<(module_path)"
                }
            ]
        }
    ]
}

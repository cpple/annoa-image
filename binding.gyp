{
  "targets": [
    {
      "target_name": "annoa_image",
      'variables': {
        'runtime_link%': 'annoa_image'
      },
      "sources": [
        "./src/image.cc",
        "./src/tools/tool.cpp",
        "./src/wrap/Image.cpp",
        "./src/wrap/MateData.cpp"
       ],
      'cflags!': [ '-fno-exceptions' ],
      'cflags_cc!': [ '-fno-exceptions' ],
      'include_dirs' : [
          "<!@(node -p \"require('node-addon-api').include\")"
      ],
      'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
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

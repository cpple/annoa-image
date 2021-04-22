{
  "targets": [
    {
      "target_name": "image",
      "sources": [
        "./src/image.cc",
        "./src/tools/tool.cpp"
       ],
      "include_dirs": ["./include"],
      'cflags!': [ '-fno-exceptions' ],
      'cflags_cc!': [ '-fno-exceptions' ],
      'defines': [
            '_GNU_SOURCE',
      ],
      'conditions': [
        ['OS=="win"', {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1
            }
          },
          'defines': [
            'uint=unsigned int',
          ]
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
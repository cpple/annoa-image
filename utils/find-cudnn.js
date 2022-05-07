"use strict";

var exec = require("child_process").exec;
var fs = require("fs");

var flags = {
    '--cflags' : '--cflags',
    '--libs' : '--libs'
}
var flag = flags[process.argv[2]] || '--exists'


// Normally |pkg-config cudnn ...| could report either cudnn 2.x or cudnn 3.y
// depending on what is installed.  To enable both 2.x and 3.y to co-exist on
// the same machine, the cudnn.pc for 3.y can be installed as cudnn3.pc and
// then selected by |export PKG_CONFIG_cudnn3=1| before building node-cudnn.

function main(){
    //Try using pkg-config, but if it fails and it is on Windows, try the fallback
    exec("pkg-config " + "cudnn" + " " + flag, function(error, stdout, stderr){
        if(error){
            if(process.platform === "win32"){
                fallback();
            }
            else{
                throw new Error("ERROR: failed to run: pkg-config" + cudnn + " " + flag + " - Is cudnn installed?");
            }
        }
        else{
            console.log(stdout);
        }
    });
}

//======================Windows Specific=======================================

function fallback(){
    exec("echo %CUDA_PATH%", function(error, stdout, stderr){
        stdout = cleanupEchoOutput(stdout);
        if(error){
            throw new Error("ERROR: There was an error reading cudnn_DIR");
        }
        else if(stdout === "%CUDA_PATH%") {
            throw new Error("ERROR: cudnn_DIR doesn't seem to be defined");
        }
        else {
            printPaths(stdout);
        }
    });
}

function printPaths(cudnnPath){
    if(flag === "--cflags") {
        console.log("\"" + cudnnPath + "\\include\"");
    }
    else if(flag === "--libs") {
        var libPath = cudnnPath + "\\lib\\x64\\";

        fs.readdir(libPath, function(err, files){
            if(err){
                throw new Error("ERROR: couldn't read the lib directory " + err);
            }

            var libs = "";
            for(var i = 0; i < files.length; i++){
                if(getExtension(files[i]) === "lib"){
                    libs = libs + " \"" + libPath + files[i] + "\" \r\n ";
                }
            }
            console.log(libs);
        });
    }
    else {
        throw new Error("Error: unknown argument '" + flag + "'");
    }
}

function cleanupEchoOutput(s){
    return s.slice(0, s.length - 2);
}

function getExtension(s){
    return s.substr(s.lastIndexOf(".") + 1);
}
main();

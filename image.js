'use strict';

var nodePreGyp = require('@mapbox/node-pre-gyp');
var path = require('path');
var binding_path = nodePreGyp.find(path.resolve(path.join(__dirname, './package.json')));
var bindings = require(binding_path);
let exp = {
    removeAlpha:function (img){
        return bindings["removeAlpha"](img);
    },
    captureBBImg:function (img, channels, w, h, bboxList) {
        return bindings["captureBBImg"](img, channels, w, h, bboxList);
    },
    convertNetData:function (img, scale){
        return bindings["convertNetData"](img, scale)
    },
    imgRandomCropHorizontalFlipNormalize:function (img, scale, batch, oh, ow, h, w, p = 0, channelFirst = false, horizontal = false, normalize = false, mean = null, std = null){
        return bindings["imgRandomCropHorizontalFlipNormalize"](img, scale, batch, oh, ow, h, w, p, channelFirst, horizontal, normalize, mean, std)
    },
    imgNormalize:function (img, scale, batch, mean, std, channelsFirst = true){
        return bindings["imgNormalize"](img, scale, batch, mean, std, channelsFirst)
    },
    imgScale:function (img, h, w, sh, sw, batch, channelsFirst = true){
        return bindings["imgScale"](img, h, w, sh, sw, batch, channelsFirst)
    },
    imgColorHSV:function (img, batch, h, w, hue = 0, sat = 1, val = 1, channelsFirst = true){
        return bindings["imgColorHSV"](img, h, w, hue, sat, val, batch, channelsFirst)
    },
    test:function (){
        return bindings["test"]()
    },
    Image:bindings.Image
};
module.exports = exp;
//
// let image = new bindings.Image(4,3,3);
// let array = Uint8Array.from([
//     10,11,12,13, 20,21,22,23, 30,31,32,33,
//     40,41,42,43, 50,51,52,53, 60,61,62,63,
//     70,71,72,73, 80,81,82,83, 90,91,92,93,
// ])
// image.SetNHWCData(array);
// let nchw = image.GetNCHWData();
// console.error(nchw);
// image.SetNCHWData(nchw);
// image.RemoveAlpha();
// let nhwco = image.GetNCHWData();
// console.error(nhwco);
//
// let nhwc = image.GetNHWCData();
//
// console.error(nhwc);

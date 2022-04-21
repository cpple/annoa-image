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

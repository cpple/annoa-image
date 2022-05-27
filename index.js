'use strict';

var nodePreGyp = require('@mapbox/node-pre-gyp');
var path = require('path');
var binding_path = nodePreGyp.find(path.resolve(path.join(__dirname, './package.json')));
var bindings = require(binding_path);
let exp = {
    Image:bindings.Image,
    MateData:bindings.MateData,
    imgRandomCropHorizontalFlipNormalize:bindings.imgRandomCropHorizontalFlipNormalize,
    imgNormalize:bindings.imgNormalize,
    imgScale:bindings.imgScale,
    ChessBoardDraw:bindings.ChessBoardDraw
};
module.exports = exp;

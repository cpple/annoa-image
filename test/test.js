
let image = require('../');

let test = async function (arrO) {
    try{
        //let result = image.imgRandomCropHorizontalFlipNormalize(arrO, 1/255, 1, 9, 9, 9, 9, 0, true, false, true, Float32Array.from([0.40789654, 0.44719302, 0.47026115]), Float32Array.from([0.28863828, 0.27408164, 0.27809835]));
        //let result = image.imgNormalize(arrO, 1 / 255, 1, Float32Array.from([0.4914, 0.4822, 0.4465]), Float32Array.from([0.2023, 0.1994, 0.2010]), false);
        //let result = image.removeAlpha(arrO);
        //let result = image.convertNetData(arrO, 1/255);
        //let result = image.imgScale(arrO, 9, 9, 1.4, 1.4, 1, true);
        //let result = image.imgColorHSV(arrO, 1, 9, 9, 0, 1, 3, true);
        let image = new image.Image(3,3,3);
        let array = Uint8Array.from([
            1,1,1, 2,2,2, 3,3,3,
            4,4,4, 5,5,5, 6,6,6,
            7,7,7, 8,8,8, 9,9,9,
        ])
        image.SetNHWCData(array);

        let nchw = image.GetNCHWData();
        console.error(nchw);
        image.SetNCHWData(nchw);
        let nhwc = image.GetNHWCData();
        console.error(nhwc);

        // console.error(JSON.stringify(image.GetNCHWData()));
        // console.error(JSON.stringify(Array.from(result)));
    } catch (err) {

        console.error(err);
    }
};

let mainTest = function () {

    let rma = [
        192,110,190,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,
        100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,100,100,255,100,110,120,255,
    ];
    let input = [
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,

        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,
        150,150,150,150,150,150,150,150,150,

        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,
        200,200,200,200,200,200,200,200,200,

        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
        255,255,255,255,255,255,255,255,255,
    ];
    let input2 = [
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,

        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,

        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
        100,100,100,100,100,100,100,100,100,
    ];
    let input0 = [
        110,110,110,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
        110,120,130,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
        110,120,130,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
        110,120,130,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
        110,120,130,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
        110,120,130,110,120,130,110,120,130,111,121,131,111,121,131,111,121,131,
    ]
    //console.log(JSON.stringify(rma));
    //test(Uint8Array.from(rma));
    test(Uint8Array.from(rma));
};

mainTest();

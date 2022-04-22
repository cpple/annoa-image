declare module 'annoa-image' {

    class Image {
        constructor(channel:number, height:number, width:number);

        GetNCHWData():Uint8Array;
        GetNHWCData():Uint8Array;
        SetNCHWData(data:Uint8Array):this;
        SetNHWCData(data:Uint8Array):this;
        RemoveAlpha():this;
        HorizontalFlip():this;
        RandomCrop(height:number, width:number, pad:number):Int32Array;
        Normalize(mean:Array<number> | Float32Array, std:Array<number> | Float32Array):MateData;
        MateData():MateData;
        ScaleSize(height:number, width:number):this;
        ColorHSV():this;
        CaptureImgByBoundingBox(boundingBox:Array<Float32Array | Uint32Array>, batch_idx:number):Image[];
    }
    class MateData {
        constructor(channel:number, height:number, weight:number);
        GetData():Float32Array;
        SetData(data:Float32Array):this;
        Normalize(mean:Array<number> | Float32Array, std:Array<number> | Float32Array):this;
        Scale(scale:number):this;
    }
}

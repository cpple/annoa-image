declare module 'annoa-image' {

    class Image {
        constructor(channels:number, height:number, width:number, has_alpha?:boolean);

        GetNCHWData():Uint8Array;
        GetNHWCData():Uint8Array;
        SetNCHWData(data:Uint8Array):this;
        SetNHWCData(data:Uint8Array):this;
        RemoveAlpha():this;
        HorizontalFlip():this;
        RandomCrop(height:number, width:number, pad:number):Int32Array;
        NormalizeToMateData(mean:Array<number> | Float32Array, std:Array<number> | Float32Array, scale:number):MateData;
        MateData():MateData;
        ScaleSize(height:number, width:number):this;
        ColorHSV(hue:number, saturation:number, value:number):this;
        CaptureImgByBoundingBox(boundingBox:Array<Float32Array | Uint32Array>, batch_idx:number):Image[];
        GreyScale(remove_alpha?:boolean, rgb_merged?:boolean, gamma?:number):this;
    }
    class MateData {
        constructor(channels:number, height:number, weight:number);
        GetData():Float32Array;
        SetData(data:Float32Array):this;
        Normalize(mean:Array<number> | Float32Array, std:Array<number> | Float32Array, scale:number):this;
        Scale(scale:number):this;
    }
}

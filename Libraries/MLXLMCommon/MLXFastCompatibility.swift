import MLX

extension MLXFast {
    /// Compatibility no-op for runtimes that do not expose prefault helpers.
    public static func prefault(_ array: MLXArray) {
        MLX.eval(array)
    }

    /// Compatibility fallback for runtimes that do not expose direct SSD expert streaming.
    /// Falls back to slicing the already-mapped quantized expert weights from the in-memory tensor.
    public static func streamedGatherMM(
        x: MLXArray,
        wShape: MLXArray,
        activeExpert: UInt32,
        safetensorsPath: String,
        tensorName: String
    ) -> MLXArray {
        let expert = Int(activeExpert)
        return wShape[expert ..< expert + 1]
    }

    /// Compatibility fallback for runtimes without SharpAI TurboKV helpers.
    /// This preserves correctness by carrying fp16 tensors through unchanged.
    public static func turboQuantEncode(
        keys: MLXArray,
        values: MLXArray,
        bits: Int
    ) -> ((MLXArray, MLXArray?), (MLXArray, MLXArray?)) {
        ((keys, nil), (values, nil))
    }

    /// Compatibility fallback that treats "packed" K as already-decoded fp16 history.
    public static func turboDecodeK(packed: MLXArray) -> MLXArray {
        packed
    }

    /// Compatibility fallback that treats "packed" V as already-decoded fp16 history.
    public static func turboDecodeV(packed: MLXArray) -> MLXArray {
        packed
    }
}

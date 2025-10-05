package com.example.emergencytriage.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import com.example.emergencytriage.data.models.ImagePrediction

class ImageClassifier(private val context: Context) {
    
    companion object {
        private const val TAG = "ImageClassifier"
        private const val MODEL_FILE = "models/rash_model.tflite"
        private const val LABELS_FILE = "models/rash_model_labels.txt"
        private const val INPUT_SIZE = 224
        private const val PIXEL_SIZE = 3
        private const val IMAGE_MEAN = 127.5f
        private const val IMAGE_STD = 127.5f
        private const val MAX_RESULTS = 3
    }

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var inputImageBuffer: ByteBuffer? = null
    private var outputProbabilityBuffer: Array<FloatArray>? = null

    init {
        try {
            loadModel()
            loadLabels()
            initializeBuffers()
            Log.d(TAG, "ImageClassifier initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing ImageClassifier", e)
        }
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseNNAPI(true) // Use Neural Networks API if available
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading model", e)
            throw e
        }
    }

    private fun loadModelFile(): MappedByteBuffer {
        return try {
            val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
            val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model file: $MODEL_FILE", e)
            throw e
        }
    }

    private fun loadLabels() {
        try {
            labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
            Log.d(TAG, "Loaded ${labels.size} labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels", e)
            // Fallback to default labels if file not found
            labels = getDefaultLabels()
        }
    }

    private fun getDefaultLabels(): List<String> {
        return listOf(
            "Acne", "Actinic Keratosis", "Benign Tumor", "Bullous", "Candidiasis",
            "Drug Eruption", "Eczema", "Infestation Bite", "Lichen", "Lupus",
            "Moles", "Psoriasis", "Rosacea", "Seborrh Keratosis", "Skin Cancer",
            "Sunlight Damage", "Tinea", "Vascular Tumor", "Vasculitis", "Vitiligo", "Warts"
        )
    }

    private fun initializeBuffers() {
        // Input buffer for image data
        inputImageBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE).apply {
            order(ByteOrder.nativeOrder())
        }

        // Output buffer for predictions
        outputProbabilityBuffer = Array(1) { FloatArray(labels.size) }
    }

    fun classifyImage(bitmap: Bitmap): ImagePrediction {
        if (interpreter == null) {
            throw IllegalStateException("Model not initialized")
        }

        try {
            // Preprocess the image
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false)
            convertBitmapToByteBuffer(resizedBitmap)

            // Run inference
            interpreter?.run(inputImageBuffer, outputProbabilityBuffer)

            // Process results
            return processResults()
        } catch (e: Exception) {
            Log.e(TAG, "Error during image classification", e)
            throw e
        }
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        inputImageBuffer?.rewind()
        
        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = bitmap.getPixel(x, y)
                
                // Extract RGB values and normalize
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f  
                val b = (pixel and 0xFF) / 255.0f
                
                // Apply normalization (adjust based on your model's requirements)
                inputImageBuffer?.putFloat(r)
                inputImageBuffer?.putFloat(g)
                inputImageBuffer?.putFloat(b)
            }
        }
    }

    private fun processResults(): ImagePrediction {
        val probabilities = outputProbabilityBuffer?.get(0) ?: FloatArray(0)
        
        // Find the top predictions
        val topPredictions = mutableListOf<Pair<String, Float>>()
        
        for (i in probabilities.indices) {
            topPredictions.add(Pair(labels[i], probabilities[i]))
        }
        
        // Sort by probability (descending)
        topPredictions.sortByDescending { it.second }
        
        // Take top predictions
        val topResults = topPredictions.take(MAX_RESULTS)
        
        // Create result object
        val topLabel = topResults.firstOrNull()?.first ?: "Unknown"
        val topConfidence = topResults.firstOrNull()?.second ?: 0.0f
        
        val allPredictions = topResults.associate { it.first to it.second }
        
        Log.d(TAG, "Top prediction: $topLabel with confidence: $topConfidence")
        
        return ImagePrediction(
            predictedClass = topLabel,
            confidence = topConfidence,
            allPredictions = allPredictions
        )
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    // Additional utility methods
    fun getModelInfo(): String {
        return """
            Model: $MODEL_FILE
            Labels: ${labels.size} classes
            Input Size: ${INPUT_SIZE}x$INPUT_SIZE
            Pixel Channels: $PIXEL_SIZE
        """.trimIndent()
    }

    fun isModelLoaded(): Boolean {
        return interpreter != null && labels.isNotEmpty()
    }
}
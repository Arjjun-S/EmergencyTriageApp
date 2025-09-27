package com.example.emergencytriage.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import com.example.emergencytriage.data.models.TextPrediction
import com.example.emergencytriage.data.models.SeverityLevel
import java.util.*
import kotlin.collections.HashMap

class SpeechNLPProcessor(private val context: Context) {
    
    companion object {
        private const val TAG = "SpeechNLPProcessor"
        private const val MODEL_FILE = "text_classifier.tflite"
        private const val LABELS_FILE = "text_classifier_labels.txt"
        private const val VOCAB_FILE = "vocab.txt"
        private const val MAX_SEQUENCE_LENGTH = 128
        private const val OOV_TOKEN = "<OOV>"
        private const val PAD_TOKEN = "<PAD>"
    }

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var vocabulary: Map<String, Int> = emptyMap()
    private var severityKeywords: Map<SeverityLevel, List<String>> = emptyMap()

    init {
        try {
            loadModel()
            loadLabels()
            loadVocabulary()
            initializeSeverityKeywords()
            Log.d(TAG, "SpeechNLPProcessor initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing SpeechNLPProcessor", e)
        }
    }

    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile()
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Text model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading text model", e)
            // Continue without TFLite model - use rule-based approach
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
            Log.e(TAG, "Error loading text model file: $MODEL_FILE", e)
            throw e
        }
    }

    private fun loadLabels() {
        try {
            labels = context.assets.open(LABELS_FILE).bufferedReader().readLines()
            Log.d(TAG, "Loaded ${labels.size} severity labels")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading labels, using defaults", e)
            labels = listOf("Mild", "Moderate", "Severe")
        }
    }

    private fun loadVocabulary() {
        try {
            val vocabLines = context.assets.open(VOCAB_FILE).bufferedReader().readLines()
            vocabulary = vocabLines.mapIndexed { index, word -> word to index }.toMap()
            Log.d(TAG, "Loaded vocabulary with ${vocabulary.size} words")
        } catch (e: Exception) {
            Log.e(TAG, "Error loading vocabulary, using basic tokenization", e)
            vocabulary = emptyMap()
        }
    }

    private fun initializeSeverityKeywords() {
        severityKeywords = mapOf(
            SeverityLevel.MILD to listOf(
                "mild", "slight", "minor", "light", "little", "small", "itching", "rash", 
                "dry", "irritation", "discomfort", "tender", "sore", "ache"
            ),
            SeverityLevel.MODERATE to listOf(
                "moderate", "noticeable", "persistent", "ongoing", "swollen", "swelling",
                "burning", "throbbing", "stiff", "painful", "fever", "nausea", "dizzy", 
                "tired", "fatigue", "headache", "cough", "difficulty"
            ),
            SeverityLevel.SEVERE to listOf(
                "severe", "intense", "excruciating", "unbearable", "sharp", "stabbing",
                "bleeding", "blood", "unconscious", "chest pain", "difficulty breathing",
                "shortness of breath", "vomiting", "high fever", "emergency", "urgent",
                "critical", "can't move", "paralyzed", "seizure", "heart", "stroke"
            )
        )
    }

    fun processSymptoms(symptomsText: String): TextPrediction {
        return try {
            // Clean and normalize the input text
            val cleanedText = preprocessText(symptomsText)
            
            // Try TFLite model first if available
            if (interpreter != null && vocabulary.isNotEmpty()) {
                return classifyWithModel(cleanedText)
            } else {
                // Fallback to rule-based classification
                return classifyWithRules(cleanedText)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error processing symptoms", e)
            // Return default mild prediction
            TextPrediction(
                severityLevel = SeverityLevel.MILD,
                confidence = 0.5f,
                detectedSymptoms = extractSymptoms(symptomsText),
                processedText = symptomsText
            )
        }
    }

    private fun preprocessText(text: String): String {
        return text.lowercase(Locale.getDefault())
            .replace(Regex("[^a-zA-Z0-9\\s]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
    }

    private fun classifyWithModel(text: String): TextPrediction {
        try {
            // Tokenize the text
            val tokens = tokenizeText(text)
            
            // Convert to input tensor
            val inputBuffer = createInputBuffer(tokens)
            
            // Run inference
            val outputBuffer = Array(1) { FloatArray(labels.size) }
            interpreter?.run(inputBuffer, outputBuffer)
            
            // Process results
            val probabilities = outputBuffer[0]
            val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
            val maxConfidence = probabilities[maxIndex]
            
            val severityLevel = when (labels[maxIndex].lowercase()) {
                "mild" -> SeverityLevel.MILD
                "moderate" -> SeverityLevel.MODERATE
                "severe" -> SeverityLevel.SEVERE
                else -> SeverityLevel.MILD
            }
            
            return TextPrediction(
                severityLevel = severityLevel,
                confidence = maxConfidence,
                detectedSymptoms = extractSymptoms(text),
                processedText = text
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in model-based classification", e)
            return classifyWithRules(text)
        }
    }

    private fun tokenizeText(text: String): List<Int> {
        val words = text.split("\\s+".toRegex())
        val tokens = mutableListOf<Int>()
        
        for (word in words) {
            val tokenId = vocabulary[word] ?: vocabulary[OOV_TOKEN] ?: 0
            tokens.add(tokenId)
        }
        
        return tokens
    }

    private fun createInputBuffer(tokens: List<Int>): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * MAX_SEQUENCE_LENGTH).apply {
            order(ByteOrder.nativeOrder())
        }
        
        // Pad or truncate to MAX_SEQUENCE_LENGTH
        val paddedTokens = tokens.take(MAX_SEQUENCE_LENGTH) + 
                          IntArray(maxOf(0, MAX_SEQUENCE_LENGTH - tokens.size)).toList()
        
        for (token in paddedTokens) {
            buffer.putFloat(token.toFloat())
        }
        
        buffer.rewind()
        return buffer
    }

    private fun classifyWithRules(text: String): TextPrediction {
        val words = text.split("\\s+".toRegex())
        val severityScores = mutableMapOf<SeverityLevel, Int>()
        val detectedSymptoms = mutableListOf<String>()
        
        // Initialize scores
        SeverityLevel.values().forEach { severityScores[it] = 0 }
        
        // Score based on keyword matching
        for (word in words) {
            for ((severity, keywords) in severityKeywords) {
                if (keywords.any { keyword -> word.contains(keyword, ignoreCase = true) }) {
                    severityScores[severity] = severityScores[severity]!! + 1
                    if (!detectedSymptoms.contains(word)) {
                        detectedSymptoms.add(word)
                    }
                }
            }
        }
        
        // Determine the severity level
        val predictedSeverity = severityScores.maxByOrNull { it.value }?.key ?: SeverityLevel.MILD
        val totalMatches = severityScores.values.sum()
        val confidence = if (totalMatches > 0) {
            (severityScores[predictedSeverity]!! / totalMatches.toFloat()).coerceIn(0.3f, 1.0f)
        } else {
            0.5f // Default confidence when no keywords match
        }
        
        Log.d(TAG, "Rule-based classification: $predictedSeverity with confidence: $confidence")
        
        return TextPrediction(
            severityLevel = predictedSeverity,
            confidence = confidence,
            detectedSymptoms = detectedSymptoms,
            processedText = text
        )
    }

    private fun extractSymptoms(text: String): List<String> {
        val commonSymptoms = listOf(
            "pain", "swelling", "redness", "itching", "burning", "rash", "fever", "nausea",
            "dizziness", "headache", "fatigue", "cough", "shortness of breath", "bleeding",
            "vomiting", "diarrhea", "constipation", "numbness", "tingling", "weakness"
        )
        
        val detectedSymptoms = mutableListOf<String>()
        val lowerText = text.lowercase()
        
        for (symptom in commonSymptoms) {
            if (lowerText.contains(symptom)) {
                detectedSymptoms.add(symptom)
            }
        }
        
        return detectedSymptoms
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    fun getProcessorInfo(): String {
        return """
            Model: $MODEL_FILE
            Labels: ${labels.size} severity levels
            Vocabulary: ${vocabulary.size} words
            Max Sequence Length: $MAX_SEQUENCE_LENGTH
            Using TFLite: ${interpreter != null}
        """.trimIndent()
    }
}
package com.example.emergencytriage.data.repository

import android.content.Context
import android.graphics.Bitmap
import com.example.emergencytriage.data.models.ImagePrediction
import com.example.emergencytriage.data.models.SeverityLevel
import com.example.emergencytriage.data.models.TriageResult
import com.example.emergencytriage.data.models.UrgencyLevel
import com.example.emergencytriage.ml.FusionEngine
import com.example.emergencytriage.ml.ImageClassifier
import com.example.emergencytriage.ml.SpeechNLPProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Repository that coordinates multimodal ML components and exposes a simple API.
 * It gracefully degrades when ML assets are missing so the app still runs.
 */
class TriageRepository(private val context: Context) {

    private val imageClassifier: ImageClassifier by lazy { ImageClassifier(context) }
    private val speechNLPProcessor: SpeechNLPProcessor by lazy { SpeechNLPProcessor(context) }
    private val fusionEngine: FusionEngine by lazy { FusionEngine(context) }

    suspend fun performTriageAnalysis(image: Bitmap?, symptoms: String): TriageResult =
        withContext(Dispatchers.Default) {
            // Image analysis (optional)
            val imageResult: ImagePrediction = when {
                image != null && imageClassifier.isModelLoaded() -> {
                    runCatching { imageClassifier.classifyImage(image) }
                        .getOrElse { defaultImagePrediction() }
                }
                else -> defaultImagePrediction()
            }

            // Text analysis (rule-based fallback is built-in if model missing)
            val textResult = speechNLPProcessor.processSymptoms(symptoms)

            // Fuse and return final triage
            fusionEngine.fuseResults(imageResult, textResult, symptoms)
        }

    private fun defaultImagePrediction(): ImagePrediction =
        ImagePrediction(
            predictedClass = "Unknown Skin Condition",
            confidence = 0.01f,
            allPredictions = emptyMap()
        )

    /**
     * Returns a human-readable status of the runtime ML environment.
     */
    fun getEnvironmentStatus(): String {
        val imageOk = if (imageClassifier.isModelLoaded()) "OK" else "Missing (fallback)"
        val nlpInfo = speechNLPProcessor.getProcessorInfo()
        val fusionInfo = fusionEngine.getFusionEngineInfo()
        return buildString {
            appendLine("Environment check:")
            appendLine("- Image model: $imageOk")
            appendLine("- NLP: see below")
            appendLine(nlpInfo)
            appendLine(fusionInfo)
        }.trim()
    }

    fun isImageModelAvailable(): Boolean = imageClassifier.isModelLoaded()
    fun isTextModelAvailable(): Boolean = speechNLPProcessor.getProcessorInfo().contains("Using TFLite: true")
}
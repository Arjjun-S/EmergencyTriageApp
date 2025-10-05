package com.example.emergencytriage.ml

import android.content.Context
import android.util.Log
import com.example.emergencytriage.data.models.*
import java.io.BufferedReader
import java.io.InputStreamReader

class FusionEngine(private val context: Context) {
    
    companion object {
        private const val TAG = "FusionEngine"
        private const val DISEASE_PRECAUTIONS_FILE = "models/DiseasePrecaution.csv"
    }

    // Disease severity mapping - maps diseases to their typical severity levels
    private val diseaseSeverityMap = mapOf(
        // High severity diseases (RED)
        "Skin Cancer" to UrgencyLevel.RED,
        "Melanoma" to UrgencyLevel.RED,
        "Severe Burn" to UrgencyLevel.RED,
        "Cellulitis" to UrgencyLevel.RED,
        
        // Moderate severity diseases (YELLOW)
        "Eczema" to UrgencyLevel.YELLOW,
        "Psoriasis" to UrgencyLevel.YELLOW,
        "Drug Eruption" to UrgencyLevel.YELLOW,
        "Lupus" to UrgencyLevel.YELLOW,
        "Seborrh Keratosis" to UrgencyLevel.YELLOW,
        "Bullous" to UrgencyLevel.YELLOW,
        "Vasculitis" to UrgencyLevel.YELLOW,
        
        // Low severity diseases (GREEN)
        "Acne" to UrgencyLevel.GREEN,
        "Moles" to UrgencyLevel.GREEN,
        "Benign Tumor" to UrgencyLevel.GREEN,
        "Vitiligo" to UrgencyLevel.GREEN,
        "Warts" to UrgencyLevel.GREEN,
        "Rosacea" to UrgencyLevel.GREEN,
        "Sunlight Damage" to UrgencyLevel.GREEN
    )

    private var precautionsMap: Map<String, List<String>> = emptyMap()

    init {
        loadPrecautionsData()
    }

    private fun loadPrecautionsData() {
        try {
            val inputStream = context.assets.open(DISEASE_PRECAUTIONS_FILE)
            val reader = BufferedReader(InputStreamReader(inputStream))
            val precautions = mutableMapOf<String, List<String>>()

            // Read and skip header if present
            var line: String? = reader.readLine()
            while (true) {
                line = reader.readLine() ?: break
                val columns = line.split(",").map { it.trim() }
                if (columns.size >= 5) {
                    val disease = columns[0]
                    val precautionList = columns.subList(1, 5)
                        .filter { it.isNotEmpty() && it.isNotBlank() }
                    precautions[disease] = precautionList
                }
            }
            reader.close()
            
            precautionsMap = precautions
            Log.d(TAG, "Loaded precautions for ${precautionsMap.size} diseases")
            
        } catch (e: Exception) {
            Log.e(TAG, "Error loading precautions data", e)
            precautionsMap = getDefaultPrecautions()
        }
    }

    private fun getDefaultPrecautions(): Map<String, List<String>> {
        return mapOf(
            "Acne" to listOf("Keep skin clean", "Avoid touching face", "Use non-comedogenic products", "Consult dermatologist if severe"),
            "Skin Cancer" to listOf("Seek immediate medical attention", "Avoid sun exposure", "Do not self-treat", "Get professional biopsy"),
            "Eczema" to listOf("Moisturize regularly", "Avoid triggers", "Use gentle soap", "Consider antihistamines"),
            "Psoriasis" to listOf("Moisturize daily", "Avoid stress", "Get sunlight in moderation", "Consult dermatologist"),
            "Unknown" to listOf("Monitor symptoms", "Keep area clean", "Avoid irritants", "Seek medical advice if worsening")
        )
    }

    fun fuseResults(
        imageResult: ImagePrediction,
        textResult: TextPrediction,
        originalSymptoms: String
    ): TriageResult {
        try {
            // Determine predicted disease from image classification
            val predictedDisease = imageResult.predictedClass
            
            // Calculate combined confidence
            val combinedConfidence = calculateCombinedConfidence(imageResult, textResult)
            
            // Determine urgency level using fusion logic
            val urgencyLevel = determineUrgencyLevel(imageResult, textResult)
            
            // Get precautions for the predicted disease
            val precautions = getPrecautionsForDisease(predictedDisease)
            
            // Create comprehensive triage result
            val triageResult = TriageResult(
                predictedDisease = predictedDisease,
                confidence = combinedConfidence,
                urgencyLevel = urgencyLevel,
                imageConfidence = imageResult.confidence,
                textSeverity = textResult.severityLevel,
                textConfidence = textResult.confidence,
                detectedSymptoms = textResult.detectedSymptoms,
                precautions = precautions,
                recommendedActions = getRecommendedActions(urgencyLevel, predictedDisease),
                timestamp = System.currentTimeMillis()
            )
            
            Log.d(TAG, "Fusion complete: $predictedDisease, Urgency: $urgencyLevel, Confidence: $combinedConfidence")
            return triageResult
            
        } catch (e: Exception) {
            Log.e(TAG, "Error in fusion process", e)
            return createDefaultResult(originalSymptoms)
        }
    }

    private fun calculateCombinedConfidence(
        imageResult: ImagePrediction,
        textResult: TextPrediction
    ): Float {
        // Weighted combination of image and text confidence
        val imageWeight = 0.7f // Image classification typically more reliable for skin conditions
        val textWeight = 0.3f
        
        return (imageResult.confidence * imageWeight + textResult.confidence * textWeight)
            .coerceIn(0.0f, 1.0f)
    }

    private fun determineUrgencyLevel(
        imageResult: ImagePrediction,
        textResult: TextPrediction
    ): UrgencyLevel {
        // Get base urgency from disease type
        val diseaseUrgency = diseaseSeverityMap[imageResult.predictedClass] ?: UrgencyLevel.YELLOW
        
        // Get urgency from symptom severity
        val symptomUrgency = when (textResult.severityLevel) {
            SeverityLevel.MILD -> UrgencyLevel.GREEN
            SeverityLevel.MODERATE -> UrgencyLevel.YELLOW
            SeverityLevel.SEVERE -> UrgencyLevel.RED
        }
        
        // Take the higher urgency level (more conservative approach)
        return when {
            diseaseUrgency == UrgencyLevel.RED || symptomUrgency == UrgencyLevel.RED -> UrgencyLevel.RED
            diseaseUrgency == UrgencyLevel.YELLOW || symptomUrgency == UrgencyLevel.YELLOW -> UrgencyLevel.YELLOW
            else -> UrgencyLevel.GREEN
        }
    }

    private fun getPrecautionsForDisease(disease: String): List<String> {
        return precautionsMap[disease] 
            ?: precautionsMap["Unknown"] 
            ?: listOf("Monitor symptoms", "Seek medical advice if symptoms worsen")
    }

    private fun getRecommendedActions(urgencyLevel: UrgencyLevel, disease: String): List<String> {
        val baseActions = when (urgencyLevel) {
            UrgencyLevel.RED -> listOf(
                "Seek immediate medical attention",
                "Call emergency services if condition is life-threatening",
                "Do not delay treatment",
                "Consider visiting emergency room"
            )
            UrgencyLevel.YELLOW -> listOf(
                "Schedule appointment with healthcare provider within 24-48 hours",
                "Monitor symptoms closely",
                "Consider telehealth consultation",
                "Seek immediate care if symptoms worsen"
            )
            UrgencyLevel.GREEN -> listOf(
                "Monitor symptoms for changes",
                "Schedule routine appointment with healthcare provider",
                "Follow general care guidelines",
                "Contact doctor if symptoms persist or worsen"
            )
        }
        
        // Add disease-specific actions
        val diseaseSpecificActions = when {
            disease.contains("Cancer", ignoreCase = true) -> listOf("Get professional biopsy", "Avoid self-medication")
            disease.contains("Infection", ignoreCase = true) -> listOf("Keep area clean", "Avoid spreading")
            disease.contains("Burn", ignoreCase = true) -> listOf("Cool with water", "Cover with clean cloth")
            else -> emptyList()
        }
        
        return baseActions + diseaseSpecificActions
    }

    private fun createDefaultResult(symptoms: String): TriageResult {
        return TriageResult(
            predictedDisease = "Unknown Condition",
            confidence = 0.5f,
            urgencyLevel = UrgencyLevel.YELLOW,
            imageConfidence = 0.5f,
            textSeverity = SeverityLevel.MODERATE,
            textConfidence = 0.5f,
            detectedSymptoms = listOf(symptoms),
            precautions = listOf("Monitor symptoms", "Seek medical advice"),
            recommendedActions = listOf("Consult healthcare provider", "Monitor for changes"),
            timestamp = System.currentTimeMillis()
        )
    }

    // Additional utility methods
    fun updateDiseaseSeverityMapping(disease: String, urgency: UrgencyLevel) {
        // This could be used to dynamically update severity mappings based on user feedback
        Log.d(TAG, "Updated severity mapping: $disease -> $urgency")
    }

    fun getFusionEngineInfo(): String {
        return """
            Fusion Engine Information:
            - Disease Severity Mappings: ${diseaseSeverityMap.size}
            - Precautions Database: ${precautionsMap.size} diseases
            - Image Weight: 70%
            - Text Weight: 30%
            - Conservative Urgency Selection: Enabled
        """.trimIndent()
    }

    fun validateResults(result: TriageResult): Boolean {
        return result.predictedDisease.isNotEmpty() &&
               result.confidence in 0.0f..1.0f &&
               result.precautions.isNotEmpty() &&
               result.recommendedActions.isNotEmpty()
    }
}
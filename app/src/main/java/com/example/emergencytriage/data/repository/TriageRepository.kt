package com.example.emergencytriage.data.repository

import android.content.Context
import android.graphics.Bitmap
import com.example.emergencytriage.data.models.TriageResult
import com.example.emergencytriage.data.models.UrgencyLevel
import com.example.emergencytriage.ml.FusionEngine
import com.example.emergencytriage.ml.ImageClassifier
import com.example.emergencytriage.ml.SpeechNLPProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Repository class that handles data operations for triage analysis
 * Coordinates between different ML components and provides clean API for ViewModels
 */
class TriageRepository(private val context: Context) {

    private val imageClassifier: ImageClassifier by lazy { ImageClassifier(context) }
    private val speechNLPProcessor: SpeechNLPProcessor by lazy { SpeechNLPProcessor(context) }
    private val fusionEngine: FusionEngine by lazy { FusionEngine(context) }

    /**
     * Performs comprehensive triage analysis using multimodal AI
     * @param image Optional medical image for analysis
     * @param symptoms Text description of symptoms
     * @return TriageResult with diagnosis and urgency level
     */
    suspend fun performTriageAnalysis(image: Bitmap?, symptoms: String): TriageResult {
        return withContext(Dispatchers.Default) {
            try {
                // Step 1: Image Analysis (if available)
                val imageResults = image?.let { bitmap ->
                    imageClassifier.classifyImage(bitmap)
                }

                // Step 2: NLP Analysis of symptoms
                val nlpResults = if (symptoms.isNotEmpty()) {
                    speechNLPProcessor.processSymptomsText(symptoms)
                } else {
                    null
                }

                // Step 3: Fusion of multimodal data
                val fusionResult = fusionEngine.fuseResults(imageResults, nlpResults, symptoms)

                // Step 4: Generate comprehensive triage result
                generateTriageResult(fusionResult, imageResults, nlpResults, symptoms)

            } catch (e: Exception) {
                // Return error result
                TriageResult(
                    primaryDiagnosis = "Analysis Error",
                    confidence = 0.0f,
                    urgencyLevel = UrgencyLevel.UNKNOWN,
                    secondaryDiagnoses = emptyList(),
                    recommendations = listOf(
                        "Analysis failed: ${e.message}",
                        "Please try again or consult a healthcare professional"
                    ),
                    riskFactors = emptyList(),
                    estimatedWaitTime = "N/A",
                    followUpInstructions = listOf("Seek immediate medical attention if symptoms worsen")
                )
            }
        }
    }

    /**
     * Generates final triage result from fused analysis
     */
    private fun generateTriageResult(
        fusionResult: Map<String, Any>?,
        imageResults: Map<String, Any>?,
        nlpResults: Map<String, Any>?,
        originalSymptoms: String
    ): TriageResult {
        
        // Extract primary diagnosis from fusion results
        val primaryDiagnosis = fusionResult?.get("primary_diagnosis") as? String 
            ?: imageResults?.get("predicted_class") as? String
            ?: "Symptom Analysis"

        // Extract confidence score
        val confidence = (fusionResult?.get("confidence") as? Number)?.toFloat()
            ?: (imageResults?.get("confidence") as? Number)?.toFloat()
            ?: 0.5f

        // Determine urgency level based on diagnosis and symptoms
        val urgencyLevel = determineUrgencyLevel(primaryDiagnosis, originalSymptoms, confidence)

        // Generate secondary diagnoses
        val secondaryDiagnoses = generateSecondaryDiagnoses(fusionResult, imageResults, nlpResults)

        // Generate recommendations
        val recommendations = generateRecommendations(primaryDiagnosis, urgencyLevel, originalSymptoms)

        // Extract risk factors
        val riskFactors = extractRiskFactors(nlpResults, originalSymptoms)

        // Estimate wait time based on urgency
        val estimatedWaitTime = estimateWaitTime(urgencyLevel)

        // Generate follow-up instructions
        val followUpInstructions = generateFollowUpInstructions(urgencyLevel, primaryDiagnosis)

        return TriageResult(
            primaryDiagnosis = primaryDiagnosis,
            confidence = confidence,
            urgencyLevel = urgencyLevel,
            secondaryDiagnoses = secondaryDiagnoses,
            recommendations = recommendations,
            riskFactors = riskFactors,
            estimatedWaitTime = estimatedWaitTime,
            followUpInstructions = followUpInstructions
        )
    }

    /**
     * Determines urgency level based on diagnosis and symptoms
     */
    private fun determineUrgencyLevel(diagnosis: String, symptoms: String, confidence: Float): UrgencyLevel {
        val lowerDiagnosis = diagnosis.lowercase()
        val lowerSymptoms = symptoms.lowercase()

        // Critical conditions
        val criticalKeywords = listOf(
            "melanoma", "heart attack", "stroke", "severe", "emergency",
            "chest pain", "difficulty breathing", "unconscious"
        )

        // High priority conditions
        val highKeywords = listOf(
            "basal cell carcinoma", "squamous cell carcinoma", "infection",
            "fever", "bleeding", "severe pain"
        )

        // Medium priority conditions
        val mediumKeywords = listOf(
            "dermatofibroma", "benign keratosis", "rash", "lesion"
        )

        return when {
            criticalKeywords.any { lowerDiagnosis.contains(it) || lowerSymptoms.contains(it) } -> 
                UrgencyLevel.CRITICAL
            highKeywords.any { lowerDiagnosis.contains(it) || lowerSymptoms.contains(it) } -> 
                UrgencyLevel.HIGH
            mediumKeywords.any { lowerDiagnosis.contains(it) || lowerSymptoms.contains(it) } -> 
                UrgencyLevel.MEDIUM
            confidence > 0.7f -> UrgencyLevel.LOW
            else -> UrgencyLevel.UNKNOWN
        }
    }

    /**
     * Generates secondary diagnoses from analysis results
     */
    private fun generateSecondaryDiagnoses(
        fusionResult: Map<String, Any>?,
        imageResults: Map<String, Any>?,
        nlpResults: Map<String, Any>?
    ): List<String> {
        val diagnoses = mutableListOf<String>()

        // Add secondary predictions from image analysis
        imageResults?.get("secondary_predictions")?.let { predictions ->
            if (predictions is List<*>) {
                predictions.filterIsInstance<String>().take(2).forEach { diagnoses.add(it) }
            }
        }

        // Add symptom-based possibilities
        nlpResults?.get("possible_conditions")?.let { conditions ->
            if (conditions is List<*>) {
                conditions.filterIsInstance<String>().take(2).forEach { diagnoses.add(it) }
            }
        }

        return diagnoses
    }

    /**
     * Generates recommendations based on diagnosis and urgency
     */
    private fun generateRecommendations(
        diagnosis: String, 
        urgency: UrgencyLevel, 
        symptoms: String
    ): List<String> {
        val recommendations = mutableListOf<String>()

        when (urgency) {
            UrgencyLevel.CRITICAL -> {
                recommendations.add("⚠️ SEEK IMMEDIATE EMERGENCY CARE")
                recommendations.add("Call 911 or go to nearest emergency room")
                recommendations.add("Do not delay treatment")
            }
            UrgencyLevel.HIGH -> {
                recommendations.add("Schedule urgent appointment with healthcare provider")
                recommendations.add("Monitor symptoms closely")
                recommendations.add("Seek immediate care if symptoms worsen")
            }
            UrgencyLevel.MEDIUM -> {
                recommendations.add("Schedule appointment with healthcare provider within 1-2 weeks")
                recommendations.add("Continue monitoring the condition")
                recommendations.add("Take photos to track changes")
            }
            UrgencyLevel.LOW -> {
                recommendations.add("Monitor condition and schedule routine checkup")
                recommendations.add("Keep the area clean and protected")
                recommendations.add("Contact doctor if changes occur")
            }
            UrgencyLevel.UNKNOWN -> {
                recommendations.add("Consult healthcare professional for proper evaluation")
                recommendations.add("Provide detailed symptom history")
                recommendations.add("Consider scheduling diagnostic tests")
            }
        }

        return recommendations
    }

    /**
     * Extracts risk factors from symptoms and analysis
     */
    private fun extractRiskFactors(nlpResults: Map<String, Any>?, symptoms: String): List<String> {
        val riskFactors = mutableListOf<String>()
        val lowerSymptoms = symptoms.lowercase()

        // Common risk factors
        if (lowerSymptoms.contains("smoking") || lowerSymptoms.contains("smoker")) {
            riskFactors.add("Smoking history")
        }
        if (lowerSymptoms.contains("family history")) {
            riskFactors.add("Family history of similar conditions")
        }
        if (lowerSymptoms.contains("sun exposure") || lowerSymptoms.contains("sunburn")) {
            riskFactors.add("Sun exposure")
        }

        // Add NLP extracted risk factors
        nlpResults?.get("risk_factors")?.let { factors ->
            if (factors is List<*>) {
                factors.filterIsInstance<String>().forEach { riskFactors.add(it) }
            }
        }

        return riskFactors
    }

    /**
     * Estimates wait time based on urgency level
     */
    private fun estimateWaitTime(urgency: UrgencyLevel): String {
        return when (urgency) {
            UrgencyLevel.CRITICAL -> "Immediate"
            UrgencyLevel.HIGH -> "< 1 hour"
            UrgencyLevel.MEDIUM -> "2-4 hours"
            UrgencyLevel.LOW -> "Same day"
            UrgencyLevel.UNKNOWN -> "Varies"
        }
    }

    /**
     * Generates follow-up instructions based on diagnosis and urgency
     */
    private fun generateFollowUpInstructions(urgency: UrgencyLevel, diagnosis: String): List<String> {
        val instructions = mutableListOf<String>()

        when (urgency) {
            UrgencyLevel.CRITICAL -> {
                instructions.add("Follow emergency care instructions")
                instructions.add("Take all prescribed medications")
                instructions.add("Attend all follow-up appointments")
            }
            UrgencyLevel.HIGH, UrgencyLevel.MEDIUM -> {
                instructions.add("Follow treatment plan as prescribed")
                instructions.add("Monitor for changes or worsening")
                instructions.add("Schedule follow-up as recommended")
            }
            UrgencyLevel.LOW -> {
                instructions.add("Continue monitoring")
                instructions.add("Maintain good hygiene")
                instructions.add("Schedule routine checkup in 3-6 months")
            }
            UrgencyLevel.UNKNOWN -> {
                instructions.add("Seek professional medical evaluation")
                instructions.add("Keep detailed symptom log")
                instructions.add("Follow up as recommended by healthcare provider")
            }
        }

        return instructions
    }
}
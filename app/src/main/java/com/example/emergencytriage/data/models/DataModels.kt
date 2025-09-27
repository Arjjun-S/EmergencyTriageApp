package com.example.emergencytriage.data.models

import android.os.Parcelable
import kotlinx.parcelize.Parcelize

/**
 * Represents the result of image classification
 */
@Parcelize
data class ImagePrediction(
    val predictedClass: String,
    val confidence: Float,
    val allPredictions: Map<String, Float> = emptyMap()
) : Parcelable

/**
 * Severity levels for symptom analysis
 */
enum class SeverityLevel {
    MILD,
    MODERATE, 
    SEVERE
}

/**
 * Represents the result of text/symptom analysis
 */
@Parcelize
data class TextPrediction(
    val severityLevel: SeverityLevel,
    val confidence: Float,
    val detectedSymptoms: List<String>,
    val processedText: String
) : Parcelable

/**
 * Urgency levels for triage classification
 */
enum class UrgencyLevel {
    GREEN,   // Low urgency
    YELLOW,  // Moderate urgency
    RED      // High urgency
}

/**
 * Complete triage result combining all analysis
 */
@Parcelize
data class TriageResult(
    val predictedDisease: String,
    val confidence: Float,
    val urgencyLevel: UrgencyLevel,
    val imageConfidence: Float,
    val textSeverity: SeverityLevel,
    val textConfidence: Float,
    val detectedSymptoms: List<String>,
    val precautions: List<String>,
    val recommendedActions: List<String>,
    val timestamp: Long
) : Parcelable {
    
    /**
     * Get urgency color for UI display
     */
    fun getUrgencyColor(): Int {
        return when (urgencyLevel) {
            UrgencyLevel.GREEN -> android.graphics.Color.GREEN
            UrgencyLevel.YELLOW -> android.graphics.Color.rgb(255, 165, 0) // Orange
            UrgencyLevel.RED -> android.graphics.Color.RED
        }
    }
    
    /**
     * Get urgency description
     */
    fun getUrgencyDescription(): String {
        return when (urgencyLevel) {
            UrgencyLevel.GREEN -> "Low Priority - Monitor symptoms and seek routine care"
            UrgencyLevel.YELLOW -> "Moderate Priority - Seek medical attention within 24-48 hours"
            UrgencyLevel.RED -> "High Priority - Seek immediate medical attention"
        }
    }
    
    /**
     * Get formatted confidence percentage
     */
    fun getFormattedConfidence(): String {
        return "${(confidence * 100).toInt()}%"
    }
}

/**
 * Disease information from dataset
 */
@Parcelize
data class Disease(
    val name: String,
    val symptoms: List<String>,
    val severity: SeverityLevel,
    val precautions: List<String>
) : Parcelable

/**
 * Individual symptom information
 */
@Parcelize
data class Symptom(
    val name: String,
    val severity: SeverityLevel,
    val relatedDiseases: List<String>
) : Parcelable

/**
 * User session data for tracking analysis history
 */
@Parcelize
data class AnalysisSession(
    val sessionId: String,
    val timestamp: Long,
    val userSymptoms: String,
    val imagePath: String?,
    val result: TriageResult,
    val userFeedback: String? = null
) : Parcelable

/**
 * App configuration and settings
 */
data class AppConfig(
    val modelVersion: String,
    val lastUpdated: Long,
    val enableTelemetry: Boolean,
    val confidenceThreshold: Float,
    val enableDebugMode: Boolean
)

/**
 * Model performance metrics
 */
data class ModelMetrics(
    val accuracy: Float,
    val precision: Float,
    val recall: Float,
    val f1Score: Float,
    val modelSize: Long,
    val inferenceTime: Long
)

/**
 * Telehealth provider information
 */
@Parcelize
data class TelehealthProvider(
    val name: String,
    val packageName: String,
    val webUrl: String,
    val description: String,
    val isAvailable24x7: Boolean,
    val supportedServices: List<String>
) : Parcelable

/**
 * Emergency contact information
 */
@Parcelize
data class EmergencyContact(
    val name: String,
    val phoneNumber: String,
    val type: ContactType,
    val isDefault: Boolean
) : Parcelable

enum class ContactType {
    EMERGENCY_SERVICES,
    FAMILY_DOCTOR,
    SPECIALIST,
    FAMILY_MEMBER,
    OTHER
}
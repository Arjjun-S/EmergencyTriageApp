package com.example.emergencytriage.utils

import android.graphics.Bitmap
import android.util.Patterns
import java.util.regex.Pattern

/**
 * Input validation utilities for the Emergency Triage App
 * Provides comprehensive validation for user inputs and data integrity
 */
object ValidationUtils {

    /**
     * Validates symptom text input
     */
    fun validateSymptomsText(symptoms: String?): ValidationResult {
        return when {
            symptoms.isNullOrBlank() -> ValidationResult(false, "Please describe your symptoms")
            symptoms.length < 5 -> ValidationResult(false, "Please provide more detailed symptoms (at least 5 characters)")
            symptoms.length > 1000 -> ValidationResult(false, "Symptom description is too long (max 1000 characters)")
            containsOnlySpecialCharacters(symptoms) -> ValidationResult(false, "Please provide meaningful symptom description")
            else -> ValidationResult(true, "Valid symptoms")
        }
    }

    /**
     * Validates medical image
     */
    fun validateMedicalImage(bitmap: Bitmap?): ValidationResult {
        return when {
            bitmap == null -> ValidationResult(false, "Please select or capture a medical image")
            bitmap.isRecycled -> ValidationResult(false, "Image is corrupted, please select another")
            bitmap.width < 100 || bitmap.height < 100 -> ValidationResult(false, "Image resolution too low (minimum 100x100)")
            bitmap.width > 4096 || bitmap.height > 4096 -> ValidationResult(false, "Image resolution too high (maximum 4096x4096)")
            else -> ValidationResult(true, "Valid medical image")
        }
    }

    /**
     * Validates analysis readiness
     */
    fun validateAnalysisReadiness(bitmap: Bitmap?, symptoms: String?): ValidationResult {
        val imageValidation = validateMedicalImage(bitmap)
        val symptomsValidation = validateSymptomsText(symptoms)

        return when {
            imageValidation.isValid && symptomsValidation.isValid -> 
                ValidationResult(true, "Ready for comprehensive analysis")
            imageValidation.isValid && !symptomsValidation.isValid -> 
                ValidationResult(true, "Image-only analysis available")
            !imageValidation.isValid && symptomsValidation.isValid -> 
                ValidationResult(true, "Symptoms-only analysis available")
            else -> ValidationResult(false, "Please provide either a medical image or symptom description")
        }
    }

    /**
     * Validates phone number for emergency contacts
     */
    fun validatePhoneNumber(phoneNumber: String?): ValidationResult {
        val phonePattern = Pattern.compile("^[+]?[1-9]\\d{1,14}$")
        return when {
            phoneNumber.isNullOrBlank() -> ValidationResult(false, "Phone number is required")
            !phonePattern.matcher(phoneNumber.replace("\\s|-|\\(|\\)".toRegex(), "")).matches() -> 
                ValidationResult(false, "Invalid phone number format")
            else -> ValidationResult(true, "Valid phone number")
        }
    }

    /**
     * Validates email address
     */
    fun validateEmail(email: String?): ValidationResult {
        return when {
            email.isNullOrBlank() -> ValidationResult(false, "Email address is required")
            !Patterns.EMAIL_ADDRESS.matcher(email).matches() -> 
                ValidationResult(false, "Invalid email address format")
            else -> ValidationResult(true, "Valid email address")
        }
    }

    /**
     * Validates age input
     */
    fun validateAge(age: String?): ValidationResult {
        return try {
            val ageInt = age?.toIntOrNull()
            when {
                age.isNullOrBlank() -> ValidationResult(false, "Age is required")
                ageInt == null -> ValidationResult(false, "Age must be a number")
                ageInt < 0 -> ValidationResult(false, "Age cannot be negative")
                ageInt > 150 -> ValidationResult(false, "Age seems unrealistic")
                else -> ValidationResult(true, "Valid age")
            }
        } catch (e: Exception) {
            ValidationResult(false, "Invalid age format")
        }
    }

    /**
     * Validates medical history text
     */
    fun validateMedicalHistory(history: String?): ValidationResult {
        return when {
            history.isNullOrBlank() -> ValidationResult(true, "Medical history is optional")
            history.length > 2000 -> ValidationResult(false, "Medical history is too long (max 2000 characters)")
            else -> ValidationResult(true, "Valid medical history")
        }
    }

    /**
     * Sanitizes user input to prevent potential security issues
     */
    fun sanitizeInput(input: String?): String {
        return input?.trim()
            ?.replace(Regex("[<>\"'&]"), "")
            ?.take(1000) // Limit length
            ?: ""
    }

    /**
     * Checks if text contains only special characters (no meaningful content)
     */
    private fun containsOnlySpecialCharacters(text: String): Boolean {
        val meaningfulPattern = Pattern.compile("[a-zA-Z0-9]")
        return !meaningfulPattern.matcher(text).find()
    }

    /**
     * Validates symptom severity on a scale of 1-10
     */
    fun validateSeverityScore(severity: String?): ValidationResult {
        return try {
            val score = severity?.toIntOrNull()
            when {
                severity.isNullOrBlank() -> ValidationResult(true, "Severity score is optional")
                score == null -> ValidationResult(false, "Severity must be a number")
                score < 1 -> ValidationResult(false, "Severity score must be at least 1")
                score > 10 -> ValidationResult(false, "Severity score cannot exceed 10")
                else -> ValidationResult(true, "Valid severity score")
            }
        } catch (e: Exception) {
            ValidationResult(false, "Invalid severity score format")
        }
    }

    /**
     * Validates duration of symptoms
     */
    fun validateSymptomDuration(duration: String?): ValidationResult {
        val durationPattern = Pattern.compile("^\\d+\\s*(day|days|hour|hours|minute|minutes|week|weeks|month|months)$", Pattern.CASE_INSENSITIVE)
        return when {
            duration.isNullOrBlank() -> ValidationResult(true, "Duration is optional")
            !durationPattern.matcher(duration.trim()).matches() -> 
                ValidationResult(false, "Duration format: e.g., '3 days', '2 hours', '1 week'")
            else -> ValidationResult(true, "Valid duration")
        }
    }

    /**
     * Comprehensive validation for emergency form data
     */
    fun validateEmergencyForm(
        symptoms: String?,
        severity: String?,
        duration: String?,
        medicalHistory: String?,
        contactPhone: String?,
        contactEmail: String?
    ): List<ValidationResult> {
        return listOf(
            validateSymptomsText(symptoms),
            validateSeverityScore(severity),
            validateSymptomDuration(duration),
            validateMedicalHistory(medicalHistory),
            validatePhoneNumber(contactPhone),
            validateEmail(contactEmail)
        )
    }

    /**
     * Generates user-friendly error messages for validation failures
     */
    fun generateValidationSummary(validationResults: List<ValidationResult>): String {
        val errors = validationResults.filter { !it.isValid }.map { it.message }
        return when {
            errors.isEmpty() -> "All inputs are valid"
            errors.size == 1 -> errors.first()
            else -> "Please fix the following issues:\n${errors.joinToString("\n• ", "• ")}"
        }
    }
}

/**
 * Data class representing validation result
 */
data class ValidationResult(
    val isValid: Boolean,
    val message: String
)
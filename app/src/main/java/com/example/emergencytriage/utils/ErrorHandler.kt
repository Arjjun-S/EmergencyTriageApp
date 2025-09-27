package com.example.emergencytriage.utils

import android.content.Context
import android.util.Log
import com.google.android.material.snackbar.Snackbar
import android.view.View
import androidx.appcompat.app.AlertDialog
import com.example.emergencytriage.R
import java.io.IOException
import java.net.SocketTimeoutException
import java.net.UnknownHostException

/**
 * Comprehensive error handling utilities for the Emergency Triage App
 * Provides consistent error handling, logging, and user feedback mechanisms
 */
object ErrorHandler {

    private const val TAG = "ErrorHandler"

    /**
     * Handles and displays errors with appropriate user feedback
     */
    fun handleError(
        context: Context,
        view: View?,
        error: Throwable,
        userMessage: String? = null,
        showDialog: Boolean = false
    ) {
        // Log the error for debugging
        Log.e(TAG, "Error occurred: ${error.message}", error)

        val errorInfo = categorizeError(error)
        val displayMessage = userMessage ?: errorInfo.userMessage

        if (showDialog) {
            showErrorDialog(context, displayMessage, errorInfo.suggestion)
        } else if (view != null) {
            showErrorSnackbar(view, displayMessage, errorInfo.actionText) {
                // Optional retry action based on error type
                when (errorInfo.type) {
                    ErrorType.NETWORK -> {
                        // Could trigger network retry
                    }
                    ErrorType.PERMISSION -> {
                        // Could open app settings
                    }
                    else -> {
                        // Default action
                    }
                }
            }
        }
    }

    /**
     * Shows a Snackbar with error message and optional action
     */
    private fun showErrorSnackbar(
        view: View,
        message: String,
        actionText: String?,
        action: (() -> Unit)? = null
    ) {
        val snackbar = Snackbar.make(view, message, Snackbar.LENGTH_LONG)
        
        if (actionText != null && action != null) {
            snackbar.setAction(actionText) { action() }
        }
        
        snackbar.show()
    }

    /**
     * Shows an AlertDialog with error details
     */
    private fun showErrorDialog(
        context: Context,
        message: String,
        suggestion: String?
    ) {
        AlertDialog.Builder(context)
            .setTitle("Error")
            .setMessage(buildString {
                append(message)
                if (suggestion != null) {
                    append("\n\nSuggestion: ")
                    append(suggestion)
                }
            })
            .setPositiveButton("OK") { dialog, _ -> dialog.dismiss() }
            .show()
    }

    /**
     * Categorizes errors and provides appropriate user messages
     */
    private fun categorizeError(error: Throwable): ErrorInfo {
        return when (error) {
            is SecurityException -> ErrorInfo(
                type = ErrorType.PERMISSION,
                userMessage = "Permission required for this operation",
                technicalMessage = error.message ?: "Security exception",
                suggestion = "Please grant the necessary permissions in app settings",
                actionText = "Settings"
            )
            
            is IOException, is SocketTimeoutException -> ErrorInfo(
                type = ErrorType.NETWORK,
                userMessage = "Network connection problem",
                technicalMessage = error.message ?: "IO exception",
                suggestion = "Check your internet connection and try again",
                actionText = "Retry"
            )
            
            is UnknownHostException -> ErrorInfo(
                type = ErrorType.NETWORK,
                userMessage = "Unable to connect to server",
                technicalMessage = error.message ?: "Unknown host",
                suggestion = "Check your internet connection",
                actionText = "Retry"
            )
            
            is OutOfMemoryError -> ErrorInfo(
                type = ErrorType.MEMORY,
                userMessage = "Not enough memory to complete operation",
                technicalMessage = error.message ?: "Out of memory",
                suggestion = "Close other apps and try again",
                actionText = null
            )
            
            is IllegalArgumentException -> ErrorInfo(
                type = ErrorType.VALIDATION,
                userMessage = "Invalid input provided",
                technicalMessage = error.message ?: "Illegal argument",
                suggestion = "Please check your input and try again",
                actionText = null
            )
            
            is IllegalStateException -> ErrorInfo(
                type = ErrorType.STATE,
                userMessage = "Operation not available at this time",
                technicalMessage = error.message ?: "Illegal state",
                suggestion = "Please try again later",
                actionText = "Retry"
            )
            
            else -> ErrorInfo(
                type = ErrorType.UNKNOWN,
                userMessage = "An unexpected error occurred",
                technicalMessage = error.message ?: "Unknown error",
                suggestion = "Please try again or contact support if the problem persists",
                actionText = "Retry"
            )
        }
    }

    /**
     * Handles ML model errors specifically
     */
    fun handleMLError(
        context: Context,
        view: View?,
        error: Throwable,
        modelName: String
    ) {
        val message = when (error) {
            is OutOfMemoryError -> "Not enough memory to run $modelName model"
            is IllegalArgumentException -> "Invalid input for $modelName model"
            is IOException -> "$modelName model file not found or corrupted"
            else -> "$modelName model failed to process"
        }
        
        handleError(context, view, error, message, false)
    }

    /**
     * Handles speech recognition errors
     */
    fun handleSpeechError(
        context: Context,
        view: View?,
        errorCode: Int
    ) {
        val message = when (errorCode) {
            android.speech.SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
            android.speech.SpeechRecognizer.ERROR_CLIENT -> "Speech recognition client error"
            android.speech.SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Microphone permission required"
            android.speech.SpeechRecognizer.ERROR_NETWORK -> "Network error during speech recognition"
            android.speech.SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Speech recognition timed out"
            android.speech.SpeechRecognizer.ERROR_NO_MATCH -> "No speech input detected"
            android.speech.SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Speech recognizer is busy"
            android.speech.SpeechRecognizer.ERROR_SERVER -> "Speech recognition server error"
            android.speech.SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input within timeout"
            else -> "Speech recognition failed"
        }
        
        if (view != null) {
            showErrorSnackbar(view, message, "Retry") {
                // Could trigger speech recognition retry
            }
        }
    }

    /**
     * Handles camera/image errors
     */
    fun handleImageError(
        context: Context,
        view: View?,
        error: Throwable
    ) {
        val message = when (error) {
            is SecurityException -> "Camera permission required"
            is IOException -> "Failed to save or load image"
            is OutOfMemoryError -> "Image too large to process"
            else -> "Image processing failed"
        }
        
        handleError(context, view, error, message, false)
    }

    /**
     * Creates a comprehensive error report for debugging
     */
    fun createErrorReport(
        error: Throwable,
        context: String,
        additionalInfo: Map<String, Any> = emptyMap()
    ): String {
        return buildString {
            appendLine("=== ERROR REPORT ===")
            appendLine("Context: $context")
            appendLine("Timestamp: ${System.currentTimeMillis()}")
            appendLine("Error Type: ${error.javaClass.simpleName}")
            appendLine("Message: ${error.message}")
            appendLine("Stack Trace:")
            error.stackTrace.forEach { element ->
                appendLine("  at $element")
            }
            if (additionalInfo.isNotEmpty()) {
                appendLine("Additional Info:")
                additionalInfo.forEach { (key, value) ->
                    appendLine("  $key: $value")
                }
            }
            appendLine("=== END REPORT ===")
        }
    }
}

/**
 * Error categorization enum
 */
enum class ErrorType {
    NETWORK,
    PERMISSION,
    MEMORY,
    VALIDATION,
    STATE,
    ML_MODEL,
    UNKNOWN
}

/**
 * Error information data class
 */
data class ErrorInfo(
    val type: ErrorType,
    val userMessage: String,
    val technicalMessage: String,
    val suggestion: String?,
    val actionText: String?
)
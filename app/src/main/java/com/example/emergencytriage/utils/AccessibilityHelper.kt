package com.example.emergencytriage.utils

import android.accessibilityservice.AccessibilityServiceInfo
import android.content.Context
import android.content.res.Configuration
import android.graphics.Typeface
import android.os.Build
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.view.accessibility.AccessibilityManager
import android.widget.TextView
import androidx.core.view.ViewCompat
import androidx.core.view.accessibility.AccessibilityNodeInfoCompat
import java.util.*

/**
 * Accessibility utilities for Emergency Triage App
 * Provides comprehensive accessibility support including TTS, high contrast, large text, and screen reader support
 */
class AccessibilityHelper(private val context: Context) {

    private var textToSpeech: TextToSpeech? = null
    private var isTextToSpeechReady = false
    private val accessibilityManager = context.getSystemService(Context.ACCESSIBILITY_SERVICE) as AccessibilityManager

    companion object {
        private const val TAG = "AccessibilityHelper"
        private const val TTS_QUEUE_ADD = TextToSpeech.QUEUE_ADD
        private const val TTS_QUEUE_FLUSH = TextToSpeech.QUEUE_FLUSH
    }

    init {
        initializeTextToSpeech()
    }

    /**
     * Initialize Text-to-Speech engine
     */
    private fun initializeTextToSpeech() {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = textToSpeech?.setLanguage(Locale.getDefault())
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.w(TAG, "Language not supported for TTS")
                    // Fallback to English
                    textToSpeech?.setLanguage(Locale.ENGLISH)
                }
                isTextToSpeechReady = true
                Log.d(TAG, "Text-to-Speech initialized successfully")
            } else {
                Log.e(TAG, "Text-to-Speech initialization failed")
            }
        }

        // Set up TTS listener for better control
        textToSpeech?.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {
                Log.d(TAG, "TTS started: $utteranceId")
            }

            override fun onDone(utteranceId: String?) {
                Log.d(TAG, "TTS completed: $utteranceId")
            }

            override fun onError(utteranceId: String?) {
                Log.e(TAG, "TTS error: $utteranceId")
            }
        })
    }

    /**
     * Speaks text using Text-to-Speech
     */
    fun speak(text: String, interrupt: Boolean = false) {
        if (!isTextToSpeechReady) {
            Log.w(TAG, "TTS not ready, cannot speak: $text")
            return
        }

        val queueMode = if (interrupt) TTS_QUEUE_FLUSH else TTS_QUEUE_ADD
        val utteranceId = System.currentTimeMillis().toString()
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            textToSpeech?.speak(text, queueMode, null, utteranceId)
        } else {
            @Suppress("DEPRECATION")
            textToSpeech?.speak(text, queueMode, null)
        }
    }

    /**
     * Stops current speech
     */
    fun stopSpeaking() {
        textToSpeech?.stop()
    }

    /**
     * Checks if accessibility services are enabled
     */
    fun isAccessibilityEnabled(): Boolean {
        return accessibilityManager.isEnabled
    }

    /**
     * Checks if screen reader (TalkBack) is active
     */
    fun isScreenReaderActive(): Boolean {
        val enabledServices = accessibilityManager.getEnabledAccessibilityServiceList(
            AccessibilityServiceInfo.FEEDBACK_SPOKEN
        )
        return enabledServices.isNotEmpty()
    }

    /**
     * Checks if high contrast mode is enabled
     */
    fun isHighContrastModeEnabled(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
            accessibilityManager.isEnabled && 
            context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK == Configuration.UI_MODE_NIGHT_YES
        } else {
            false
        }
    }

    /**
     * Gets current text scaling factor
     */
    fun getTextScalingFactor(): Float {
        return context.resources.configuration.fontScale
    }

    /**
     * Checks if large text is enabled
     */
    fun isLargeTextEnabled(): Boolean {
        return getTextScalingFactor() > 1.15f
    }

    /**
     * Applies accessibility improvements to a view
     */
    fun enhanceViewAccessibility(view: View, contentDescription: String? = null) {
        // Set content description
        contentDescription?.let {
            view.contentDescription = it
        }

        // Make view focusable for screen readers
        ViewCompat.setImportantForAccessibility(view, ViewCompat.IMPORTANT_FOR_ACCESSIBILITY_YES)

        // Add accessibility delegate for custom behavior
        ViewCompat.setAccessibilityDelegate(view, object : androidx.core.view.AccessibilityDelegateCompat() {
            override fun onInitializeAccessibilityNodeInfo(
                host: View,
                info: AccessibilityNodeInfoCompat
            ) {
                super.onInitializeAccessibilityNodeInfo(host, info)
                
                // Add custom actions if needed
                when (view.id) {
                    android.R.id.button -> {
                        info.addAction(AccessibilityNodeInfoCompat.ACTION_CLICK)
                        info.className = "android.widget.Button"
                    }
                }
            }
        })

        // Announce important changes
        if (isScreenReaderActive()) {
            view.setOnFocusChangeListener { _, hasFocus ->
                if (hasFocus && !contentDescription.isNullOrEmpty()) {
                    speak(contentDescription)
                }
            }
        }
    }

    /**
     * Announces important information to screen readers
     */
    fun announceForAccessibility(view: View, message: String) {
        if (isScreenReaderActive()) {
            view.announceForAccessibility(message)
        } else {
            // Fallback to TTS if no screen reader
            speak(message)
        }
    }

    /**
     * Enhances text views for better accessibility
     */
    fun enhanceTextViewAccessibility(textView: TextView) {
        // Adjust text size for large text users
        if (isLargeTextEnabled()) {
            val currentSize = textView.textSize / context.resources.displayMetrics.scaledDensity
            textView.textSize = currentSize * getTextScalingFactor()
        }

        // Improve contrast for high contrast mode
        if (isHighContrastModeEnabled()) {
            textView.setTextColor(context.getColor(android.R.color.white))
            textView.setBackgroundColor(context.getColor(android.R.color.black))
        }

        // Make text more readable
        textView.typeface = Typeface.DEFAULT
        textView.includeFontPadding = true
    }

    /**
     * Creates accessibility-friendly error messages
     */
    fun createAccessibleErrorMessage(error: String, suggestions: List<String>): String {
        return buildString {
            append("Error: ")
            append(error)
            if (suggestions.isNotEmpty()) {
                append(". Suggestions: ")
                append(suggestions.joinToString(", "))
            }
        }
    }

    /**
     * Announces analysis results in an accessible way
     */
    fun announceAnalysisResult(diagnosis: String, urgency: String, confidence: Float) {
        val message = buildString {
            append("Analysis complete. ")
            append("Primary diagnosis: $diagnosis. ")
            append("Urgency level: $urgency. ")
            append("Confidence: ${(confidence * 100).toInt()} percent. ")
            append("Please review the detailed results.")
        }
        speak(message, true)
    }

    /**
     * Provides audio feedback for UI interactions
     */
    fun provideAudioFeedback(action: String) {
        when (action) {
            "image_selected" -> speak("Medical image selected")
            "recording_started" -> speak("Recording started. Please describe your symptoms.")
            "recording_stopped" -> speak("Recording stopped. Processing your speech.")
            "analysis_started" -> speak("Starting medical analysis. Please wait.")
            "analysis_complete" -> speak("Analysis complete. Results are now available.")
            "error_occurred" -> speak("An error occurred. Please check the details.")
        }
    }

    /**
     * Makes emergency call interface more accessible
     */
    fun enhanceEmergencyCallAccessibility(view: View) {
        enhanceViewAccessibility(view, "Emergency call button. Double tap to call emergency services.")
        
        // Add extra emphasis for emergency actions
        view.setOnClickListener {
            speak("Calling emergency services", true)
        }
    }

    /**
     * Provides guided navigation for complex UI
     */
    fun provideNavigationGuidance(currentScreen: String) {
        val guidance = when (currentScreen) {
            "main" -> "Main screen. Use the image buttons to capture or select a medical image. Use the microphone button to record symptoms. Use the analyze button when ready."
            "results" -> "Results screen. Review your analysis results including diagnosis, urgency level, and recommendations."
            "history" -> "History screen. View your previous medical analyses and emergency contacts."
            else -> "Navigation available. Use screen reader gestures to explore."
        }
        speak(guidance)
    }

    /**
     * Cleanup resources
     */
    fun shutdown() {
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        isTextToSpeechReady = false
    }

    /**
     * Adjusts view hierarchy for better accessibility
     */
    fun optimizeViewHierarchyAccessibility(rootView: ViewGroup) {
        fun processView(view: View) {
            when (view) {
                is ViewGroup -> {
                    // Process child views
                    for (i in 0 until view.childCount) {
                        processView(view.getChildAt(i))
                    }
                    
                    // Set proper traversal order if needed
                    if (view.childCount > 1) {
                        for (i in 0 until view.childCount - 1) {
                            val current = view.getChildAt(i)
                            val next = view.getChildAt(i + 1)
                            ViewCompat.setAccessibilityTraversalAfter(next, current.id)
                        }
                    }
                }
                
                is TextView -> {
                    enhanceTextViewAccessibility(view)
                }
                
                else -> {
                    // Ensure interactive elements are properly labeled
                    if (view.isClickable && view.contentDescription.isNullOrEmpty()) {
                        view.contentDescription = "Interactive element"
                    }
                }
            }
        }
        
        processView(rootView)
    }

    /**
     * Creates voice-guided symptom recording
     */
    fun startVoiceGuidedRecording() {
        speak("Starting voice-guided symptom recording. Please describe your symptoms clearly. Include location, severity, duration, and any relevant details.", true)
    }

    /**
     * Provides real-time TTS feedback during recording
     */
    fun provideRecordingFeedback(partialText: String) {
        if (partialText.isNotEmpty() && partialText.length > 10) {
            // Only announce longer partial results to avoid overwhelming feedback
            if (partialText.endsWith(".") || partialText.endsWith("?") || partialText.endsWith("!")) {
                speak("Recorded: $partialText")
            }
        }
    }
}
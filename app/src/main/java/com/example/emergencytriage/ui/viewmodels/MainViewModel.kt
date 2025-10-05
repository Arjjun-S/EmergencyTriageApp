package com.example.emergencytriage.ui.viewmodels

import android.app.Application
import android.graphics.Bitmap
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.viewModelScope
import com.example.emergencytriage.data.models.TriageResult
import com.example.emergencytriage.data.models.UrgencyLevel
import com.example.emergencytriage.data.repository.TriageRepository
import kotlinx.coroutines.launch

/**
 * ViewModel for MainActivity following MVVM architecture pattern
 * Manages UI-related data and business logic for the triage functionality
 */
class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val repository = TriageRepository(application)

    // UI State Management
    private val _isLoading = MutableLiveData<Boolean>(false)
    val isLoading: LiveData<Boolean> = _isLoading

    private val _statusMessage = MutableLiveData<String>("Ready - Select image and record symptoms")
    val statusMessage: LiveData<String> = _statusMessage

    private val _isRecording = MutableLiveData<Boolean>(false)
    val isRecording: LiveData<Boolean> = _isRecording

    private val _transcribedText = MutableLiveData<String>("")
    val transcribedText: LiveData<String> = _transcribedText

    private val _selectedImage = MutableLiveData<Bitmap?>()
    val selectedImage: LiveData<Bitmap?> = _selectedImage

    private val _analysisResult = MutableLiveData<TriageResult?>()
    val analysisResult: LiveData<TriageResult?> = _analysisResult

    private val _errorMessage = MutableLiveData<String?>()
    val errorMessage: LiveData<String?> = _errorMessage

    private val _canAnalyze = MutableLiveData<Boolean>(false)
    val canAnalyze: LiveData<Boolean> = _canAnalyze

    // Animation States
    private val _showImagePreviewAnimation = MutableLiveData<Boolean>(false)
    val showImagePreviewAnimation: LiveData<Boolean> = _showImagePreviewAnimation

    private val _showRecordingAnimation = MutableLiveData<Boolean>(false)
    val showRecordingAnimation: LiveData<Boolean> = _showRecordingAnimation

    /**
     * Updates the selected image and triggers UI updates
     */
    fun setSelectedImage(bitmap: Bitmap?) {
        _selectedImage.value = bitmap
        _showImagePreviewAnimation.value = bitmap != null
        updateAnalysisAvailability()
        updateStatusMessage()
    }

    /**
     * Updates the transcribed text from speech recognition
     */
    fun setTranscribedText(text: String) {
        _transcribedText.value = text
        updateAnalysisAvailability()
        updateStatusMessage()
    }

    /**
     * Starts voice recording
     */
    fun startRecording() {
        _isRecording.value = true
        _showRecordingAnimation.value = true
        _statusMessage.value = "Listening... Describe your symptoms"
    }

    /**
     * Stops voice recording
     */
    fun stopRecording() {
        _isRecording.value = false
        _showRecordingAnimation.value = false
        _statusMessage.value = "Processing speech..."
    }

    /**
     * Performs the triage analysis using repository
     */
    fun performAnalysis() {
        val bitmap = _selectedImage.value
        val symptoms = _transcribedText.value
        
        // Validate inputs using ValidationUtils
        val validationResult = com.example.emergencytriage.utils.ValidationUtils.validateAnalysisReadiness(bitmap, symptoms)
        
        if (!validationResult.isValid) {
            _errorMessage.value = validationResult.message
            return
        }

        viewModelScope.launch {
            try {
                _isLoading.value = true
                _statusMessage.value = "Analyzing with AI..."
                
                val result = repository.performTriageAnalysis(bitmap, symptoms ?: "")
                
                _analysisResult.value = result
                _statusMessage.value = "Analysis complete"
                
            } catch (e: Exception) {
                _errorMessage.value = "Analysis failed: ${e.message}"
                _statusMessage.value = "Analysis failed"
            } finally {
                _isLoading.value = false
            }
        }
    }

    /**
     * Resets the analysis for a new session
     */
    fun resetAnalysis() {
        _selectedImage.value = null
        _transcribedText.value = ""
        _analysisResult.value = null
        _errorMessage.value = null
        _isLoading.value = false
        _isRecording.value = false
        _showImagePreviewAnimation.value = false
        _showRecordingAnimation.value = false
        _statusMessage.value = initialStatus()
        updateAnalysisAvailability()
    }

    /**
     * Clears error messages
     */
    fun clearError() {
        _errorMessage.value = null
    }

    /**
     * Updates whether analysis can be performed based on available data
     */
    private fun updateAnalysisAvailability() {
        val hasImage = _selectedImage.value != null
        val hasText = !_transcribedText.value.isNullOrEmpty()
        _canAnalyze.value = hasImage || hasText
    }

    /**
     * Updates status message based on current state
     */
    private fun updateStatusMessage() {
        val hasImage = _selectedImage.value != null
        val hasText = !_transcribedText.value.isNullOrEmpty()
        
        when {
            hasImage && hasText -> _statusMessage.value = "Ready for analysis"
            hasImage -> _statusMessage.value = "Image selected - Add symptoms or analyze"
            hasText -> _statusMessage.value = "Symptoms recorded - Add image or analyze"
            else -> _statusMessage.value = initialStatus()
        }
    }

    private fun initialStatus(): String {
        val imageOk = repository.isImageModelAvailable()
        val textOk = repository.isTextModelAvailable()
        return if (imageOk && textOk) {
            "Ready - Select image and record symptoms"
        } else {
            "Models missing (using fallback). You can still test the flow."
        }
    }

    /**
     * Handles speech recognition errors
     */
    fun onSpeechRecognitionError(error: String) {
        _isRecording.value = false
        _showRecordingAnimation.value = false
        _errorMessage.value = "Speech recognition error: $error"
        _statusMessage.value = "Speech recognition failed"
    }

    /**
     * Called when speech recognition is ready
     */
    fun onSpeechRecognitionReady() {
        _statusMessage.value = "Listening... Start speaking"
    }

    /**
     * Called when speech recognition begins listening
     */
    fun onSpeechRecognitionBeginning() {
        _statusMessage.value = "Listening..."
    }

    /**
     * Called when speech recognition ends
     */
    fun onSpeechRecognitionEnd() {
        _isRecording.value = false
        _showRecordingAnimation.value = false
        if (_transcribedText.value.isNullOrEmpty()) {
            _statusMessage.value = "No speech detected"
        } else {
            updateStatusMessage()
        }
    }
}
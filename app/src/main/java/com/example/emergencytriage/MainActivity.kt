package com.example.emergencytriage

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Observer
import com.example.emergencytriage.data.models.UrgencyLevel
import com.example.emergencytriage.ui.screens.ResultScreen
import com.example.emergencytriage.ui.viewmodels.MainViewModel
import com.example.emergencytriage.utils.PermissionsHelper
import com.example.emergencytriage.data.repository.TriageRepository
import com.google.android.material.button.MaterialButton
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton
import com.google.android.material.imageview.ShapeableImageView
import com.google.android.material.snackbar.Snackbar
import com.google.android.material.textfield.TextInputEditText
import java.io.InputStream

class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_RECORD_AUDIO_PERMISSION = 200
        private const val REQUEST_CAMERA_PERMISSION = 201
    }

    // ViewModel using modern Android architecture
    private val viewModel: MainViewModel by viewModels()

    // UI Components (Material Design 3)
    private lateinit var btnStartRecording: MaterialButton
    private lateinit var btnStopRecording: MaterialButton
    private lateinit var btnSelectImage: MaterialButton
    private lateinit var btnCaptureImage: MaterialButton
    private lateinit var btnAnalyze: MaterialButton
    private lateinit var imgPreview: ShapeableImageView
    private lateinit var tvTranscribedText: TextInputEditText
    private lateinit var tvStatus: android.widget.TextView
    private lateinit var fabEmergency: ExtendedFloatingActionButton
    private lateinit var emptyImageOverlay: View

    // Speech Recognition
    private var speechRecognizer: SpeechRecognizer? = null

    // Activity Result Launchers
    private val imagePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleImageSelection(it) }
    }

    private val cameraLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicturePreview()
    ) { bitmap: Bitmap? ->
        bitmap?.let { handleCapturedImage(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        initializeViews()
        setupObservers()
        setupListeners()
        requestPermissions()

        // Show environment status at startup if models are missing
        val repo = TriageRepository(applicationContext)
        val imageOk = repo.isImageModelAvailable()
        val textOk = repo.isTextModelAvailable()
        if (!(imageOk && textOk)) {
            Snackbar.make(
                findViewById(android.R.id.content),
                "ML models not found. Using safe fallback.",
                Snackbar.LENGTH_LONG
            ).show()
            tvStatus.text = "Models missing (fallback active)."
        }
    }

    private fun initializeViews() {
        btnStartRecording = findViewById(R.id.btn_start_recording)
        btnStopRecording = findViewById(R.id.btn_stop_recording)
        btnSelectImage = findViewById(R.id.btn_select_image)
        btnCaptureImage = findViewById(R.id.btn_capture_image)
        btnAnalyze = findViewById(R.id.btn_analyze)
        imgPreview = findViewById(R.id.img_preview)
        tvTranscribedText = findViewById(R.id.tv_transcribed_text)
        tvStatus = findViewById(R.id.tv_status)
        fabEmergency = findViewById(R.id.fab_emergency)
        emptyImageOverlay = findViewById(R.id.empty_image_overlay)

        // Setup toolbar
        setSupportActionBar(findViewById(R.id.toolbar))
    }

    /**
     * Setup observers for ViewModel LiveData
     */
    private fun setupObservers() {
        // Observe loading state
        viewModel.isLoading.observe(this) { isLoading ->
            btnAnalyze.isEnabled = !isLoading && (viewModel.canAnalyze.value == true)
            if (isLoading) {
                btnAnalyze.text = "Analyzing..."
            } else {
                btnAnalyze.text = "Analyze Symptoms"
            }
        }

        // Observe status messages
        viewModel.statusMessage.observe(this) { message ->
            tvStatus.text = message
        }

        // Observe recording state
        viewModel.isRecording.observe(this) { isRecording ->
            btnStartRecording.isEnabled = !isRecording
            btnStopRecording.isEnabled = isRecording
            
            if (isRecording) {
                btnStartRecording.text = "Recording..."
                btnStartRecording.setIconResource(android.R.drawable.ic_media_pause)
            } else {
                btnStartRecording.text = "Record"
                btnStartRecording.setIconResource(android.R.drawable.ic_btn_speak_now)
            }
        }

        // Observe transcribed text
        viewModel.transcribedText.observe(this) { text ->
            tvTranscribedText.setText(text)
        }

        // Observe selected image
        viewModel.selectedImage.observe(this) { bitmap ->
            if (bitmap != null) {
                imgPreview.setImageBitmap(bitmap)
                emptyImageOverlay.visibility = View.GONE
            } else {
                imgPreview.setImageResource(android.R.drawable.ic_menu_camera)
                emptyImageOverlay.visibility = View.VISIBLE
            }
        }

        // Observe analysis capability
        viewModel.canAnalyze.observe(this) { canAnalyze ->
            btnAnalyze.isEnabled = canAnalyze && (viewModel.isLoading.value == false)
        }

        // Observe analysis results
        viewModel.analysisResult.observe(this) { result ->
            result?.let {
                navigateToResults(it)
            }
        }

        // Observe error messages
        viewModel.errorMessage.observe(this) { error ->
            error?.let {
                Snackbar.make(findViewById(android.R.id.content), it, Snackbar.LENGTH_LONG)
                    .setAction("Dismiss") { viewModel.clearError() }
                    .show()
            }
        }

        // Observe animation states
        viewModel.showImagePreviewAnimation.observe(this) { showAnimation ->
            if (showAnimation) {
                // Add image preview animation
                imgPreview.animate()
                    .scaleX(1.1f)
                    .scaleY(1.1f)
                    .setDuration(200)
                    .withEndAction {
                        imgPreview.animate()
                            .scaleX(1.0f)
                            .scaleY(1.0f)
                            .setDuration(200)
                            .start()
                    }
                    .start()
            }
        }
    }

    private fun setupListeners() {
        btnStartRecording.setOnClickListener { startVoiceRecording() }
        btnStopRecording.setOnClickListener { stopVoiceRecording() }
        btnSelectImage.setOnClickListener { selectImageFromGallery() }
        btnCaptureImage.setOnClickListener { captureImageFromCamera() }
        btnAnalyze.setOnClickListener { viewModel.performAnalysis() }
        
        // Emergency FAB
        fabEmergency.setOnClickListener { makeEmergencyCall() }
        
        // Text input listener for manual symptom entry
        tvTranscribedText.setOnFocusChangeListener { _, hasFocus ->
            if (!hasFocus) {
                val text = tvTranscribedText.text?.toString() ?: ""
                viewModel.setTranscribedText(text)
            }
        }
    }

    private fun requestPermissions() {
        val permissions = mutableListOf<String>()
        
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) 
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.RECORD_AUDIO)
        }
        
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
            != PackageManager.PERMISSION_GRANTED) {
            permissions.add(Manifest.permission.CAMERA)
        }

        if (permissions.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, permissions.toTypedArray(), REQUEST_RECORD_AUDIO_PERMISSION)
        }
    }

    private fun startVoiceRecording() {
        if (!PermissionsHelper.hasAudioPermission(this)) {
            Toast.makeText(this, "Microphone permission required", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            viewModel.startRecording()
            speechRecognizer = SpeechRecognizer.createSpeechRecognizer(this)
            speechRecognizer?.setRecognitionListener(speechRecognitionListener)

            val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "en-US")
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
            }

            speechRecognizer?.startListening(intent)
            btnStartRecording.isEnabled = false
            btnStopRecording.isEnabled = true
            updateStatusUi("Listening... Describe your symptoms")
        } catch (e: Exception) {
            Log.e(TAG, "Error starting speech recognition", e)
            updateStatusUi("Error starting voice recording")
        }
    }

    private fun stopVoiceRecording() {
        speechRecognizer?.stopListening()
        viewModel.stopRecording()
    }

    private val speechRecognitionListener = object : RecognitionListener {
        override fun onReadyForSpeech(params: Bundle?) {
            viewModel.onSpeechRecognitionReady()
        }

        override fun onBeginningOfSpeech() {
            viewModel.onSpeechRecognitionBeginning()
        }

        override fun onRmsChanged(rmsdB: Float) {}
        override fun onBufferReceived(buffer: ByteArray?) {}
        
        override fun onEndOfSpeech() {
            viewModel.onSpeechRecognitionEnd()
        }
        
        override fun onError(error: Int) {
            val errorMessage = when (error) {
                SpeechRecognizer.ERROR_AUDIO -> "Audio recording error"
                SpeechRecognizer.ERROR_CLIENT -> "Client side error"
                SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Insufficient permissions"
                SpeechRecognizer.ERROR_NETWORK -> "Network error"
                SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Network timeout"
                SpeechRecognizer.ERROR_NO_MATCH -> "No speech input detected"
                SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Recognition service busy"
                SpeechRecognizer.ERROR_SERVER -> "Server error"
                SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "No speech input"
                else -> "Unknown error"
            }
            viewModel.onSpeechRecognitionError(errorMessage)
        }

        override fun onResults(results: Bundle?) {
            results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.let { matches ->
                if (matches.isNotEmpty()) {
                    viewModel.setTranscribedText(matches[0])
                }
            }
        }

        override fun onPartialResults(partialResults: Bundle?) {
            partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.let { matches ->
                if (matches.isNotEmpty()) {
                    // Show partial results while speaking
                    tvTranscribedText.setText(matches[0])
                }
            }
        }

        override fun onEvent(eventType: Int, params: Bundle?) {}
    }

    private fun selectImageFromGallery() {
        imagePickerLauncher.launch("image/*")
    }

    private fun captureImageFromCamera() {
        if (!PermissionsHelper.hasCameraPermission(this)) {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
            return
        }
        cameraLauncher.launch(null)
    }

    private fun handleImageSelection(uri: Uri) {
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            viewModel.setSelectedImage(bitmap)
        } catch (e: Exception) {
            Log.e(TAG, "Error loading image", e)
            Snackbar.make(findViewById(android.R.id.content), 
                "Error loading image: ${e.message}", 
                Snackbar.LENGTH_LONG).show()
        }
    }

    private fun handleCapturedImage(bitmap: Bitmap) {
        viewModel.setSelectedImage(bitmap)
    }

    override fun onDestroy() {
        super.onDestroy()
        speechRecognizer?.destroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            REQUEST_RECORD_AUDIO_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Snackbar.make(findViewById(android.R.id.content), 
                        "Microphone permission granted", 
                        Snackbar.LENGTH_SHORT).show()
                } else {
                    Snackbar.make(findViewById(android.R.id.content), 
                        "Microphone permission is required for symptom recording", 
                        Snackbar.LENGTH_LONG).show()
                }
            }
            REQUEST_CAMERA_PERMISSION -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Snackbar.make(findViewById(android.R.id.content), 
                        "Camera permission granted", 
                        Snackbar.LENGTH_SHORT).show()
                } else {
                    Snackbar.make(findViewById(android.R.id.content), 
                        "Camera permission is required for image capture", 
                        Snackbar.LENGTH_LONG).show()
                }
            }
        }
    }

    /**
     * Emergency call functionality
     */
    private fun makeEmergencyCall() {
        val intent = Intent(Intent.ACTION_DIAL).apply {
            data = Uri.parse("tel:911")
        }
        if (intent.resolveActivity(packageManager) != null) {
            startActivity(intent)
        } else {
            Toast.makeText(this, "No dialer app available", Toast.LENGTH_SHORT).show()
        }
    }

    /**
     * Navigate to results screen with analysis
     */
    private fun navigateToResults(result: com.example.emergencytriage.data.models.TriageResult) {
        val intent = Intent(this, ResultScreen::class.java).apply {
            putExtra("triage_result", result)
        }
        startActivity(intent)
    }

    private fun updateStatusUi(message: String) {
        tvStatus.text = message
    }
}
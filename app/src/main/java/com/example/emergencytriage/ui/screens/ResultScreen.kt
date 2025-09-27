package com.example.emergencytriage.ui.screens

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.example.emergencytriage.R
import com.example.emergencytriage.data.models.TriageResult
import com.example.emergencytriage.data.models.UrgencyLevel

class ResultScreen : AppCompatActivity() {

    private lateinit var imgResult: ImageView
    private lateinit var tvDisease: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvUrgencyLevel: TextView
    private lateinit var tvSymptoms: TextView
    private lateinit var tvPrecautions: TextView
    private lateinit var btnCallDoctor: Button
    private lateinit var btnTelehealth: Button
    private lateinit var btnBackToHome: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        initializeViews()
        displayResults()
        setupListeners()
    }

    private fun initializeViews() {
        imgResult = findViewById(R.id.img_result)
        tvDisease = findViewById(R.id.tv_disease)
        tvConfidence = findViewById(R.id.tv_confidence)
        tvUrgencyLevel = findViewById(R.id.tv_urgency_level)
        tvSymptoms = findViewById(R.id.tv_symptoms)
        tvPrecautions = findViewById(R.id.tv_precautions)
        btnCallDoctor = findViewById(R.id.btn_call_doctor)
        btnTelehealth = findViewById(R.id.btn_telehealth)
        btnBackToHome = findViewById(R.id.btn_back_to_home)
    }

    private fun displayResults() {
        // Get data from intent
        val triageResult = intent.getParcelableExtra<TriageResult>("triage_result")
        val originalImage = intent.getParcelableExtra<Bitmap>("original_image")
        val symptomsText = intent.getStringExtra("symptoms_text") ?: ""

        // Display image
        originalImage?.let { imgResult.setImageBitmap(it) }

        // Display results
        triageResult?.let { result ->
            tvDisease.text = "Predicted Condition: ${result.predictedDisease}"
            tvConfidence.text = "Confidence: ${String.format("%.1f", result.confidence * 100)}%"
            tvSymptoms.text = "Reported Symptoms: $symptomsText"
            
            // Display urgency level with color coding
            displayUrgencyLevel(result.urgencyLevel)
            
            // Display precautions
            if (result.precautions.isNotEmpty()) {
                val precautionsText = result.precautions.joinToString("\n• ", "• ")
                tvPrecautions.text = "Recommended Precautions:\n$precautionsText"
            } else {
                tvPrecautions.text = "No specific precautions available."
            }
        }
    }

    private fun displayUrgencyLevel(urgencyLevel: UrgencyLevel) {
        when (urgencyLevel) {
            UrgencyLevel.GREEN -> {
                tvUrgencyLevel.text = "Urgency Level: LOW (Green)"
                tvUrgencyLevel.setTextColor(ContextCompat.getColor(this, android.R.color.holo_green_dark))
                tvUrgencyLevel.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
            }
            UrgencyLevel.YELLOW -> {
                tvUrgencyLevel.text = "Urgency Level: MODERATE (Yellow)"
                tvUrgencyLevel.setTextColor(ContextCompat.getColor(this, android.R.color.holo_orange_dark))
                tvUrgencyLevel.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_orange_light))
            }
            UrgencyLevel.RED -> {
                tvUrgencyLevel.text = "Urgency Level: HIGH (Red)"
                tvUrgencyLevel.setTextColor(ContextCompat.getColor(this, android.R.color.holo_red_dark))
                tvUrgencyLevel.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_red_light))
            }
        }
        tvUrgencyLevel.setPadding(16, 8, 16, 8)
    }

    private fun setupListeners() {
        btnCallDoctor.setOnClickListener {
            // Intent to make a phone call
            val intent = Intent(Intent.ACTION_DIAL).apply {
                data = Uri.parse("tel:911") // Emergency number
            }
            if (intent.resolveActivity(packageManager) != null) {
                startActivity(intent)
            }
        }

        btnTelehealth.setOnClickListener {
            // Navigate to telehealth screen
            val intent = Intent(this, TelehealthScreen::class.java)
            startActivity(intent)
        }

        btnBackToHome.setOnClickListener {
            // Go back to main activity
            finish()
        }
    }

    companion object {
        private const val TAG = "ResultScreen"
    }
}
package com.example.emergencytriage.ui.screens

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.emergencytriage.R

class TelehealthScreen : AppCompatActivity() {

    private lateinit var tvTitle: TextView
    private lateinit var tvDescription: TextView
    private lateinit var tvServices: TextView
    private lateinit var btnConnectTeladoc: Button
    private lateinit var btnConnectAmwell: Button
    private lateinit var btnConnectDoctorOnDemand: Button
    private lateinit var btnConnectMDLive: Button
    private lateinit var btnEmergencyCall: Button
    private lateinit var btnBackToResults: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_telehealth)

        initializeViews()
        setupContent()
        setupListeners()
    }

    private fun initializeViews() {
        tvTitle = findViewById(R.id.tv_title)
        tvDescription = findViewById(R.id.tv_description)
        tvServices = findViewById(R.id.tv_services)
        btnConnectTeladoc = findViewById(R.id.btn_connect_teladoc)
        btnConnectAmwell = findViewById(R.id.btn_connect_amwell)
        btnConnectDoctorOnDemand = findViewById(R.id.btn_connect_doctor_on_demand)
        btnConnectMDLive = findViewById(R.id.btn_connect_mdlive)
        btnEmergencyCall = findViewById(R.id.btn_emergency_call)
        btnBackToResults = findViewById(R.id.btn_back_to_results)
    }

    private fun setupContent() {
        tvTitle.text = "Connect with Healthcare Professionals"
        
        tvDescription.text = """
            Based on your triage results, you can connect with licensed healthcare professionals 
            through these telehealth services for immediate consultation and guidance.
        """.trimIndent()

        tvServices.text = """
            Available Services:
            • Video consultations with licensed doctors
            • 24/7 medical support
            • Prescription services (where applicable)
            • Follow-up care recommendations
            • Specialist referrals
            
            Note: For life-threatening emergencies, always call 911 immediately.
        """.trimIndent()
    }

    private fun setupListeners() {
        btnConnectTeladoc.setOnClickListener {
            openTelehealthApp("com.teladoc.teladocapp", "https://www.teladoc.com/")
        }

        btnConnectAmwell.setOnClickListener {
            openTelehealthApp("com.amwell.android.amwell", "https://amwell.com/")
        }

        btnConnectDoctorOnDemand.setOnClickListener {
            openTelehealthApp("com.doctorondemand.mobile", "https://www.doctorondemand.com/")
        }

        btnConnectMDLive.setOnClickListener {
            openTelehealthApp("com.mdlive.mobile", "https://www.mdlive.com/")
        }

        btnEmergencyCall.setOnClickListener {
            // Emergency call intent
            val intent = Intent(Intent.ACTION_CALL).apply {
                data = Uri.parse("tel:911")
            }
            if (intent.resolveActivity(packageManager) != null) {
                startActivity(intent)
            } else {
                // Fallback to dialer
                val dialIntent = Intent(Intent.ACTION_DIAL).apply {
                    data = Uri.parse("tel:911")
                }
                startActivity(dialIntent)
            }
        }

        btnBackToResults.setOnClickListener {
            finish()
        }
    }

    private fun openTelehealthApp(packageName: String, webUrl: String) {
        try {
            // Try to open the app if installed
            val intent = packageManager.getLaunchIntentForPackage(packageName)
            if (intent != null) {
                startActivity(intent)
            } else {
                // If app is not installed, try to open Play Store
                openPlayStore(packageName, webUrl)
            }
        } catch (e: Exception) {
            // Fallback to web browser
            openWebBrowser(webUrl)
        }
    }

    private fun openPlayStore(packageName: String, fallbackUrl: String) {
        try {
            val intent = Intent(Intent.ACTION_VIEW).apply {
                data = Uri.parse("market://details?id=$packageName")
            }
            startActivity(intent)
        } catch (e: Exception) {
            // If Play Store is not available, open web browser
            openWebBrowser(fallbackUrl)
        }
    }

    private fun openWebBrowser(url: String) {
        val intent = Intent(Intent.ACTION_VIEW).apply {
            data = Uri.parse(url)
        }
        if (intent.resolveActivity(packageManager) != null) {
            startActivity(intent)
        }
    }

    companion object {
        private const val TAG = "TelehealthScreen"
    }
}
package com.example.emergencytriage.data.cache

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import androidx.room.*
import com.example.emergencytriage.data.models.TriageResult
import com.example.emergencytriage.data.models.UrgencyLevel
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*

/**
 * Offline support system for Emergency Triage App
 * Provides local data caching, offline analysis storage, and sync capabilities
 */

// Room Database Entities
@Entity(tableName = "offline_analyses")
data class OfflineAnalysis(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val symptomsText: String,
    val imagePath: String?,
    val analysisResult: String, // JSON serialized TriageResult
    val isSynced: Boolean = false,
    val urgencyLevel: String,
    val confidence: Float
)

@Entity(tableName = "emergency_contacts")
data class EmergencyContact(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val name: String,
    val phone: String,
    val email: String?,
    val relationship: String,
    val isPrimary: Boolean = false
)

@Entity(tableName = "medical_history")
data class MedicalHistoryEntry(
    @PrimaryKey val id: String = UUID.randomUUID().toString(),
    val timestamp: Long = System.currentTimeMillis(),
    val condition: String,
    val description: String,
    val severity: Int, // 1-10 scale
    val medications: String?,
    val allergies: String?
)

// Room DAOs
@Dao
interface OfflineAnalysisDao {
    @Query("SELECT * FROM offline_analyses ORDER BY timestamp DESC")
    suspend fun getAllAnalyses(): List<OfflineAnalysis>

    @Query("SELECT * FROM offline_analyses WHERE isSynced = 0")
    suspend fun getUnsyncedAnalyses(): List<OfflineAnalysis>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertAnalysis(analysis: OfflineAnalysis)

    @Update
    suspend fun updateAnalysis(analysis: OfflineAnalysis)

    @Delete
    suspend fun deleteAnalysis(analysis: OfflineAnalysis)

    @Query("UPDATE offline_analyses SET isSynced = 1 WHERE id = :id")
    suspend fun markAsSynced(id: String)
}

@Dao
interface EmergencyContactDao {
    @Query("SELECT * FROM emergency_contacts ORDER BY isPrimary DESC, name ASC")
    suspend fun getAllContacts(): List<EmergencyContact>

    @Query("SELECT * FROM emergency_contacts WHERE isPrimary = 1 LIMIT 1")
    suspend fun getPrimaryContact(): EmergencyContact?

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertContact(contact: EmergencyContact)

    @Update
    suspend fun updateContact(contact: EmergencyContact)

    @Delete
    suspend fun deleteContact(contact: EmergencyContact)
}

@Dao
interface MedicalHistoryDao {
    @Query("SELECT * FROM medical_history ORDER BY timestamp DESC")
    suspend fun getAllHistory(): List<MedicalHistoryEntry>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertHistoryEntry(entry: MedicalHistoryEntry)

    @Update
    suspend fun updateHistoryEntry(entry: MedicalHistoryEntry)

    @Delete
    suspend fun deleteHistoryEntry(entry: MedicalHistoryEntry)
}

// Room Database
@Database(
    entities = [OfflineAnalysis::class, EmergencyContact::class, MedicalHistoryEntry::class],
    version = 1,
    exportSchema = false
)
@TypeConverters(Converters::class)
abstract class TriageDatabase : RoomDatabase() {
    abstract fun offlineAnalysisDao(): OfflineAnalysisDao
    abstract fun emergencyContactDao(): EmergencyContactDao
    abstract fun medicalHistoryDao(): MedicalHistoryDao

    companion object {
        @Volatile
        private var INSTANCE: TriageDatabase? = null

        fun getDatabase(context: Context): TriageDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    TriageDatabase::class.java,
                    "triage_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}

// Type Converters for Room
class Converters {
    @TypeConverter
    fun fromStringList(value: List<String>): String {
        return Gson().toJson(value)
    }

    @TypeConverter
    fun toStringList(value: String): List<String> {
        return try {
            Gson().fromJson(value, object : TypeToken<List<String>>() {}.type)
        } catch (e: Exception) {
            emptyList()
        }
    }
}

/**
 * Offline Cache Manager
 * Handles local storage, image caching, and data synchronization
 */
class OfflineCacheManager(private val context: Context) {

    private val database = TriageDatabase.getDatabase(context)
    private val gson = Gson()
    private val imagesCacheDir = File(context.cacheDir, "medical_images")

    companion object {
        private const val TAG = "OfflineCacheManager"
        private const val MAX_CACHE_SIZE_MB = 100
        private const val MAX_IMAGES = 50
    }

    init {
        // Create images cache directory
        if (!imagesCacheDir.exists()) {
            imagesCacheDir.mkdirs()
        }
    }

    /**
     * Saves analysis result for offline access
     */
    suspend fun saveAnalysisOffline(
        symptomsText: String,
        image: Bitmap?,
        result: TriageResult
    ): String = withContext(Dispatchers.IO) {
        try {
            val analysisId = UUID.randomUUID().toString()
            
            // Save image if provided
            val imagePath = image?.let { bitmap ->
                saveImageToCache(bitmap, analysisId)
            }

            // Serialize analysis result
            val resultJson = gson.toJson(result)

            // Create offline analysis record
            val offlineAnalysis = OfflineAnalysis(
                id = analysisId,
                symptomsText = symptomsText,
                imagePath = imagePath,
                analysisResult = resultJson,
                urgencyLevel = result.urgencyLevel.name,
                confidence = result.confidence
            )

            // Save to database
            database.offlineAnalysisDao().insertAnalysis(offlineAnalysis)
            
            Log.d(TAG, "Analysis saved offline with ID: $analysisId")
            analysisId
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save analysis offline", e)
            throw e
        }
    }

    /**
     * Retrieves all offline analyses
     */
    suspend fun getOfflineAnalyses(): List<OfflineAnalysisResult> = withContext(Dispatchers.IO) {
        try {
            val analyses = database.offlineAnalysisDao().getAllAnalyses()
            analyses.map { analysis ->
                val result = gson.fromJson(analysis.analysisResult, TriageResult::class.java)
                val image = analysis.imagePath?.let { path ->
                    loadImageFromCache(path)
                }
                
                OfflineAnalysisResult(
                    id = analysis.id,
                    timestamp = analysis.timestamp,
                    symptomsText = analysis.symptomsText,
                    image = image,
                    result = result,
                    isSynced = analysis.isSynced
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve offline analyses", e)
            emptyList()
        }
    }

    /**
     * Saves image to cache directory
     */
    private suspend fun saveImageToCache(bitmap: Bitmap, analysisId: String): String = withContext(Dispatchers.IO) {
        try {
            val filename = "${analysisId}_image.jpg"
            val file = File(imagesCacheDir, filename)
            
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            
            // Clean up old images if cache is too large
            cleanupCache()
            
            file.absolutePath
        } catch (e: IOException) {
            Log.e(TAG, "Failed to save image to cache", e)
            throw e
        }
    }

    /**
     * Loads image from cache
     */
    private suspend fun loadImageFromCache(imagePath: String): Bitmap? = withContext(Dispatchers.IO) {
        try {
            val file = File(imagePath)
            if (file.exists()) {
                BitmapFactory.decodeFile(imagePath)
            } else {
                Log.w(TAG, "Cached image not found: $imagePath")
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load image from cache", e)
            null
        }
    }

    /**
     * Cleans up cache to maintain size limits
     */
    private fun cleanupCache() {
        try {
            val files = imagesCacheDir.listFiles() ?: return
            
            // Sort by last modified (oldest first)
            files.sortBy { it.lastModified() }
            
            // Calculate total size
            val totalSize = files.sumOf { it.length() } / (1024 * 1024) // MB
            
            // Remove oldest files if cache is too large or has too many files
            var currentSize = totalSize
            var fileCount = files.size
            
            for (file in files) {
                if (currentSize <= MAX_CACHE_SIZE_MB && fileCount <= MAX_IMAGES) {
                    break
                }
                
                val fileSize = file.length() / (1024 * 1024)
                if (file.delete()) {
                    currentSize -= fileSize
                    fileCount--
                    Log.d(TAG, "Deleted cached file: ${file.name}")
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to cleanup cache", e)
        }
    }

    /**
     * Saves emergency contact
     */
    suspend fun saveEmergencyContact(contact: EmergencyContact): String = withContext(Dispatchers.IO) {
        try {
            database.emergencyContactDao().insertContact(contact)
            Log.d(TAG, "Emergency contact saved: ${contact.name}")
            contact.id
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save emergency contact", e)
            throw e
        }
    }

    /**
     * Gets all emergency contacts
     */
    suspend fun getEmergencyContacts(): List<EmergencyContact> = withContext(Dispatchers.IO) {
        try {
            database.emergencyContactDao().getAllContacts()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve emergency contacts", e)
            emptyList()
        }
    }

    /**
     * Gets primary emergency contact
     */
    suspend fun getPrimaryEmergencyContact(): EmergencyContact? = withContext(Dispatchers.IO) {
        try {
            database.emergencyContactDao().getPrimaryContact()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve primary emergency contact", e)
            null
        }
    }

    /**
     * Saves medical history entry
     */
    suspend fun saveMedicalHistory(entry: MedicalHistoryEntry): String = withContext(Dispatchers.IO) {
        try {
            database.medicalHistoryDao().insertHistoryEntry(entry)
            Log.d(TAG, "Medical history saved: ${entry.condition}")
            entry.id
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save medical history", e)
            throw e
        }
    }

    /**
     * Gets medical history
     */
    suspend fun getMedicalHistory(): List<MedicalHistoryEntry> = withContext(Dispatchers.IO) {
        try {
            database.medicalHistoryDao().getAllHistory()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to retrieve medical history", e)
            emptyList()
        }
    }

    /**
     * Clears all offline data
     */
    suspend fun clearAllOfflineData() = withContext(Dispatchers.IO) {
        try {
            // Clear database
            database.clearAllTables()
            
            // Clear cached images
            imagesCacheDir.listFiles()?.forEach { it.delete() }
            
            Log.d(TAG, "All offline data cleared")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to clear offline data", e)
            throw e
        }
    }

    /**
     * Gets cache statistics
     */
    suspend fun getCacheStats(): CacheStats = withContext(Dispatchers.IO) {
        try {
            val analysesCount = database.offlineAnalysisDao().getAllAnalyses().size
            val unsyncedCount = database.offlineAnalysisDao().getUnsyncedAnalyses().size
            val contactsCount = database.emergencyContactDao().getAllContacts().size
            val historyCount = database.medicalHistoryDao().getAllHistory().size
            
            val cacheFiles = imagesCacheDir.listFiles() ?: emptyArray()
            val cacheSizeMB = cacheFiles.sumOf { it.length() } / (1024 * 1024)
            
            CacheStats(
                totalAnalyses = analysesCount,
                unsyncedAnalyses = unsyncedCount,
                emergencyContacts = contactsCount,
                medicalHistoryEntries = historyCount,
                cachedImages = cacheFiles.size,
                cacheSizeMB = cacheSizeMB.toInt()
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to get cache stats", e)
            CacheStats()
        }
    }
}

/**
 * Data classes for offline support
 */
data class OfflineAnalysisResult(
    val id: String,
    val timestamp: Long,
    val symptomsText: String,
    val image: Bitmap?,
    val result: TriageResult,
    val isSynced: Boolean
)

data class CacheStats(
    val totalAnalyses: Int = 0,
    val unsyncedAnalyses: Int = 0,
    val emergencyContacts: Int = 0,
    val medicalHistoryEntries: Int = 0,
    val cachedImages: Int = 0,
    val cacheSizeMB: Int = 0
)
package com.example.emergencytriage.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import java.io.*
import java.text.SimpleDateFormat
import java.util.*

/**
 * Utility class for file operations and image processing
 */
object FileUtils {
    
    private const val TAG = "FileUtils"
    private const val IMAGE_QUALITY = 85
    private const val MAX_IMAGE_SIZE = 1024
    
    /**
     * Save bitmap to internal storage
     */
    fun saveBitmapToInternalStorage(
        context: Context,
        bitmap: Bitmap,
        fileName: String
    ): String? {
        return try {
            val file = File(context.filesDir, fileName)
            val outputStream = FileOutputStream(file)
            bitmap.compress(Bitmap.CompressFormat.JPEG, IMAGE_QUALITY, outputStream)
            outputStream.close()
            file.absolutePath
        } catch (e: Exception) {
            Log.e(TAG, "Error saving bitmap", e)
            null
        }
    }
    
    /**
     * Load bitmap from internal storage
     */
    fun loadBitmapFromInternalStorage(context: Context, fileName: String): Bitmap? {
        return try {
            val file = File(context.filesDir, fileName)
            if (file.exists()) {
                BitmapFactory.decodeFile(file.absolutePath)
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading bitmap", e)
            null
        }
    }
    
    /**
     * Resize bitmap to specified dimensions while maintaining aspect ratio
     */
    fun resizeBitmap(bitmap: Bitmap, maxWidth: Int, maxHeight: Int): Bitmap {
        val width = bitmap.width
        val height = bitmap.height
        
        if (width <= maxWidth && height <= maxHeight) {
            return bitmap
        }
        
        val aspectRatio = width.toFloat() / height.toFloat()
        
        val (newWidth, newHeight) = if (aspectRatio > 1) {
            // Landscape
            val w = minOf(maxWidth, width)
            val h = (w / aspectRatio).toInt()
            Pair(w, h)
        } else {
            // Portrait or square
            val h = minOf(maxHeight, height)
            val w = (h * aspectRatio).toInt()
            Pair(w, h)
        }
        
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true)
    }
    
    /**
     * Rotate bitmap based on EXIF orientation
     */
    fun rotateBitmapIfNeeded(bitmap: Bitmap, uri: Uri, context: Context): Bitmap {
        return try {
            val inputStream = context.contentResolver.openInputStream(uri)
            val exif = ExifInterface(inputStream!!)
            val orientation = exif.getAttributeInt(
                ExifInterface.TAG_ORIENTATION,
                ExifInterface.ORIENTATION_NORMAL
            )
            
            val rotationAngle = when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> 90f
                ExifInterface.ORIENTATION_ROTATE_180 -> 180f
                ExifInterface.ORIENTATION_ROTATE_270 -> 270f
                else -> 0f
            }
            
            if (rotationAngle != 0f) {
                val matrix = Matrix()
                matrix.postRotate(rotationAngle)
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            } else {
                bitmap
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error rotating bitmap", e)
            bitmap
        }
    }
    
    /**
     * Compress bitmap for analysis while maintaining quality
     */
    fun prepareImageForAnalysis(bitmap: Bitmap): Bitmap {
        return resizeBitmap(bitmap, MAX_IMAGE_SIZE, MAX_IMAGE_SIZE)
    }
    
    /**
     * Generate unique filename with timestamp
     */
    fun generateUniqueFileName(prefix: String, extension: String): String {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        return "${prefix}_${timestamp}.$extension"
    }
    
    /**
     * Load CSV data from assets
     */
    fun loadCSVFromAssets(context: Context, fileName: String): List<List<String>> {
        val data = mutableListOf<List<String>>()
        try {
            val inputStream = context.assets.open(fileName)
            val reader = BufferedReader(InputStreamReader(inputStream))
            
            var line: String?
            while (reader.readLine().also { line = it } != null) {
                val row = line!!.split(",").map { it.trim() }
                data.add(row)
            }
            reader.close()
        } catch (e: Exception) {
            Log.e(TAG, "Error loading CSV: $fileName", e)
        }
        return data
    }
    
    /**
     * Save text data to internal storage
     */
    fun saveTextToInternalStorage(
        context: Context,
        text: String,
        fileName: String
    ): Boolean {
        return try {
            val file = File(context.filesDir, fileName)
            val writer = FileWriter(file)
            writer.write(text)
            writer.close()
            true
        } catch (e: Exception) {
            Log.e(TAG, "Error saving text file", e)
            false
        }
    }
    
    /**
     * Load text data from internal storage
     */
    fun loadTextFromInternalStorage(context: Context, fileName: String): String? {
        return try {
            val file = File(context.filesDir, fileName)
            if (file.exists()) {
                file.readText()
            } else {
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error loading text file", e)
            null
        }
    }
    
    /**
     * Delete file from internal storage
     */
    fun deleteFileFromInternalStorage(context: Context, fileName: String): Boolean {
        return try {
            val file = File(context.filesDir, fileName)
            file.delete()
        } catch (e: Exception) {
            Log.e(TAG, "Error deleting file", e)
            false
        }
    }
    
    /**
     * Get file size in bytes
     */
    fun getFileSize(context: Context, fileName: String): Long {
        return try {
            val file = File(context.filesDir, fileName)
            if (file.exists()) file.length() else 0L
        } catch (e: Exception) {
            Log.e(TAG, "Error getting file size", e)
            0L
        }
    }
    
    /**
     * Check available storage space
     */
    fun getAvailableStorageSpace(context: Context): Long {
        return try {
            val filesDir = context.filesDir
            filesDir.freeSpace
        } catch (e: Exception) {
            Log.e(TAG, "Error getting available space", e)
            0L
        }
    }
}
package com.example.emergencytriage.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.util.Log
import androidx.exifinterface.media.ExifInterface
import kotlinx.coroutines.*
import java.io.IOException
import java.io.InputStream
import kotlin.math.max
import kotlin.math.min

/**
 * Performance optimization utilities for Emergency Triage App
 * Provides image optimization, memory management, and background processing improvements
 */
object PerformanceOptimizer {

    private const val TAG = "PerformanceOptimizer"
    private const val MAX_IMAGE_SIZE = 1024 // Maximum dimension for processed images
    private const val JPEG_QUALITY = 85 // JPEG compression quality (0-100)

    /**
     * Image optimization utilities
     */
    object ImageOptimizer {

        /**
         * Loads and optimizes image from URI with memory-efficient approach
         */
        suspend fun loadOptimizedImage(
            context: Context,
            uri: Uri,
            maxWidth: Int = MAX_IMAGE_SIZE,
            maxHeight: Int = MAX_IMAGE_SIZE
        ): Bitmap? = withContext(Dispatchers.IO) {
            try {
                val inputStream = context.contentResolver.openInputStream(uri)
                inputStream?.use { stream ->
                    // First, decode image dimensions without loading the full image
                    val options = BitmapFactory.Options().apply {
                        inJustDecodeBounds = true
                    }
                    BitmapFactory.decodeStream(stream, null, options)

                    // Calculate sample size to reduce memory usage
                    val sampleSize = calculateInSampleSize(options, maxWidth, maxHeight)
                    
                    // Reopen stream and decode with sample size
                    context.contentResolver.openInputStream(uri)?.use { newStream ->
                        val decodeOptions = BitmapFactory.Options().apply {
                            inSampleSize = sampleSize
                            inPreferredConfig = Bitmap.Config.RGB_565 // Use less memory
                            inDither = false
                            inPurgeable = true
                            inInputShareable = true
                        }
                        
                        val bitmap = BitmapFactory.decodeStream(newStream, null, decodeOptions)
                        bitmap?.let { 
                            // Apply EXIF orientation correction
                            correctImageOrientation(it, uri, context)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load optimized image", e)
                null
            }
        }

        /**
         * Calculates optimal sample size to reduce memory usage
         */
        private fun calculateInSampleSize(
            options: BitmapFactory.Options,
            reqWidth: Int,
            reqHeight: Int
        ): Int {
            val height = options.outHeight
            val width = options.outWidth
            var inSampleSize = 1

            if (height > reqHeight || width > reqWidth) {
                val halfHeight = height / 2
                val halfWidth = width / 2

                while (halfHeight / inSampleSize >= reqHeight && halfWidth / inSampleSize >= reqWidth) {
                    inSampleSize *= 2
                }
            }

            return inSampleSize
        }

        /**
         * Corrects image orientation based on EXIF data
         */
        private suspend fun correctImageOrientation(
            bitmap: Bitmap,
            uri: Uri,
            context: Context
        ): Bitmap = withContext(Dispatchers.IO) {
            try {
                val inputStream = context.contentResolver.openInputStream(uri)
                inputStream?.use { stream ->
                    val exif = ExifInterface(stream)
                    val orientation = exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION,
                        ExifInterface.ORIENTATION_NORMAL
                    )

                    val matrix = Matrix()
                    when (orientation) {
                        ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
                        ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
                        ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
                        ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.preScale(-1f, 1f)
                        ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.preScale(1f, -1f)
                        else -> return@withContext bitmap
                    }

                    Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
                        .also { 
                            if (it != bitmap) {
                                bitmap.recycle() // Free original bitmap memory
                            }
                        }
                } ?: bitmap
            } catch (e: IOException) {
                Log.w(TAG, "Could not read EXIF data", e)
                bitmap
            }
        }

        /**
         * Resizes bitmap to specified dimensions while maintaining aspect ratio
         */
        fun resizeBitmap(
            bitmap: Bitmap,
            maxWidth: Int,
            maxHeight: Int
        ): Bitmap {
            val width = bitmap.width
            val height = bitmap.height

            if (width <= maxWidth && height <= maxHeight) {
                return bitmap
            }

            val aspectRatio = width.toFloat() / height.toFloat()
            val newWidth: Int
            val newHeight: Int

            if (width > height) {
                newWidth = maxWidth
                newHeight = (maxWidth / aspectRatio).toInt()
            } else {
                newHeight = maxHeight
                newWidth = (maxHeight * aspectRatio).toInt()
            }

            return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true).also {
                if (it != bitmap) {
                    bitmap.recycle()
                }
            }
        }

        /**
         * Compresses bitmap to reduce memory footprint
         */
        fun compressBitmap(bitmap: Bitmap, quality: Int = JPEG_QUALITY): Bitmap {
            return try {
                val stream = java.io.ByteArrayOutputStream()
                bitmap.compress(Bitmap.CompressFormat.JPEG, quality, stream)
                val byteArray = stream.toByteArray()
                
                BitmapFactory.decodeByteArray(byteArray, 0, byteArray.size).also {
                    if (it != bitmap) {
                        bitmap.recycle()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to compress bitmap", e)
                bitmap
            }
        }
    }

    /**
     * Memory management utilities
     */
    object MemoryManager {

        /**
         * Gets current memory usage information
         */
        fun getMemoryInfo(): MemoryInfo {
            val runtime = Runtime.getRuntime()
            val maxMemory = runtime.maxMemory() / (1024 * 1024) // MB
            val totalMemory = runtime.totalMemory() / (1024 * 1024) // MB
            val freeMemory = runtime.freeMemory() / (1024 * 1024) // MB
            val usedMemory = totalMemory - freeMemory

            return MemoryInfo(
                maxMemoryMB = maxMemory,
                usedMemoryMB = usedMemory,
                availableMemoryMB = maxMemory - usedMemory,
                memoryUsagePercentage = (usedMemory * 100 / maxMemory).toInt()
            )
        }

        /**
         * Checks if memory usage is high
         */
        fun isMemoryUsageHigh(): Boolean {
            val memoryInfo = getMemoryInfo()
            return memoryInfo.memoryUsagePercentage > 80
        }

        /**
         * Triggers garbage collection if memory usage is high
         */
        fun optimizeMemoryUsage() {
            val memoryInfo = getMemoryInfo()
            Log.d(TAG, "Memory usage: ${memoryInfo.memoryUsagePercentage}%")
            
            if (memoryInfo.memoryUsagePercentage > 70) {
                Log.d(TAG, "High memory usage detected, suggesting GC")
                System.gc()
            }
        }

        /**
         * Estimates memory usage for a bitmap
         */
        fun estimateBitmapMemoryUsage(width: Int, height: Int, config: Bitmap.Config): Long {
            val bytesPerPixel = when (config) {
                Bitmap.Config.ARGB_8888 -> 4
                Bitmap.Config.RGB_565 -> 2
                Bitmap.Config.ARGB_4444 -> 2
                Bitmap.Config.ALPHA_8 -> 1
                else -> 4
            }
            return (width * height * bytesPerPixel).toLong()
        }
    }

    /**
     * Background processing utilities
     */
    object BackgroundProcessor {

        private val backgroundScope = CoroutineScope(
            Dispatchers.Default + SupervisorJob() + 
            CoroutineExceptionHandler { _, exception ->
                Log.e(TAG, "Background processing error", exception)
            }
        )

        /**
         * Processes multiple images concurrently with controlled parallelism
         */
        suspend fun processImagesInParallel(
            context: Context,
            imageUris: List<Uri>,
            maxConcurrency: Int = 3
        ): List<Bitmap?> = withContext(Dispatchers.IO) {
            val semaphore = Semaphore(maxConcurrency)
            
            imageUris.map { uri ->
                async {
                    semaphore.withPermit {
                        ImageOptimizer.loadOptimizedImage(context, uri)
                    }
                }
            }.awaitAll()
        }

        /**
         * Debounced execution to prevent rapid successive calls
         */
        fun debounce(
            delay: Long = 300L,
            action: suspend () -> Unit
        ): suspend () -> Unit {
            var debounceJob: Job? = null
            
            return {
                debounceJob?.cancel()
                debounceJob = backgroundScope.launch {
                    delay(delay)
                    action()
                }
            }
        }

        /**
         * Throttled execution to limit call frequency
         */
        fun throttle(
            interval: Long = 1000L,
            action: suspend () -> Unit
        ): suspend () -> Unit {
            var lastExecution = 0L
            var throttleJob: Job? = null
            
            return {
                val now = System.currentTimeMillis()
                val timeSinceLastExecution = now - lastExecution
                
                if (timeSinceLastExecution >= interval) {
                    lastExecution = now
                    throttleJob?.cancel()
                    throttleJob = backgroundScope.launch { action() }
                }
            }
        }

        /**
         * Cleanup background resources
         */
        fun cleanup() {
            backgroundScope.cancel()
        }
    }

    /**
     * ML Model optimization utilities
     */
    object MLOptimizer {

        /**
         * Prepares input tensor with optimal format for ML models
         */
        fun prepareModelInput(
            bitmap: Bitmap,
            inputSize: Int = 224,
            normalize: Boolean = true
        ): FloatArray {
            // Resize to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            
            val pixels = IntArray(inputSize * inputSize)
            resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
            
            val input = FloatArray(inputSize * inputSize * 3)
            var idx = 0
            
            for (pixel in pixels) {
                val r = (pixel shr 16 and 0xFF)
                val g = (pixel shr 8 and 0xFF)
                val b = (pixel and 0xFF)
                
                if (normalize) {
                    // Normalize to [0, 1]
                    input[idx++] = r / 255.0f
                    input[idx++] = g / 255.0f
                    input[idx++] = b / 255.0f
                } else {
                    input[idx++] = r.toFloat()
                    input[idx++] = g.toFloat()
                    input[idx++] = b.toFloat()
                }
            }
            
            // Clean up temporary bitmap
            if (resizedBitmap != bitmap) {
                resizedBitmap.recycle()
            }
            
            return input
        }

        /**
         * Batch processes multiple inputs for better throughput
         */
        suspend fun batchProcess(
            inputs: List<FloatArray>,
            processor: suspend (List<FloatArray>) -> List<FloatArray>,
            batchSize: Int = 4
        ): List<FloatArray> = withContext(Dispatchers.Default) {
            inputs.chunked(batchSize).flatMap { batch ->
                processor(batch)
            }
        }
    }

    /**
     * UI optimization utilities
     */
    object UIOptimizer {

        /**
         * Lazy loading implementation for RecyclerView
         */
        class LazyLoader<T>(
            private val pageSize: Int = 20,
            private val loadData: suspend (offset: Int, limit: Int) -> List<T>
        ) {
            private val items = mutableListOf<T>()
            private var isLoading = false
            private var hasMore = true

            suspend fun loadNextPage(): List<T> {
                if (isLoading || !hasMore) return emptyList()
                
                isLoading = true
                return try {
                    val newItems = loadData(items.size, pageSize)
                    if (newItems.size < pageSize) {
                        hasMore = false
                    }
                    items.addAll(newItems)
                    newItems
                } finally {
                    isLoading = false
                }
            }

            fun getAllItems(): List<T> = items.toList()
            fun hasMoreItems(): Boolean = hasMore
            fun isCurrentlyLoading(): Boolean = isLoading
        }

        /**
         * View recycling for better memory usage
         */
        fun recycleViews(vararg bitmaps: Bitmap?) {
            bitmaps.forEach { bitmap ->
                bitmap?.takeIf { !it.isRecycled }?.recycle()
            }
        }
    }
}

/**
 * Data class for memory information
 */
data class MemoryInfo(
    val maxMemoryMB: Long,
    val usedMemoryMB: Long,
    val availableMemoryMB: Long,
    val memoryUsagePercentage: Int
)

/**
 * Performance monitoring utilities
 */
object PerformanceMonitor {
    
    private const val TAG = "PerformanceMonitor"
    
    /**
     * Measures execution time of a suspend function
     */
    suspend inline fun <T> measureTime(
        operation: String,
        block: suspend () -> T
    ): Pair<T, Long> {
        val startTime = System.currentTimeMillis()
        val result = block()
        val duration = System.currentTimeMillis() - startTime
        
        Log.d(TAG, "$operation took ${duration}ms")
        return Pair(result, duration)
    }
    
    /**
     * Monitors memory usage during operation
     */
    suspend inline fun <T> monitorMemory(
        operation: String,
        block: suspend () -> T
    ): T {
        val initialMemory = PerformanceOptimizer.MemoryManager.getMemoryInfo()
        Log.d(TAG, "$operation - Initial memory: ${initialMemory.usedMemoryMB}MB")
        
        val result = block()
        
        val finalMemory = PerformanceOptimizer.MemoryManager.getMemoryInfo()
        val memoryDelta = finalMemory.usedMemoryMB - initialMemory.usedMemoryMB
        Log.d(TAG, "$operation - Final memory: ${finalMemory.usedMemoryMB}MB (Δ${memoryDelta}MB)")
        
        return result
    }
}
package com.tflite.demo

import android.app.Activity
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class DigitsDetector(activity: Activity) {

    private val TAG = this.javaClass.simpleName

    // tensorflow lite 文件
    private lateinit var tflite: Interpreter

    // Input byte buffer
    private lateinit var inputBuffer: ByteBuffer

    // Output array [batch_size, 10]
    private lateinit var mnistOutput: Array<FloatArray>

    init {
        try {
            tflite = Interpreter(loadModelFile(activity))

            inputBuffer = ByteBuffer.allocateDirect(
                    BYTE_SIZE_OF_FLOAT * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE)
            inputBuffer.order(ByteOrder.nativeOrder())
            mnistOutput = Array(DIM_BATCH_SIZE) { FloatArray(NUMBER_LENGTH) }
            Log.d(TAG, "创建了Tensorflow Lite MNIST分类器")
        } catch (e: IOException) {
            Log.e(TAG, "加载tflite文件失败")
        }

    }

    /*
     * 从assets文件夹加载模型文件
     */
    @Throws(IOException::class)
    private fun loadModelFile(activity: Activity): MappedByteBuffer {

        val fileDescriptor = activity.assets.openFd(MODEL_PATH)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /*
     *使用mnist模型对数字进行分类
     *返回 验证结果
     */
    fun classify(bitmap: Bitmap): Int {

        if (tflite == null) {
            Log.e(TAG, "Image classifier has not been initialized; Skipped.")
        }

        preProcess(bitmap)
        runModel()
        return postProcess()
    }

    /*
     * 将其转换为字节缓冲区以馈送到模型中
     */
    private fun preProcess(bitmap: Bitmap?) {

        if (bitmap == null || inputBuffer == null) {
            return
        }

        // 重置图像数据
        inputBuffer.rewind()

        val width = bitmap.width
        val height = bitmap.height

        // 图形状应为28 x 28
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (i in pixels.indices) {
            // 将白色像素设置为0，黑色像素设置为255
            val pixel = pixels[i]
            // 输入的颜色为黑色，因此蓝色通道将为0xFF
            val channel = pixel and 0xff
            inputBuffer.putFloat((0xff - channel).toFloat())
        }
    }

    /*
     * 运行TFLite模型
     */
    private fun runModel() = tflite.run(inputBuffer, mnistOutput)

    /*
     * 检查输出并找到已识别的编号
     *返回已标识的编号（如果未找到，则返回-1）
     */
    private fun postProcess(): Int {

        for (i in 0 until mnistOutput[0].size) {
            val value = mnistOutput[0][i]
            if (value == 1f) {
                return i
            }
        }

        return -1
    }

    companion object {

        private val MODEL_PATH = "mnist.tflite"

        // 指定输出大小
        private val NUMBER_LENGTH = 10

        // 指定输入大小
        private val DIM_BATCH_SIZE = 1
        private val DIM_IMG_SIZE_X = 28
        private val DIM_IMG_SIZE_Y = 28
        private val DIM_PIXEL_SIZE = 1

        // 保存浮点的字节数 (32 bits / float) / (8 bits / byte) = 4 bytes / float
        private val BYTE_SIZE_OF_FLOAT = 4
    }
}
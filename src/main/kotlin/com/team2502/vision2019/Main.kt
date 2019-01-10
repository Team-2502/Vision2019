package com.team2502.vision2019

import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import java.nio.file.Files
import java.nio.file.Paths


fun main(args: Array<String>) {

    val imgPath = Paths.get("img")

    Files.createDirectories(imgPath)

    /*
    first look at global OpenCV installation (https://github.com/openpnp/opencv)
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
    if DNE install appropriate binary
     */
    nu.pattern.OpenCV.loadShared() // first look at global OpenCV installation (

    val videoCapture = VideoCapture(0) // 0 = default device

    if (videoCapture.isOpened) {
        println("open")
        val frame = Mat()
        videoCapture.read(frame)
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_BGR2GRAY) // convert to grayscale

//        frame.
        val buffer = MatOfByte()

        Imgcodecs.imencode(".png", frame, buffer)


        val path = imgPath.resolve("test.png")

        if (!Files.exists(path)) {
            Files.createFile(path)
        }

        Files.write(path, buffer.toArray())
    } else throw IllegalArgumentException("Could not open the default camera device!")

}
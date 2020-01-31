/*----------------------------------------------------------------------------*/
/* Copyright (c) 2020 FIRST Team 2393. All Rights Reserved.                   */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import edu.wpi.cscore.CvSource;
import edu.wpi.first.vision.VisionPipeline;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/** Pipeline that looks for target */
public class TargetPipeline implements VisionPipeline
{
  private final CvSource output;
  private final int width, height;

  // Intermediate images used for processing
  private final Mat norm = new Mat(), blur = new Mat(), 
                    hls = new Mat(), filt = new Mat(), tmp = new Mat();
  
  // Counter for calls to `process()`
  public AtomicInteger calls = new AtomicInteger();

  // Hue (0-180), Luminance (0-255), Saturation (0-255) filter
  // In principle looking for 'green' light, hue ~ 75
  // but biggest emphasis is on 'bright'.
  private final Scalar hls_min = new Scalar( 75-10,  50.0,  20.0);
  private final Scalar hls_max = new Scalar( 75+10, 255.0, 255.0);

  private final List<MatOfPoint> contours = new ArrayList<>();

  // Colors for drawing overlay
  private final Scalar overlay_bgr = new Scalar(200.0, 100.0, 255.0),
                      contrast_bgr = new Scalar(0 ,0, 0);

  public final UDPServer udp;

  TargetPipeline(final CvSource output, int width, int height) throws Exception
  {
    this.output = output;
    this.width = width;
    this.height = height;

    udp = new UDPServer(5801);

    SmartDashboard.setDefaultNumber("HueMin", hls_min.val[0]);
    SmartDashboard.setDefaultNumber("HueMax", hls_max.val[0]);
    SmartDashboard.setDefaultNumber("LumMin", hls_min.val[1]);
    SmartDashboard.setDefaultNumber("LumMax", hls_max.val[1]);
    SmartDashboard.setDefaultNumber("SatMin", hls_min.val[2]);
    SmartDashboard.setDefaultNumber("SatMax", hls_max.val[2]);
    SmartDashboard.setDefaultNumber("AreaMin", 0.0);
  }

  private int[] center_hls = new int[] { 0, 0, 0 };

  @Override
  public void process(final Mat frame)
  {
    // In principle, should be possible to re-use Mat()s:
    // 1) Resize original frame to smaller tmp1
    // 2) Normalize tmp1 into tmp2
    // 3) Convert RGB from tmp2 into HLS tmp1
    // 4) ...
    // .. but that resulted in strange values for HLS,
    // like H > 180.
    // So re-using Mats across process calls,
    // but within one process call always using it for the
    // same purpose.

    // Resizing the image would save CPU & memory,
    // but image is already small enough for us to handle
    
    // Scale colors to use full 0..255 range in case image was dark
    Core.normalize(frame, norm, 0.0, 255.0, Core.NORM_MINMAX);

    // When moving the camera, or turning auto-focus off and de-focusing,
    // we would detect the target, but when standing still and in perfect focus,
    // we missed it?!
    // --> Blurring the image helps detect the target!
    Imgproc.blur(norm, blur, new Size(4, 4));

    // Convert to HLS
    Imgproc.cvtColor(blur, hls, Imgproc.COLOR_BGR2HLS);

    // Probe HLS at center of image
    final byte[] probe = new byte[3];
    hls.get(height/2, width/2, probe);
    center_hls[0] = Byte.toUnsignedInt(probe[0]);
    center_hls[1] = Byte.toUnsignedInt(probe[1]);
    center_hls[2] = Byte.toUnsignedInt(probe[2]);
    
    // Filter on Hue, Luminance and Saturation
    hls_min.val[0] = SmartDashboard.getNumber("HueMin", hls_min.val[0]);
    hls_max.val[0] = SmartDashboard.getNumber("HueMax", hls_max.val[0]);
    hls_min.val[1] = SmartDashboard.getNumber("LumMin", hls_min.val[1]);
    hls_max.val[1] = SmartDashboard.getNumber("LumMax", hls_max.val[1]);
    hls_min.val[2] = SmartDashboard.getNumber("SatMin", hls_min.val[2]);
    hls_max.val[2] = SmartDashboard.getNumber("SatMax", hls_max.val[2]);
    Core.inRange(hls, hls_min, hls_max, filt);

    // Find contours
    contours.clear();
    Imgproc.findContours(filt, contours, tmp, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

    // Get largest contour
    int largest_contour_index = -1;
    double max_area = SmartDashboard.getNumber("AreaMin", 0.0);
    for (int i=0; i<contours.size(); ++i)
    {
      final MatOfPoint contour = contours.get(i);
      final double area = Imgproc.contourArea(contour);
      if (area > max_area)
      {
        max_area = area;
        largest_contour_index = i;
      }
    }
    if (largest_contour_index >= 0)
    {
      // Show largest contour
      Imgproc.drawContours(frame, contours, largest_contour_index, overlay_bgr);

      MatOfPoint largest_contour = contours.get(largest_contour_index);


      // Arrow from mid-bottom of image to center of blob
      Rect largest = Imgproc.boundingRect(largest_contour);
      // Imgproc.rectangle(frame,
      //                   new Point(largest.x*scale, largest.y*scale),
      //                   new Point((largest.x + largest.width)*scale,
      //                             (largest.y + largest.height)*scale),
      //                  overlay_bgr);
      final int pos = largest.x + largest.width/2;
      Imgproc.arrowedLine(frame,
                          new Point(width/2, height-1),
                          new Point(pos,
                                    largest.y + largest.height/2),
                          overlay_bgr);
      // Publish direction to detected blob in pixels from center
      // 0 - In center or not found, i.e. no reason to move
      // positive 1 .. width/2: Blob is to the right of center
      // negative -1 .. -width/2: .. left of center
      SmartDashboard.putNumber("Direction", pos - width/2);
      udp.send(pos - width/2);
      SmartDashboard.putNumber("Area", max_area);    
    }
    else
    {
      SmartDashboard.putNumber("Direction", 0);
      SmartDashboard.putNumber("Area", 0);    
      udp.send(0);
    }

    // Show rect in center of image where pixel info is probed
    Imgproc.rectangle(frame,
                      new Point(width/2 - 2, height/2 - 2),
                      new Point(width/2 + 2, height/2 + 2),
                      overlay_bgr);

    // Show info at bottom of image.
    // Paint is twice, overlay-on-black, for better contrast
    final String info = String.format("Frame %3d H %3d L %3d S %3d",
                                      calls.get(),
                                      center_hls[0],
                                      center_hls[1],
                                      center_hls[2]);
    Imgproc.putText(frame,
                    info,
                    new Point(1, height-16),
                    Core.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    contrast_bgr,
                    1);
    Imgproc.putText(frame,
                    info,
                    new Point(2, height-15),
                    Core.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    overlay_bgr,
                    1);

    // Publish 'output'
    // a) Show HLS filter
    // output.putFrame(filt);
    // b) Show original image with overlay
    output.putFrame(frame);

    // Indicate that we handled one more frame
    calls.incrementAndGet();
  }
}

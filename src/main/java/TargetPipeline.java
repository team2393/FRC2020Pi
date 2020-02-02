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

/** Pipeline that looks for target
 * 
 *  Filters on HSV, then picks largest contour
 *  based on range of size, aspect ratio, fullness
 */
public class TargetPipeline implements VisionPipeline
{
  private final CvSource output;
  private final int width, height;

  // Intermediate images used for processing
  private final Mat norm = new Mat(), blur = new Mat(), 
                    hsv = new Mat(), filt = new Mat(), tmp = new Mat();
  
  // Counter for calls to `process()`
  public AtomicInteger calls = new AtomicInteger();

  // Hue (0-180), Luminance (0-255), Saturation (0-255) filter
  // In principle looking for 'green' light, hue ~ 75
  // but biggest emphasis is on 'bright'.
  private final Scalar hsv_min = new Scalar( 75-20,  30.0,  50.0);
  private final Scalar hsv_max = new Scalar( 75+20, 255.0, 255.0);

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

    SmartDashboard.setDefaultNumber("HueMin", hsv_min.val[0]);
    SmartDashboard.setDefaultNumber("HueMax", hsv_max.val[0]);
    SmartDashboard.setDefaultNumber("SatMin", hsv_min.val[1]);
    SmartDashboard.setDefaultNumber("SatMax", hsv_max.val[1]);
    SmartDashboard.setDefaultNumber("ValMin", hsv_min.val[2]);
    SmartDashboard.setDefaultNumber("ValMax", hsv_max.val[2]);
    SmartDashboard.setDefaultNumber("AreaMin", 0.0);
    SmartDashboard.setDefaultNumber("AreaMax", width * height);
    SmartDashboard.setDefaultNumber("AspectMin", 0.0);
    SmartDashboard.setDefaultNumber("AspectMax", 20);
    SmartDashboard.setDefaultNumber("FullnessMin", 0.0);
    SmartDashboard.setDefaultNumber("FullnessMax", 100.0);
  }

  private int[] center_hsv = new int[] { 0, 0, 0 };

  @Override
  public void process(final Mat frame)
  {
    // In principle, should be possible to re-use Mat()s:
    // 1) Resize original frame to smaller tmp1
    // 2) Normalize tmp1 into tmp2
    // 3) Convert RGB from tmp2 into HSV tmp1
    // 4) ...
    // .. but that resulted in strange values for HSV,
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

    // Convert to HSV
    Imgproc.cvtColor(blur, hsv, Imgproc.COLOR_BGR2HSV);

    // Probe HSV at center of image
    final byte[] probe = new byte[3];
    hsv.get(height/2, width/2, probe);
    center_hsv[0] = Byte.toUnsignedInt(probe[0]);
    center_hsv[1] = Byte.toUnsignedInt(probe[1]);
    center_hsv[2] = Byte.toUnsignedInt(probe[2]);
    
    // Filter on Hue, Saturation and value
    hsv_min.val[0] = SmartDashboard.getNumber("HueMin", hsv_min.val[0]);
    hsv_max.val[0] = SmartDashboard.getNumber("HueMax", hsv_max.val[0]);
    hsv_min.val[1] = SmartDashboard.getNumber("SatMin", hsv_min.val[1]);
    hsv_max.val[1] = SmartDashboard.getNumber("SatMax", hsv_max.val[1]);
    hsv_min.val[2] = SmartDashboard.getNumber("ValMin", hsv_min.val[2]);
    hsv_max.val[2] = SmartDashboard.getNumber("ValMax", hsv_max.val[2]);
    Core.inRange(hsv, hsv_min, hsv_max, filt);

    // Find contours
    contours.clear();
    Imgproc.findContours(filt, contours, tmp, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

    // Get largest contour
    int largest_contour_index = -1;
    double largest_area = SmartDashboard.getNumber("AreaMin", 0.0);
    final double area_max = SmartDashboard.getNumber("AreaMax", width * height);

    final double aspect_min = SmartDashboard.getNumber("AspectMin", 0.0);
    final double aspect_max = SmartDashboard.getNumber("AspectMax", 20);

    final double fullness_min = SmartDashboard.getNumber("FullnessMin", 0.0);
    final double fullness_max = SmartDashboard.getNumber("FullnessMax", 100.0);
    for (int i=0; i<contours.size(); ++i)
    {
      final MatOfPoint contour = contours.get(i);
      final double area = Imgproc.contourArea(contour);
      // Filter on min..max area.
      // Ceiling lights etc. could be larger than expected target
      if (area < largest_area  ||  area > area_max)
        continue;
         
      // Filter on aspect ratio 0 (tall) .. 1 (square) .. 20 (wide)
      final Rect bounds = Imgproc.boundingRect(contour);
      final double aspect = (double)bounds.width / bounds.height;
      if (aspect < aspect_min  ||  aspect > aspect_max)
        continue;
      
      // Filter on fullness (percent): 0% (hollow) .. 100% (solid, full)
      final double fullness = 100.0 * area / (bounds.width * bounds.height);
      if (fullness < fullness_min  ||  fullness > fullness_max)
        continue;
      
      // Passed all tests: This is so far the largest area that we like
      largest_area = area;
      largest_contour_index = i;
    }
    // Found anything?
    if (largest_contour_index >= 0)
    {
      // Show largest contour
      Imgproc.drawContours(frame, contours, largest_contour_index, overlay_bgr);
      final MatOfPoint largest_contour = contours.get(largest_contour_index);

      // Arrow from mid-bottom of image to center of blob
      Rect bounds = Imgproc.boundingRect(largest_contour);
      // Imgproc.rectangle(frame,
      //                   new Point(largest.x*scale, largest.y*scale),
      //                   new Point((largest.x + largest.width)*scale,
      //                             (largest.y + largest.height)*scale),
      //                  overlay_bgr);
      final int horiz_pos = bounds.x + bounds.width/2;
      final int vert_pos  = bounds.y + bounds.height/2;
      Imgproc.arrowedLine(frame,
                          new Point(width/2, height-1),
                          new Point(horiz_pos, vert_pos),
                          overlay_bgr);
      // Publish direction to detected blob in pixels from center
      // 0 - In center or not found, i.e. no reason to move
      // positive 1 .. width/2: Blob is to the right of center
      // negative -1 .. -width/2: .. left of center
      final int direction = horiz_pos - width/2;
      // Publish distance to detected blob in pixels from center
      // 0 - In center or not found, i.e. no reason to move
      // positive 1 .. height/2: Blob is ahead of center
      // negative -1 .. -height/2: .. below center
      final int distance = height/2 - vert_pos;
      SmartDashboard.putNumber("Direction", direction);
      SmartDashboard.putNumber("Distance", distance);
      udp.send(direction);
      SmartDashboard.putNumber("Area", largest_area);    

      final double fullness = 100.0 * largest_area / (bounds.width * bounds.height);
      SmartDashboard.putNumber("Fullness", fullness);    

      final double aspect = (double)bounds.width / bounds.height;
      SmartDashboard.putNumber("Aspect", aspect);    
    }
    else
    {
      SmartDashboard.putNumber("Direction", 0);
      SmartDashboard.putNumber("Distance", 0);
      SmartDashboard.putNumber("Area", 0);    
      SmartDashboard.putNumber("Fullness", -1);    
      SmartDashboard.putNumber("Aspect", -1);    
      udp.send(0);
    }

    // Show rect in center of image where pixel info is probed
    Imgproc.rectangle(frame,
                      new Point(width/2 - 2, height/2 - 2),
                      new Point(width/2 + 2, height/2 + 2),
                      overlay_bgr);

    // Show info at bottom of image.
    // Paint is twice, overlay-on-black, for better contrast
    final String info = String.format("Frame %3d H %3d S %3d V %3d",
                                      calls.get(),
                                      center_hsv[0],
                                      center_hsv[1],
                                      center_hsv[2]);
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
    output.putFrame(frame);

    // Indicate that we handled one more frame
    calls.incrementAndGet();
  }
}

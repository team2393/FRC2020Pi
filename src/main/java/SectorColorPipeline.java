/*----------------------------------------------------------------------------*/
/* Copyright (c) 2019 FIRST Team 2393. All Rights Reserved.                   */
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

/** Pipeline that detects the color wheel sector */
public class SectorColorPipeline implements VisionPipeline
{
  // Scaling factor for reduced size of processed image
  public static final int scale = 2;

  private final CvSource output;
  private final int width, height, proc_width, proc_height;

  // Intermediate images used for processing
  private final Mat small = new Mat(), norm = new Mat(), hls = new Mat(), filt = new Mat(), tmp = new Mat();

  // Counter for calls to `process()`
  public AtomicInteger calls = new AtomicInteger();

  // Hue, Luminance, Saturation filter
  // Pink has hue 300 on 0-360 scale.
  // OpenCV uses a 0-180 scale, i.e. pink = 150 (within 140..160).

  final String[] colors = { "Blue", "Green", "Red", "Yellow" };

  // Minumum and maximum HLS range for each color
  private final Scalar[] hls_min =
  {
    new Scalar( 90.0, 100.0,   0.0), // Blue
    new Scalar( 30.0,  90.0,   0.0), // Green
    new Scalar(  0.0,  70.0, 200.0), // Red
    new Scalar( 15.0, 100.0, 200.0)  // Yellow
  };
  private final Scalar[] hls_max =
  {
    new Scalar(130.0, 180.0,  40.0), // Blue
    new Scalar( 45.0, 180.0,  70.0), // Green
    new Scalar( 10.0, 120.0, 255.0), // Red
    new Scalar( 25.0, 180.0, 255.0)  // Yellow
  };

  private final List<MatOfPoint> contours = new ArrayList<>();

  // Colors for drawing overlay
  private final Scalar overlay_bgr = new Scalar(200.0, 100.0, 255.0), contrast_bgr = new Scalar(0, 0, 0);

  SectorColorPipeline(final CvSource output, int width, int height)
  {
    this.output = output;
    this.width = width;
    this.height = height;
    proc_width = width / scale;
    proc_height = height / scale;
  }

  /** @param hue
   *  @param lum
   *  @param sat
   *  @param min Minimum HLS
   *  @param max Maximum HLS
   *  @return true if HLS within min..max
   */
  private boolean isMatch(final int hue, final int lum, final int sat,
                          final Scalar min, final Scalar max)
  {
    return min.val[0] <= hue  &&  hue <= max.val[0]  &&
           min.val[1] <= lum  &&  lum <= max.val[1]  &&
           min.val[2] <= sat  &&  sat <= max.val[2]; 
  }

  /** Check if HLS matches one of the expected colors
   *  @param hue
   *  @param lum
   *  @param sat
   *  @return Color 0, 1, 2, 3 or -1 if no match found
   */
  private int getMatchingColor(final int hue, final int lum, final int sat)
  {
    for (int i=0; i<colors.length; ++i)
      if (isMatch(hue, lum, sat, hls_min[i], hls_max[i]))
        return i;
    return -1;
  }

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

    // Resize to use less CPU & memory to process 
    Imgproc.resize(frame, small, new Size(proc_width, proc_height));
    
    // Scale colors to use full 0..255 range in case image was dark
    Core.normalize(small, norm, 0.0, 255.0, Core.NORM_MINMAX);

    // Convert to HLS
    Imgproc.cvtColor(norm, hls, Imgproc.COLOR_BGR2HLS);

    // Probe HLS at center of image,
    // averaging over 9 pixels at center x, y +-1
    int center_h = 0, center_l = 0, center_s = 0;
    for (int x=-1; x<2; ++x)
      for (int y=-1; y<2; ++y)
      {
        final byte[] probe = new byte[3];
        hls.get(proc_height/2 + x, proc_width/2 + y, probe);

        // Consider 175 close enough to red == 0
        int hue = Byte.toUnsignedInt(probe[0]);
        if (hue >= 175)
          hue = 0;
        center_h += hue;
        center_l += Byte.toUnsignedInt(probe[1]);
        center_s += Byte.toUnsignedInt(probe[2]);
      }
    center_h /= 9;
    center_l /= 9;
    center_s /= 9;

    // Check if center color matches any of the expected colors
    final int color_index = getMatchingColor(center_h, center_l, center_s);

    double max_area = 0.0;
    if (color_index >= 0)
    {
      // Filter on that Hue, Luminance and Saturation to get pink,
      Core.inRange(hls, hls_min[color_index], hls_max[color_index], filt);
      
      // Find contours
      contours.clear();
      Imgproc.findContours(filt, contours, tmp, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

      // Get largest contour
      Rect largest = null;
      for (int i=0; i<contours.size(); ++i)
      {
        final MatOfPoint contour = contours.get(i);
        final double area = Imgproc.contourArea(contour);
        if (area > max_area)
        {
          max_area = area;
          largest = Imgproc.boundingRect(contour);
        }
      }
  
      if (largest != null)
      {
        // Rect around the largest blob
        Imgproc.rectangle(frame,
                          new Point(largest.x * scale, largest.y * scale),
                          new Point((largest.x + largest.width) * scale,
                                    (largest.y + largest.height)* scale),
                          overlay_bgr);
      }
    }
    
    // Show rect in center of image where pixel info is probed
    Imgproc.rectangle(frame,
    new Point(width/2 - 2, height/2 - 2),
    new Point(width/2 + 2, height/2 + 2),
    overlay_bgr);
    
    // Show info at bottom of image.
    // Paint it twice, overlay-on-black, for better contrast
    final String color_name = color_index >= 0 ? colors[color_index] : "Unknown";
    SmartDashboard.putNumber("Color Area", max_area);
    SmartDashboard.putString("Color", color_name);
    SmartDashboard.putNumber("Color Idx", color_index);
    SmartDashboard.putNumber("Center H", center_h);
    SmartDashboard.putNumber("Center L", center_l);
    SmartDashboard.putNumber("Center S", center_s);
    final String info = String.format("Frame %3d H %3d L %3d S %3d %s",
                                      calls.incrementAndGet(),
                                      center_h,
                                      center_l,
                                      center_s,
                                      color_name);
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
    // output.putFrame(tmp1);
    // b) Show original image with overlay
    output.putFrame(frame);
  }
}

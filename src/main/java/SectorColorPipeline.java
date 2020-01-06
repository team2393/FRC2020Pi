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
  private final Mat small = new Mat(), norm = new Mat(), filt = new Mat(), tmp = new Mat();

  // Counter for calls to `process()`
  public AtomicInteger calls = new AtomicInteger();

  // Color names
  final String[] colors = { "Blue", "Green", "Red", "Yellow" };

  // The game manual defines the colors via CMY,
  // which should convert to RGB via
  // R = 255 - C
  // G = 255 - M
  // M = 255 - Y
  //
  //          C      M    Y      R    G    B
  // "Blue"   255,   0,   0        0, 255, 255
  // "Green"  255,   0, 255        0, 255,   0
  // "Red"      0, 255, 255      255,   0,   0
  // "Yellow"   0,   0, 255      255, 255,   0
  //
  // Minumum and maximum BGR range for each color
  private final Scalar[] bgr_min =
  {
    new Scalar(120.0,   0.0,   0.0), // Blue
    new Scalar(  0.0, 130.0,   0.0), // Green
    new Scalar(  0.0,   0.0, 140.0), // Red
    new Scalar(  0.0, 165.0, 190.0)  // Yellow
  },                     bgr_max =
  {
    new Scalar(255.0, 255.0, 140.0), // Blue
    new Scalar(115.0, 255.0, 140.0), // Green
    new Scalar( 75.0,  50.0, 255.0), // Red
    new Scalar(125.0, 255.0, 255.0)  // Yellow
  };

  private final List<MatOfPoint> contours = new ArrayList<>();

  // Colors for drawing overlay
  private final Scalar overlay_bgr = new Scalar(200.0, 100.0, 255.0), contrast_bgr = new Scalar(0, 0, 0);

  SectorColorPipeline(final CvSource output, final int width, final int height)
  {
    this.output = output;
    this.width = width;
    this.height = height;
    proc_width = width / scale;
    proc_height = height / scale;
  }

  /** @param blue
   *  @param green
   *  @param red
   *  @param min Minimum BGR
   *  @param max Maximum BGR
   *  @return true if BGR within min..max
   */
  private boolean isMatch(final int blue, final int green, final int red,
                          final Scalar min, final Scalar max)
  {
    return min.val[0] <= blue   &&  blue  <= max.val[0]  &&
           min.val[1] <= green  &&  green <= max.val[1]  &&
           min.val[2] <= red    &&  red   <= max.val[2]; 
  }

  /** Check if BGR matches one of the expected colors
   *  @param blue
   *  @param green
   *  @param red
   *  @return Color 0, 1, 2, 3 or -1 if no match found
   */
  private int getMatchingColor(final int blue, final int green, final int red)
  {
    for (int i=0; i<colors.length; ++i)
      if (isMatch(blue, green, red, bgr_min[i], bgr_max[i]))
        return i;
    // No match
    return -1;
  }

  @Override
  public void process(final Mat frame)
  {
    // In principle, should be possible to re-use Mat()s:
    // 1) Resize original frame to smaller tmp1
    // 2) Normalize tmp1 into tmp2
    // 3) Convert RGB from tmp2 into HLS tmp1
    // .. but that resulted in strange values for HLS,
    // like H > 180.
    // So re-using Mats across process calls,
    // but within one process call always using it for the
    // same purpose.

    // Resize to use less CPU & memory to process 
    Imgproc.resize(frame, small, new Size(proc_width, proc_height));
    
    // Scale colors to use full 0..255 range in case image was dark
    Core.normalize(small, norm, 0.0, 255.0, Core.NORM_MINMAX);

    // Probe BGR at center of image,
    int center_b = 0, center_g = 0, center_r = 0;
    // Average over 9 pixels at center x, y +-1
    final byte[] probe = new byte[3];
    int avg = 0;
    for (int x=-1; x<=1; ++x)
      for (int y=-1; y<=1; ++y)
      {
        norm.get(proc_height/2 + x, proc_width/2 + y, probe);
        center_b += Byte.toUnsignedInt(probe[0]);
        center_g += Byte.toUnsignedInt(probe[1]);
        center_r += Byte.toUnsignedInt(probe[2]);
        ++avg;
      }
    center_b /= avg;
    center_g /= avg;
    center_r /= avg;

    // Check if center color matches any of the expected colors
    final int color_index = getMatchingColor(center_b, center_g, center_r);

    double max_area = 0.0;
    if (color_index >= 0)
    {
      // Filter on that BGR range
      Core.inRange(norm, bgr_min[color_index], bgr_max[color_index], filt);
      
      // Find contours
      contours.clear();
      Imgproc.findContours(filt, contours, tmp, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
      
      // Get largest contour
      Rect largest = null;
      MatOfPoint largest_contour = null;
      for (int i=0; i<contours.size(); ++i)
      {
        final MatOfPoint contour = contours.get(i);
        final double area = Imgproc.contourArea(contour);
        if (area > max_area)
        {
          max_area = area;
          largest_contour = contour;
          // largest = Imgproc.boundingRect(contour);
        }
      }
  
      // Show the largest contour (slower)
      if (largest_contour != null)
      {
        final int points = largest_contour.size(0);
        Point last = null;
        for (int i=0; i<points; ++i)
        {
          final double[] xy = largest_contour.get(i, 0);
          final Point point = new Point(xy[0]*scale, xy[1]*scale);
          if (i > 0)
            Imgproc.line(frame, last, point, overlay_bgr);
          last = point;
        }
      }
      // Just show rect around the largest blob (faster)
      if (largest != null)
        Imgproc.rectangle(frame,
                          new Point(largest.x * scale, largest.y * scale),
                          new Point((largest.x + largest.width) * scale,
                                    (largest.y + largest.height)* scale),
                          overlay_bgr);
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
    SmartDashboard.putNumber("Center B", center_b);
    SmartDashboard.putNumber("Center G", center_g);
    SmartDashboard.putNumber("Center R", center_r);
    final String info = String.format("Frame %3d B %3d G %3d R %3d %s",
                                      calls.incrementAndGet(),
                                      center_b,
                                      center_g,
                                      center_r,
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

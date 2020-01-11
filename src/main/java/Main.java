/*----------------------------------------------------------------------------*/
/* Copyright (c) 2019 FIRST Team 2393. All Rights Reserved.                   */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.channels.SocketChannel;
import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.UsbCamera;
import edu.wpi.cscore.VideoProperty;
import edu.wpi.cscore.VideoSource;
import edu.wpi.cscore.VideoMode.PixelFormat;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.VisionThread;
import edu.wpi.first.wpilibj.smartdashboard.SmartDashboard;

/** Vision processing code for Raspberry Pi FRCVision */
public final class Main
{
  public static final int team = 2393;
  public static final boolean server = false;
  public static final int width = 320, height = 240, fps = 30;

  /** Connect to the web port of the RIO
   *  @return Seconds spent trying
   */
  public static int waitForRIO() throws Exception
  {
    final String address = String.format("10.%02d.%02d.2", team / 100, team % 100);
    int connect_trials = 0;
    SocketChannel rio = SocketChannel.open(new InetSocketAddress(InetAddress.getByName(address), 80));
    while (! rio.isConnected())
    {
      ++connect_trials;
      Thread.sleep(1000);
    }
    return connect_trials;
  }

  public static void main(String... args) throws Exception
  {
    // When RIO, Radio/Network switch and Pi are all powered up,
    // the Pi tends to be 'up' before it can connect to the Network Tables on the RIO.
    // NT 'isConnected()' will report true, but the NT values still don't
    // seem to change on the RIO.
    // First waiting until we can reach the RIO seems to help.
    final int connect_trials = waitForRIO();

    // Start NetworkTables
    int nt_attempts = 1;
    final NetworkTableInstance ntinst = NetworkTableInstance.getDefault();
    if (server)
    {
      System.out.println("Acting as NetworkTables server");
      ntinst.startServer();
    }
    else
    {
      System.out.println("NetworkTables client for team " + team);
      ntinst.startClientTeam(team);
      while (! ntinst.isConnected())
      {
        ++nt_attempts;
        Thread.sleep(1000);
      }
    }

    // From https://www.chiefdelphi.com/t/networking-a-raspberry-pi/335503/16
    // Note you can use Flush() to get an immediate flush of NetworkTables data changes
    // ...
    // If you set the periodic rate very slow and call Flush() immediately
    // after updating the values, latency is minimized to basically zero.
    // This is how the Limelight gets low latency updates via NetworkTables.    
    ntinst.setUpdateRate(1.00);

    // Start camera
    System.out.println("Starting camera");

    final UsbCamera camera = new UsbCamera("usbcam", 0);
    camera.setConnectVerbose(1);
    camera.setConnectionStrategy(VideoSource.ConnectionStrategy.kKeepOpen);
    camera.setVideoMode(PixelFormat.kYUYV, width, height, fps);

    // Default uses 'auto' white balance.
    // This creates overly colorful images, but better for color detection
    camera.setWhiteBalanceManual(6500);

    // TODO low brightness, ..
    // to get all initial settings correct at bootup
    camera.setBrightness(10);
    camera.setExposureManual(10);

    // Focus: Auto?

    for (VideoProperty property : camera.enumerateProperties())
    {
      System.out.println(property.getName() + " = " + property.get());
    }
    
    System.out.println("Starting camera image server");
    final CameraServer server = CameraServer.getInstance();
    server.startAutomaticCapture(camera);
    
    System.out.println("Starting processing pipeline");
    final CvSource processed = server.putVideo("Processed", width, height);

    // Select a pipeline to process the image
    // final PinkBlobPipeline my_pipeline = new PinkBlobPipeline(processed, width, height);
    // final SectorColorPipeline my_pipeline = new SectorColorPipeline(processed, width, height);
    final TargetPipeline my_pipeline = new TargetPipeline(processed, width, height);
    
    final VisionThread vision_thread = new VisionThread(camera, my_pipeline, pipeline ->
    {
      // Our pipeline just updated image image on the dashboard.
      // Add # of calls.
      SmartDashboard.putNumber("PipelineCalls", pipeline.calls.get());

      // Flush network tables so RIO can see the info ASAP
      ntinst.flush();
    });
    vision_thread.start();
    
    // loop forever
    while (true)
    {
      // Every 10 seconds, publish how often the pipeline ran
      try
      {
        TimeUnit.SECONDS.sleep(10);
      }
      catch (InterruptedException ex)
      {
        break;
      }
      final int calls = my_pipeline.calls.getAndSet(0);
      final int cps = calls/10;
      System.out.println(LocalDateTime.now() + " - My Pipeline: " + cps + " calls per second, " +
                         (ntinst.isConnected() ? "NT connected" : "NT disconnected") +
                         " after " + nt_attempts + " attempts " +
                         " with RIO first seen after " + connect_trials);

      SmartDashboard.putNumber("PipelineCPS", cps);
    }
  }
}

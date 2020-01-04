FRC 2020 Pi
===========

Raspberry Pi code for FRC 2020 vision processing.

In VSCode, use File, Open Folder to edit.

To build in Windows cmd:

```
cd git\FRC2020Pi
\Users\Public\wpilib\2020\frccode\frcvars2020.bat
gradlew.bat clean
gradlew.bat build
```

Building for the first time requires internet access because gradle downloads itself.


Then connect to robot wireless network and upload via the rPi web dashboard http://frcvision.local
Select "Writable",
then in Application tab select the "Uploaded Java jar" option for Application,
browse to the "...-all.jar" file in the build/libs subdirectory.

Application will restart after 5 minute delay.
Console output can be seen by enabling console output in the Vision Status tab.

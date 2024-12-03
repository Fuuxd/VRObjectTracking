# AI Approach

This is code that holds the main logic for usage of an AI approach.

The value of this approach could be higher on a different better camera from luxonis rather than the current one

The main logic behind it is Chaining the use of an Object Detection model along with the use of a Pose Estimation model.

The specific models used are YOLO11 and DemseFusion, the only real classes that coincide of these two models are "bottle" or "bowl" but "bottle" is transparent and thus difficult by definition, so bowl was picked. Which was also not a a particularly an ideal object for testing due to its difficult rotation determination on certain angles

Current approach runs the DenseFusion model in the Raspberry Pi after loading it from .pth files. And the YOLO model in the edge device intel myriad in the OAK-D Lite through leveraging of a blob file which is converted primarily online into 6 shaves for maximum performance.

The conversion process can be difficult and our .pth and .blob files are not provided in this repo due to copyright considerations.

when using the combination of the blob file, the DenseFusion model being ran externally proves too big of a bottleneck, conversion of it into blob to run in the edge device would be needed.  **It is assumed FullTracking.py or trackingDemo.py are both inside the DenseFusion repository as the blob conversion has not been done yet**

Additionally, the current camera (OAK-D-LITE) has a bug in Luxonis issue #624 https://github.com/luxonis/depthai/issues/624 , where the current camera is capped at around 35 FPS although the sensor model can go to 60 FPS and does so in for example the OAK-D. 

Once tested, the accuracy of the rotation or Detection of the objects itself was not sufficiently good to justify creating and curating our own dataset along with training it for our specific objects, considering that the additional computational time needed would excessively slow our already slowed down system with a maximum performance of 35FPS, for a less accurate solution. Additionally, doing so would probably be out of our time constraint.

However, if in the future more time and a better camera are available, the extra obtrusiveness of the arUco markers could be completely removed by sacrificing some accuracy and latency.

A similar approach working off of this example using more accurate models developed in the future could also be a possibility. 
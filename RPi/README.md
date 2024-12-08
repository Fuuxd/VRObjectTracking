# Raspberry Pi files

use `install_requirements.py` to install python requirements, when in a raspberry pi, do so in a python env.

replace paths with appropiate paths to RPi machine in arUcoTrack.service, which can be used to run the headless version of arucoTrackWeb.py, RPIArucoTrackWeb.py on boot on the RPi so that it's working as long as there's a WiFi connection and the camera is connected properly.

`sudo cp arUcoTrack.service /etc/systemd/system/`
`sudo systemctl enable arUcoTrack.service`
`sudo systemctl start arUcoTrack.service`

to see the camera output or debug and test issues use the more verbose arucoTrackWeb.py. 
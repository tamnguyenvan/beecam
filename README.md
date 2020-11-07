# Beecam

Real time Virtual Background Desktop App (Windows only for now).
## What it can do?
This app can read frames from real physical webcam/camera, replace it with virtual background. It could be used as a virtual camera source for other apps like Skype, Zoom, etc. Specially, it can run on **CPU** in real time. We got ~24 FPS on `Intel core i5 9500F`.
To archieve the goal, we combined several components.
- `PyQt5` - GUI
- `Pytorch` - Deep Learning model for background removal component
- `OpenCV` - To read webcam/camera source.
- `pyvirtualcam`: To create virtual camera. See details at their [home page](https://github.com/letmaik/pyvirtualcam)
- `pygrabber`: To get webcam/camera names.
- `OpenVino`: Deep Learning model optimization framework.
## Installation
*Note: Windows 10 only for now*

**Pre-built binary**
You can download the [pre-built version]() and use it directly without any effort.

**From source**
In case you want to build it manually from source, please go through the following steps.
- Clone this repo to local: `git clone https://github.com/tamnguyenvan/beecam`
- Install [OpenVino for Windows 10](https://docs.openvinotoolkit.org/2018_R5/_docs_install_guides_installing_openvino_windows.html). This project only requires Python 3 (no need Visual Studio and CMake).
- Setup OpenVino environment variables.
- Install dependencies: `pip install -r requirements.txt`.
- Test it in Python: `python beecam.py`

If everything is ok, let's build it using `pyinstaller` with the following command.
```
pyinstaller --onedir beecam.py \
	--name beecam \
	--add-binary "[OPENVINO_PATH]\python\python3.8\openvino\inference_engine;openvino/inference_engine" \
	--add-data "config/;config/" \
	--add-data "background.jpg;." \
	--add-data "model.bin;." \
	--add-data "model.xml;." \
	--add-binary "[OPENVINO_PATH]\deployment_tools\inference_engine\bin\intel64\Release\*;." \
	--add-binary "[OPENVINO_PATH]\deployment_tools\ngraph\lib;." \
	--add-data "OBS-VirtualCam;OBS-VirtualCam" \
	--ico beecam.ico --windowed --clean -y
```
Replace `[OPENVINO_PATH]` with your own OpenVino installation folder. The distribution should be appeared in `dist` folder.
## TODO
- [ ] Clean code
- [ ]  Reduce distribution size.
- [ ] Improve the deep learning model.
- [ ]  More elegent UI
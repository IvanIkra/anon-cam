# AnonCam

AnonCam is a real-time face anonymization application using a webcam. The application uses computer vision to detect faces and applies various anonymization methods to protect privacy.

## Features

- **Real-time face detection** - uses MediaPipe for accurate face detection
- **Multiple anonymization modes**:
  - **Auto** - automatically switches between modes depending on whether faces are present
  - **Faces** - anonymizes only detected faces
  - **All** - applies anonymization to the entire frame
  - **None** - disables anonymization
- **Configurable anonymization strength** - from 1 to 10 levels
- **Smooth transitions** - adjustable blur of mask edges
- **Virtual camera** - outputs to a virtual camera for use in other applications
- **Black-and-white mode** - optional monochrome output
- **Detailed tuning** - many parameters for fine control

## Installation

### Requirements

- Python 3.8+
- macOS (tested on macOS)
- Webcam

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Start the application

```bash
python anon_cam.py
```

### Main settings

- **Mode**: Choose the anonymization mode
- **Strength**: Anonymization level (1-10)
- **Feather**: Blur for mask edges (0-60)
- **Only the largest**: Anonymize only the largest face
- **Detection threshold**: Face detection sensitivity (0-100%)
- **B/W output**: Black-and-white mode

### Advanced settings

- **Miss thresh**: Number of frames without faces to switch modes
- **Recover frames**: Number of frames to restore the mode
- **Det every**: Face detection frequency (every N-th frame)
- **Det width**: Frame width for detection (160-960px)
- **Mask expansion**: Expands the anonymized area (0-100%)

## Technical Details

### Architecture

- **`app.py`** - Application entry point
- **`ui.py`** - PyQt6 user interface
- **`engine.py`** - Main anonymization engine
- **`detector.py`** - Face detection using MediaPipe
- **`anonymize.py`** - Anonymization algorithms
- **`logging_utils.py`** - Logging utilities

### Anonymization algorithms

The application uses irreversible anonymization methods:

1. **Quantization** - reduces the number of color levels
2. **Noise addition** - random noise to degrade image quality
3. **Block shuffling** - permutes image blocks
4. **Blur** - Gaussian blur for smoothing

### Performance

- Optimized face detection (every N-th frame)
- Configurable detection resolution
- Caching face detection results
- Efficient frame processing

## Virtual Camera

To use a virtual camera, install an additional dependency:

```bash
pip install pyvirtualcam
```

After installation, enable the "Virtual camera output" option in the interface.

## Logging

The application includes a built-in logging system:

- Click the "Logs" button to open the logs window
- Logs show face detection activity information
- Switches between anonymization modes
- Errors and warnings

## Troubleshooting

### Camera issues

- Make sure the camera is not used by other applications
- Check camera access permissions in system settings

### Low performance

- Increase "Det every" (detect less often)
- Decrease "Det width" (lower detection resolution)
- Decrease the "Strength" of anonymization

### Virtual camera issues

- Make sure the `pyvirtualcam` package is installed
- Check that the virtual camera is not used by other applications

## License

This project is distributed under the MIT license. See the `LICENSE` file for details.

## Contributing

We welcome contributions to the project. Please:

1. Fork the repository
2. Create a branch for a new feature
3. Make your changes
4. Open a Pull Request

## Support

If you have problems or questions:

1. Check the "Troubleshooting" section
2. Create an Issue in the GitHub repository
3. Attach logs from the application's "Logs" window
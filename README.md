# Nearest-Neighbor Visual Learning

A research project that learns to predict the timing of future visual events from video streams. Video frames are converted into binary events (simulating an event-based camera), and a neural network learns to predict, for each pixel, how many frames until it next changes — using only how recently it last changed as input.

Developed from November 2022 to January 2023.

## Approach

### Event-Based Input

Standard video frames are converted into binary events by comparing consecutive frames: a pixel produces a positive event if its brightness increased beyond a threshold, or a negative event if it decreased. This simulates a neuromorphic event camera, reducing each frame to a sparse binary representation of what changed.

### Proximal Times

The system encodes temporal context as "proximal times" — for each pixel, the number of frames since it last produced an event (left proximal) and the number of frames until it next produces one (right proximal). These integer distances are converted to continuous values using exponential decay: `exp(-proximal_time * 0.06)`, so recent events have high values and distant events decay toward zero.

### Prediction Network

A single-hidden-layer MLP (multi-layer perceptron) maps left-proximal values to right-proximal values — given when each pixel last changed, predict when it will next change. The network trains continuously in an online fashion: each frame provides one training example from the middle of a sliding temporal window, and one prediction from the current moment. Prediction error is tracked as the mean relative difference between actual and predicted future event times.

An optional context feature feeds the hidden layer's activations (converted to events) back as additional input, though this was disabled in the final configuration due to performance cost.

### Architecture

```
VideoPlaybackSensor → EventPreProcessor → ContinuousEventBrain → BasicVisualizer
```

- **VideoPlaybackSensor** loads video frames from disk, caching to pickle files for efficient replay
- **EventPreProcessor** simulates an event camera via frame differencing with a brightness threshold
- **ContinuousEventBrain** maintains a circular buffer of recent events, computes proximal times (using Cython for speed), trains and queries the MLP each step
- **BasicVisualizer** displays input frames and brain outputs via OpenCV

## Project Structure

```
run_demo_vision.py            # Main demo: single long-running simulation
run_sweep_vision.py           # Hyperparameter sweep (e.g., hidden layer size)
run_demo_ib.py                # Stub for financial time-series input (unused)
check_for_tf_gpu.py           # Print available TensorFlow GPUs

brain/
  continuous_event_brain.py   # Core: online MLP training and proximal-time prediction
  event_predict_brain.py      # Earlier batch-training variant
  tf_mlp.py                   # TensorFlow 2 (eager) MLP wrapper
  tf_1_mlp.py                 # TensorFlow 1.x compat-mode MLP (used by default)

sensors/
  video_playback.py           # Video loading with frame caching

preprocessors/
  event_pre_processor.py      # Frame-to-event conversion (simulated event camera)

utils/
  states_history.py           # Circular buffers and proximal-time computation
  sim_folder_manager.py       # Output directory management
  fps_counter.py              # Performance monitoring
  rf_im_maker.py              # Weight visualization utility

visualizers/
  basic_visualizer.py         # Real-time OpenCV display

cython_proximal/
  cython_proxim.pyx           # Cython-accelerated proximal distance computation

ib_utils/
  data_downloader.py          # Interactive Brokers data download (stub)
```

## Dependencies

Listed in `requirements.txt`:

- numpy, scipy
- tensorflow (1.x compat mode used by default)
- opencv-python
- matplotlib
- cython (for accelerated proximal computation)
- scikit-learn (legacy, no longer actively used)
- ib_insync (for financial data mode, not used in vision experiments)

## Running

### Demo

```bash
python run_demo_vision.py
```

Prompts for a simulation name prefix, then processes video frames continuously. Default configuration:

- Input: 8x8 pixel crops from an 11-hour sea turtle video
- Brightness threshold: 2/255
- Hidden layer size: 0.5x the input dimension (64 hidden units for 128-dim input)
- Learning rate: 0.01
- Max prediction horizon: 100 frames
- Runs for up to 5M steps

Plots are saved every 30 seconds to the output directory:
- `actual_g_predict_b_*.png` — scatter plot of actual vs. predicted event times
- `predict_error.png` — prediction error over time
- `raster_input_*.png` — raster plot of input events
- `raster_interm_*.png` — raster plot of hidden-layer activations

To change the video source, edit `video_filename` in `get_sensors_params()`. To adjust the image crop size, change `RF_IM_DIM` in the `main_params` dict in `demo()`.

### Hyperparameter Sweep

```bash
python run_sweep_vision.py
```

Runs the same pipeline with different values of a chosen parameter (default: `hidden_state_factor` over [0.1, 0.5, 1, 5]). Generates comparison plots of prediction error vs. parameter value. Currently runs sequentially; multiprocessing support is planned but not implemented.

### Notes

- Video files are expected at `{ROOT_DIR}/projects/video-downloads/`. Adjust `ROOT_DIR` in the main params dict.
- Output goes to `{ROOT_DIR}/projects/NL-sim-venv/{prefix}_{timestamp}/`.
- The Cython proximal module can be built with `rebuild_cython_libraries.sh` for faster proximal-time computation.
- TensorFlow 1.x compatibility mode (`tf.compat.v1.disable_eager_execution()`) is used by default for training speed. The TF2 eager-mode wrapper (`tf_mlp.py`) is available but slower.

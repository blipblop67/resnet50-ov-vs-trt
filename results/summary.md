| System | Config | p50 (ms) | p50 range | p95 (ms) | FPS | Power (W) | FPS/W |
|---|---|---|---|---|---|---|---|
| Jetson Orin Nano | tensorrt-gpu int8 * | 1.22 | 1.22-1.22 | 1.26 | 788.0 | 17.1 | 46.1 |
| Jetson Orin Nano | tensorrt-gpu fp16 | 1.93 | 1.93-1.93 | 2.00 | 503.5 | 19.8 | 25.5 |
| Jetson Orin Nano | tensorrt-gpu fp32 | 3.74 | 3.73-3.74 | 4.36 | 256.0 | 21.0 | 12.2 |
| Laptop (i7-8665U) | openvino-cpu int8 | 19.80 | 19.49-21.12 | 30.68 | 46.3 |  |  |
| Laptop (i7-8665U) | openvino-gpu int8 | 19.80 | 19.75-20.15 | 20.95 | 50.6 |  |  |
| Laptop (i7-8665U) | openvino-gpu fp16 | 32.51 | 31.78-32.99 | 37.66 | 29.9 |  |  |
| Laptop (i7-8665U) | openvino-cpu fp32 | 34.73 | 33.49-43.53 | 47.23 | 27.3 |  |  |
| Laptop (i7-8665U) | onnxruntime-cpu fp32 | 43.70 | 42.90-43.74 | 50.95 | 24.1 |  |  |
| Laptop (i7-8665U) | openvino-gpu fp32 | 48.78 | 40.96-49.36 | 61.25 | 19.5 |  |  |

\* TensorRT INT8 engine built without calibration data: speed only, accuracy not valid.
Medians over 3-3 rounds; range = min-max of per-round p50.

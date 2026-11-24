# Jetson setup (TensorRT benchmarks)

- Board: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
- L4T: R36 revision 5.0 (36.5.0), from /etc/nv_tegra_release
- TensorRT: 10.3.0.30 (CUDA 12.5 packages), trtexec at /usr/src/tensorrt/bin/trtexec
- Power mode: MAXN_SUPER (nvpmodel ID 2), checked with `sudo nvpmodel -q`
- Clocks: locked with `sudo jetson_clocks` before each session (resets on reboot)
- Memory: 7.4 GiB total; desktop (GUI) running during setup
- Method: batch 1, same ONNX file as the laptop (sha256 in models/resnet50.onnx.sha256)

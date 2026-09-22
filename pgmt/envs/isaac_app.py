"""Headless single-GPU launcher with an explicit Vulkan device override.

CUDA and Vulkan indices can differ when a GPU is unavailable. The optional
PGMT_RENDER_GPU is a Vulkan index from Kit's device table, not nvidia-smi.
The caller must still select the correct CUDA device independently.
"""
import os


def launch_isaac_app(device):
    from isaaclab.app import AppLauncher

    kit_args = ("--/renderer/multiGpu/enabled=False "
                "--/renderer/multiGpu/autoEnable=False "
                "--/renderer/multiGpu/maxGpuCount=1")
    render_gpu = os.environ.get("PGMT_RENDER_GPU")
    if render_gpu is not None:
        if not render_gpu.isdecimal():
            raise ValueError("PGMT_RENDER_GPU must be a nonnegative Vulkan device index")
        kit_args += f" --/renderer/activeGpu={render_gpu}"
    return AppLauncher(headless=True, device=str(device), multi_gpu=False,
                       kit_args=kit_args).app

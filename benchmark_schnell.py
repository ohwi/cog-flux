import itertools
import os
import time
from typing import Any, Dict, Optional

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark_limit = 20

import logging
from attr import dataclass
from flux.math import set_attention_mode
from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from fp8.flux_pipeline import FluxPipeline
from fp8.util import LoadedModels

from loguru import logger
import numpy as np
from einops import rearrange
from PIL import Image
from typing import List
from torchvision import transforms
from cog import BasePredictor, Input, Path
from flux.util import load_ae, load_clip, load_flow_model, load_t5, download_weights

# Suppress diffusers nsfw warnings
logging.getLogger("diffusers").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)

MAX_IMAGE_SIZE = 1440

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "21:9": (1536, 640),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:5": (896, 1088),
    "5:4": (1088, 896),
    "3:4": (896, 1152),
    "4:3": (1152, 896),
    "9:16": (768, 1344),
    "9:21": (640, 1536),
}


class Predictor(BasePredictor):
    def setup(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def base_setup(
            self,
            flow_model_name: str,
            compile_fp8: bool = False,
            compile_bf16: bool = False,
            max_autotune: bool = False,
    ) -> None:
        self.flow_model_name = flow_model_name
        print(f"Booting model {self.flow_model_name}")

        gpu_name = (
            os.popen("nvidia-smi --query-gpu=name --format=csv,noheader,nounits")
            .read()
            .strip()
        )
        print("Detected GPU:", gpu_name)

        # need > 48 GB of ram to store all models in VRAM
        self.offload = "A40" in gpu_name

        device = "cuda"
        max_length = 256 if self.flow_model_name == "flux-schnell" else 512
        self.t5 = load_t5(device, max_length=max_length)
        self.clip = load_clip(device)
        self.flux = load_flow_model(
            self.flow_model_name, device="cpu" if self.offload else device
        )
        self.flux = self.flux.eval()
        self.ae = load_ae(
            self.flow_model_name, device="cpu" if self.offload else device
        )

        self.num_steps = 4 if self.flow_model_name == "flux-schnell" else 28
        self.shift = self.flow_model_name != "flux-schnell"
        self.compile_run = False

        shared_models = LoadedModels(
            flow=None, ae=self.ae, clip=self.clip, t5=self.t5, config=None
        )

        # fp8 only works w/compute capability >= 8.9
        self.disable_fp8 = False
        if compile_fp8 and torch.cuda.get_device_capability() >= (8, 9):
            self.fp8_pipe = FluxPipeline.load_pipeline_from_config_path(
                f"fp8/configs/my-config-1-{flow_model_name}-h100.json",
                shared_models=shared_models,
            )
            self.compile_fp8()
        else:
            self.disable_fp8 = True

        if compile_bf16:
            self.compile_bf16()

        if max_autotune:
            self.autotune()

    def compile_fp8(self):
        print("compiling fp8 model")
        st = time.time()
        self.fp8_pipe.generate(
            prompt="a cool dog",
            width=1344,
            height=768,
            num_steps=self.num_steps,
            guidance=3,
            seed=123,
            compiling=True,
        )

        for k in ASPECT_RATIOS:
            print(f"warming kernel for {k}")
            width, height = self.aspect_ratio_to_width_height(k)
            self.fp8_pipe.generate(
                prompt="godzilla!", width=width, height=height, num_steps=4, guidance=3
            )
            self.fp8_pipe.generate(
                prompt="godzilla!",
                width=width // 2,
                height=height // 2,
                num_steps=4,
                guidance=3,
            )

        print("compiled in ", time.time() - st)

    def compile_bf16(self):
        print("compiling bf16 model")
        st = time.time()

        self.compile_run = True
        width, height = self.aspect_ratio_to_width_height("1:1")
        self.base_predict(
            prompt="a cool dog",
            width=width,
            height=height,
            num_outputs=1,
            num_inference_steps=self.num_steps,
            guidance=3.5,
            seed=123,
        )
        print("compiled in ", time.time() - st)

    def autotune(self):
        def _autotune(m):
            return torch.compile(
                m,
                mode="max-autotune",
                fullgraph=True,
            )
        self.flux = _autotune(self.flux)
        self.ae = _autotune(self.ae)

        # has some dynamo unsupported functions
        # self.clip = _autotune(self.clip)
        # self.t5 = _autotune(self.t5)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        return ASPECT_RATIOS.get(aspect_ratio)

    def get_image(self, image: str):
        if image is None:
            return None
        image = Image.open(image).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 2.0 * x - 1.0),
            ]
        )
        img: torch.Tensor = transform(image)
        return img[None, ...]

    def predict(self, *args, **kwargs):
        raise Exception("You need to instantiate a predictor for a specific flux model")

    def preprocess(
            self, aspect_ratio: str, seed: Optional[int], megapixels: str
    ) -> Dict:
        width, height = ASPECT_RATIOS.get(aspect_ratio)
        if megapixels == "0.25":
            width, height = width // 2, height // 2

        if not seed:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        return {"width": width, "height": height, "seed": seed}

    @torch.inference_mode()
    def base_predict(
            self,
            prompt: str,
            num_outputs: int,
            num_inference_steps: int,
            guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
            image: Path = None,  # img2img for flux-dev
            prompt_strength: float = 0.8,
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
    ) -> List[Path]:
        """Run a single prediction on the model"""
        torch_device = torch.device("cuda")
        init_image = None

        # img2img only works for flux-dev
        if image:
            print("Image detected - settting to img2img mode")
            init_image = self.get_image(image)
            width = init_image.shape[-1]
            height = init_image.shape[-2]
            print(f"Input image size: {width}x{height}")

            # Calculate the scaling factor if the image exceeds MAX_IMAGE_SIZE
            scale = min(MAX_IMAGE_SIZE / width, MAX_IMAGE_SIZE / height, 1)
            if scale < 1:
                width = int(width * scale)
                height = int(height * scale)
                print(f"Scaling image down to {width}x{height}")

            # Round image width and height to nearest multiple of 16
            width = round(width / 16) * 16
            height = round(height / 16) * 16
            print(f"Input image size set to: {width}x{height}")

            # Resize
            init_image = init_image.to(torch_device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae.encoder.to(torch_device)
            init_image = self.ae.encode(init_image)
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

        # prepare input
        x = get_noise(
            num_outputs,
            height,
            width,
            device=torch_device,
            dtype=torch.bfloat16,
            seed=seed,
        )
        timesteps = get_schedule(
            num_inference_steps, (x.shape[-1] * x.shape[-2]) // 4, shift=self.shift
        )

        if init_image is not None:
            t_idx = int((1.0 - prompt_strength) * num_inference_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(torch_device), self.clip.to(torch_device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=[prompt] * num_outputs)

        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.flux = self.flux.to(torch_device)

        x, flux = denoise(
            self.flux,
            **inp,
            timesteps=timesteps,
            guidance=guidance,
            compile_run=self.compile_run,
        )

        if self.compile_run:
            self.compile_run = False
            self.flux = flux

        if self.offload:
            self.flux.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        np_images = [
            (127.5 * (rearrange(x[i], "c h w -> h w c").clamp(-1, 1) + 1.0))
            .cpu()
            .byte()
            .numpy()
            for i in range(num_outputs)
        ]
        images = [Image.fromarray(img) for img in np_images]
        return images, np_images

    def fp8_predict(
            self,
            prompt: str,
            num_outputs: int,
            num_inference_steps: int,
            guidance: float = 3.5,  # schnell ignores guidance within the model, fine to have default
            image: Path = None,  # img2img for flux-dev
            prompt_strength: float = 0.8,
            seed: int = None,
            width: int = 1024,
            height: int = 1024,
    ):
        """Run a single prediction on the model"""
        logger.info("Running quantized prediction")

        return self.fp8_pipe.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_inference_steps,
            guidance=guidance,
            seed=seed,
            init_image=image,
            strength=prompt_strength,
            num_images=num_outputs,
        )

    def postprocess(
            self,
            images,
            disable_safety_checker: bool,
            output_format: str,
            output_quality: int,
            np_images: Optional = None,
    ) -> List[Path]:
        has_nsfw_content = [False] * len(images)

        if not np_images:
            np_images = [np.array(val) for val in images]

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(images, np_images)

        output_paths = []
        for i, (img, is_nsfw) in enumerate(zip(images, has_nsfw_content)):
            if is_nsfw:
                try:
                    falcon_is_safe = self.run_falcon_safety_checker(img)
                except Exception as e:
                    print(f"Error running safety checker: {e}")
                    falcon_is_safe = False
                if not falcon_is_safe:
                    print(f"NSFW content detected in image {i}")
                    continue

            output_path = f"out-{i}.{output_format}"
            save_params = (
                {"quality": output_quality, "optimize": True}
                if output_format != "png"
                else {}
            )
            img.save(output_path, **save_params)
            output_paths.append(Path(output_path))

        if not output_paths:
            raise Exception(
                "All generated images contained NSFW content. Try running it again with a different prompt."
            )

        print(f"Total safe images: {len(output_paths)} out of {len(images)}")
        return output_paths

    def run_safety_checker(self, images, np_images):
        safety_checker_input = self.feature_extractor(images, return_tensors="pt").to(
            "cuda"
        )
        image, has_nsfw_concept = self.safety_checker(
            images=np_images,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    def run_falcon_safety_checker(self, image):
        with torch.no_grad():
            inputs = self.falcon_processor(images=image, return_tensors="pt")
            outputs = self.falcon_model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            result = self.falcon_model.config.id2label[predicted_label]

        return result == "normal"


class SchnellPredictor(Predictor):
    def setup(
            self,
            compile_fp8: bool = False,
            compile_bf16: bool = False,
            max_autotune: bool = False,
    ) -> None:
        self.base_setup(
            "flux-schnell",
            compile_fp8=compile_fp8,
            compile_bf16=compile_bf16,
            max_autotune=max_autotune,
        )

    def predict(
            self,
            prompt: str,
            aspect_ratio: str = "1:1",
            num_outputs: int = 1,
            seed: int = None,
            output_format: str = "webp",
            output_quality: int = 80,
            go_fast: bool = True,
            megapixels: str = "1",
            disable_safety_checker: bool = True,
            num_inference_steps: int = 4,
    ) -> List[Path]:

        hws_kwargs = self.preprocess(aspect_ratio, seed, megapixels)

        # overwrite
        hws_kwargs['width'] = 768
        hws_kwargs['height'] = 1024

        if go_fast and not self.disable_fp8:
            logger.info("Running FP8 model")

            imgs, np_imgs = self.fp8_predict(
                prompt,
                num_outputs,
                num_inference_steps=num_inference_steps,
                **hws_kwargs,
            )
        else:
            if self.disable_fp8:
                logger.info("running bf16 model, fp8 disabled")

            imgs, np_imgs = self.base_predict(
                prompt,
                num_outputs,
                num_inference_steps=num_inference_steps,
                **hws_kwargs,
            )

        return imgs


# https://github.com/ai-compiler-study/flux/blob/main/benchmark/benchmark_flux.py#L16
def _benchmark(iters, f, *args, **kwargs):
    # warmup
    for _ in range(10):
        f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    # elapsed_time has a resolution of 0.5 microseconds:
    # but returns milliseconds, so we need to multiply it to increase resolution
    return start_event.elapsed_time(end_event) / iters, *f(*args, **kwargs)


def benchmark(
        *,
        compile_fp8,
        compile_bf16,
        max_autotune,
        attention_mode,
        **kwargs,
):
    set_attention_mode(attention_mode)
    predictor = SchnellPredictor()
    predictor.setup(
        compile_fp8=compile_fp8,
        compile_bf16=compile_bf16,
        max_autotune=max_autotune,
    )

    prompt = "a cat in a hat"

    result = _benchmark(
        iters=100,
        f=predictor.predict,
        prompt=prompt,
    )
    return result[0]


if __name__ == '__main__':
    # Need to prepare models below mode-cache
    # See flux/util.py and wget/unzip each files

    benchmark_kwargs = list()

    # just for testing spending time
    benchmark_kwargs.append({
        "compile_fp8": True,
        "compile_bf16": False,
        "max_autotune": False,
        "attention_mode": "fa3",
    })
    # benchmark_kwargs.append({
    #     "compile_fp8": True,
    #     "compile_bf16": False,
    #     "max_autotune": False,
    #     "attention_mode": "int8",
    # })

    # benchmark_kwargs.append({
    #     "compile_fp8": True,
    #     "compile_bf16": False,
    #     "max_autotune": True,
    #     "attention_mode": "replicate",
    # })
    # benchmark_kwargs.append({
    #     "compile_fp8": True,
    #     "compile_bf16": False,
    #     "max_autotune": False,
    #     "attention_mode": "replicate",
    # })
    # benchmark_kwargs.append({
    #     "compile_fp8": True,
    #     "compile_bf16": False,
    #     "max_autotune": False,
    #     "attention_mode": "naive",
    # })
    # benchmark_kwargs.append({
    #     "compile_fp8": False,
    #     "compile_bf16": True,
    #     "max_autotune": True,
    #     "attention_mode": "naive",
    # })
    # benchmark_kwargs.append({
    #     "compile_fp8": False,
    #     "compile_bf16": True,
    #     "max_autotune": False,
    #     "attention_mode": "naive",
    # })
    # benchmark_kwargs.append({
    #     "compile_fp8": False,
    #     "compile_bf16": True,
    #     "max_autotune": False,
    #     "attention_mode": "replicate",
    # })

    results = []
    for kwargs in benchmark_kwargs:
        avg_time = benchmark(**kwargs)
        results.append({
            "avg time/iter": f"{avg_time:.4f}",
            "compile_fp8": str(kwargs["compile_fp8"]),
            "compile_bf16": str(kwargs["compile_bf16"]),
            "max_autotune": str(kwargs["max_autotune"]),
            "attention": str(kwargs["attention_mode"]),
        })

    headers = results[0].keys()
    print(", ".join(headers))
    for result in results:
        print(", ".join(result.values()))

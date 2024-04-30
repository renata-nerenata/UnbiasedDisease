import inspect
from itertools import repeat
from typing import Callable, List, Optional, Union

from src.constants import *

import torch

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging
from . import UnbiasedDiseasePipelineOutput

logger = logging.get_logger(__name__)


class UnbiasedDiseasePipeline(DiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def enable_attention_slicing(self, slice_size):
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = PROMPT,
        height: int = HEIGHT,
        width: int = WIDTH,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        guidance_scale: float = GUIDANCE_SCALE,
        negative_prompt: Optional[Union[str, List[str]]] = NEGATIVE_PROMPT,
        num_images_per_prompt: Optional[int] = NUM_IMAGES_PER_PROMPT,
        eta: float = ETA,
        generator: Optional[torch.Generator] = GENERATOR,
        latents: Optional[torch.FloatTensor] = LATENTS,
        output_type: Optional[str] = OUTPUT_TYPE,
        return_dict: bool = RETURN_DICT,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = CALLBACK,
        callback_steps: Optional[int] = CALLBACK_STEPS,
        editing_prompt: Optional[Union[str, List[str]]] = EDITING_PROMPT,
        editing_prompt_prompt_embeddings=EDITING_PROMPT_PROMPT_EMBEDDINGS,
        reverse_editing_direction: Optional[
            Union[bool, List[bool]]
        ] = REVERSE_EDITING_DIRECTION,
        edit_guidance_scale: Optional[Union[float, List[float]]] = EDIT_GUIDANCE_SCALE,
        edit_warmup_steps: Optional[Union[int, List[int]]] = EDIT_WARMUP_STEPS,
        edit_cooldown_steps: Optional[Union[int, List[int]]] = EDIT_COOLDOWN_STEPS,
        edit_threshold: Optional[Union[float, List[float]]] = EDIT_THRESHOLD,
        edit_momentum_scale: Optional[float] = EDIT_MOMENTUM_SCALE,
        edit_mom_beta: Optional[float] = EDIT_MOM_BETA,
        edit_weights: Optional[List[float]] = EDIT_WEIGHTS,
        sem_guidance=SEM_GUIDANCE,
        **kwargs,
    ):
        batch_size = len(prompt)
        if editing_prompt:
            enable_edit_guidance = True
            if isinstance(editing_prompt, str):
                editing_prompt = [editing_prompt]
            enabled_editing_prompts = len(editing_prompt)
        elif editing_prompt_prompt_embeddings is not None:
            enable_edit_guidance = True
            enabled_editing_prompts = editing_prompt_prompt_embeddings.shape[0]
        else:
            enabled_editing_prompts = 0
            enable_edit_guidance = False

        input_prompt = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = input_prompt.input_ids

        if input_ids.shape[-1] > self.tokenizer.model_max_length:
            input_ids = input_ids[:, : self.tokenizer.model_max_length]
        text_embeddings = self.text_encoder(input_ids.to(self.device))[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_images_per_prompt, 1)
        text_embeddings = text_embeddings.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        if enable_edit_guidance:

            if editing_prompt_prompt_embeddings is None:
                edit_concepts_input = self.tokenizer(
                    [x for item in editing_prompt for x in repeat(item, batch_size)],
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                )

                edit_concepts_input_ids = edit_concepts_input.input_ids

                if edit_concepts_input_ids.shape[-1] > self.tokenizer.model_max_length:
                    edit_concepts_input_ids = edit_concepts_input_ids[
                        :, : self.tokenizer.model_max_length
                    ]
                edit_concepts = self.text_encoder(
                    edit_concepts_input_ids.to(self.device)
                )[0]
            else:
                edit_concepts = editing_prompt_prompt_embeddings.to(self.device).repeat(
                    batch_size, 1, 1
                )

            edit_batch_size, edit_sequence_length, _ = edit_concepts.shape
            edit_concepts = edit_concepts.repeat(1, num_images_per_prompt, 1)
            edit_concepts = edit_concepts.view(
                edit_batch_size * num_images_per_prompt, edit_sequence_length, -1
            )

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            unconditional_tokens = [""]

            max_length = input_ids.shape[-1]
            unconditional_input = self.tokenizer(
                unconditional_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            unconditional_embeddings = self.text_encoder(
                unconditional_input.input_ids.to(self.device)
            )[0]

            seq_len = unconditional_embeddings.shape[1]
            unconditional_embeddings = unconditional_embeddings.repeat(
                batch_size, num_images_per_prompt, 1
            )
            unconditional_embeddings = unconditional_embeddings.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            if enable_edit_guidance:
                text_embeddings = torch.cat(
                    [unconditional_embeddings, text_embeddings, edit_concepts]
                )
            else:
                text_embeddings = torch.cat([unconditional_embeddings, text_embeddings])

        latents_shape = (
            batch_size * num_images_per_prompt,
            self.unet.in_channels,
            height // 8,
            width // 8,
        )
        latents_dtype = text_embeddings.dtype
        if latents is None:
            latents = torch.randn(
                latents_shape,
                generator=generator,
                device=self.device,
                dtype=latents_dtype,
            )
        else:
            latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        edit_momentum = None

        self.uncond_estimates = None
        self.text_estimates = None
        self.edit_estimates = None
        self.sem_guidance = None

        for i, t in enumerate(self.progress_bar(timesteps_tensor)):

            latent_model_input = (
                torch.cat([latents] * (2 + enabled_editing_prompts))
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            if do_classifier_free_guidance:
                noise_pred_out = noise_pred.chunk(2 + enabled_editing_prompts)
                noise_pred_uncond, noise_pred_text = (
                    noise_pred_out[0],
                    noise_pred_out[1],
                )
                noise_pred_edit_concepts = noise_pred_out[2:]

                noise_guidance = guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.uncond_estimates is None:
                    self.uncond_estimates = torch.zeros(
                        (num_inference_steps + 1, *noise_pred_uncond.shape)
                    )
                self.uncond_estimates[i] = noise_pred_uncond.detach().cpu()

                if self.text_estimates is None:
                    self.text_estimates = torch.zeros(
                        (num_inference_steps + 1, *noise_pred_text.shape)
                    )
                self.text_estimates[i] = noise_pred_text.detach().cpu()

                if self.edit_estimates is None and enable_edit_guidance:
                    self.edit_estimates = torch.zeros(
                        (
                            num_inference_steps + 1,
                            len(noise_pred_edit_concepts),
                            *noise_pred_edit_concepts[0].shape,
                        )
                    )

                if self.sem_guidance is None:
                    self.sem_guidance = torch.zeros(
                        (num_inference_steps + 1, *noise_pred_text.shape)
                    )

                if edit_momentum is None:
                    edit_momentum = torch.zeros_like(noise_guidance)

                if enable_edit_guidance:

                    concept_weights = torch.zeros(
                        (len(noise_pred_edit_concepts), noise_guidance.shape[0]),
                        device=self.device,
                    )
                    noise_guidance_edit = torch.zeros(
                        (len(noise_pred_edit_concepts), *noise_guidance.shape),
                        device=self.device,
                    )

                    warmup_inds = []
                    for c, noise_pred_edit_concept in enumerate(
                        noise_pred_edit_concepts
                    ):
                        self.edit_estimates[i, c] = noise_pred_edit_concept
                        if isinstance(edit_guidance_scale, list):
                            edit_guidance_scale_c = edit_guidance_scale[c]
                        else:
                            edit_guidance_scale_c = edit_guidance_scale

                        if isinstance(edit_threshold, list):
                            edit_threshold_c = edit_threshold[c]
                        else:
                            edit_threshold_c = edit_threshold
                        if isinstance(reverse_editing_direction, list):
                            reverse_editing_direction_c = reverse_editing_direction[c]
                        else:
                            reverse_editing_direction_c = reverse_editing_direction
                        if edit_weights:
                            edit_weight_c = edit_weights[c]
                        else:
                            edit_weight_c = 1.0
                        if isinstance(edit_warmup_steps, list):
                            edit_warmup_steps_c = edit_warmup_steps[c]
                        else:
                            edit_warmup_steps_c = edit_warmup_steps

                        if isinstance(edit_cooldown_steps, list):
                            edit_cooldown_steps_c = edit_cooldown_steps[c]
                        elif edit_cooldown_steps is None:
                            edit_cooldown_steps_c = i + 1
                        else:
                            edit_cooldown_steps_c = edit_cooldown_steps
                        if i >= edit_warmup_steps_c:
                            warmup_inds.append(c)
                        if i >= edit_cooldown_steps_c:
                            noise_guidance_edit[c, :, :, :, :] = torch.zeros_like(
                                noise_pred_edit_concept
                            )
                            continue

                        noise_guidance_edit_tmp = (
                            noise_pred_edit_concept - noise_pred_uncond
                        )

                        tmp_weights = (noise_guidance - noise_pred_edit_concept).sum(
                            dim=(1, 2, 3)
                        )

                        tmp_weights = torch.full_like(tmp_weights, edit_weight_c)
                        if reverse_editing_direction_c:
                            noise_guidance_edit_tmp = noise_guidance_edit_tmp * -1
                        concept_weights[c, :] = tmp_weights

                        noise_guidance_edit_tmp = (
                            noise_guidance_edit_tmp * edit_guidance_scale_c
                        )
                        tmp = torch.quantile(
                            torch.abs(noise_guidance_edit_tmp).flatten(start_dim=2),
                            edit_threshold_c,
                            dim=2,
                            keepdim=False,
                        )
                        noise_guidance_edit_tmp = torch.where(
                            torch.abs(noise_guidance_edit_tmp) >= tmp[:, :, None, None],
                            noise_guidance_edit_tmp,
                            torch.zeros_like(noise_guidance_edit_tmp),
                        )
                        noise_guidance_edit[c, :, :, :, :] = noise_guidance_edit_tmp

                    warmup_inds = torch.tensor(warmup_inds).to(self.device)
                    if len(noise_pred_edit_concepts) > warmup_inds.shape[0] > 0:
                        concept_weights = concept_weights.to("cpu")
                        noise_guidance_edit = noise_guidance_edit.to("cpu")

                        concept_weights_tmp = torch.index_select(
                            concept_weights.to(self.device), 0, warmup_inds
                        )
                        concept_weights_tmp = torch.where(
                            concept_weights_tmp < 0,
                            torch.zeros_like(concept_weights_tmp),
                            concept_weights_tmp,
                        )
                        concept_weights_tmp = (
                            concept_weights_tmp / concept_weights_tmp.sum(dim=0)
                        )

                        noise_guidance_edit_tmp = torch.index_select(
                            noise_guidance_edit.to(self.device), 0, warmup_inds
                        )
                        noise_guidance_edit_tmp = torch.einsum(
                            "cb,cbijk->bijk",
                            concept_weights_tmp,
                            noise_guidance_edit_tmp,
                        )
                        noise_guidance_edit_tmp = noise_guidance_edit_tmp
                        noise_guidance = noise_guidance + noise_guidance_edit_tmp

                        self.sem_guidance[i] = noise_guidance_edit_tmp.detach().cpu()

                        del noise_guidance_edit_tmp
                        del concept_weights_tmp
                        concept_weights = concept_weights.to(self.device)
                        noise_guidance_edit = noise_guidance_edit.to(self.device)

                    concept_weights = torch.where(
                        concept_weights < 0,
                        torch.zeros_like(concept_weights),
                        concept_weights,
                    )

                    concept_weights = torch.nan_to_num(concept_weights)
                    noise_guidance_edit = torch.einsum(
                        "cb,cbijk->bijk", concept_weights, noise_guidance_edit
                    )

                    noise_guidance_edit = (
                        noise_guidance_edit + edit_momentum_scale * edit_momentum
                    )

                    edit_momentum = (
                        edit_mom_beta * edit_momentum
                        + (1 - edit_mom_beta) * noise_guidance_edit
                    )

                    if warmup_inds.shape[0] == len(noise_pred_edit_concepts):
                        noise_guidance = noise_guidance + noise_guidance_edit
                        self.sem_guidance[i] = noise_guidance_edit.detach().cpu()

                if sem_guidance is not None:
                    edit_guidance = sem_guidance[i].to(self.device)
                    noise_guidance = noise_guidance + edit_guidance

                noise_pred = noise_pred_uncond + noise_guidance

            latents = self.scheduler.step(
                noise_pred, t, latents, **extra_step_kwargs
            ).prev_sample

            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample

        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return UnbiasedDiseasePipelineOutput(
            images=image, inappropriate_content_detected=has_nsfw_concept
        )

-------------------------------------------------------------------------------

	THE MODELS FOLDER

	Put the AI models you already have here, so Spellcast3D does not download them
	again - and keep the ones you add: updates and reinstalls never touch this folder.

-------------------------------------------------------------------------------

	QWEN-IMAGE 2.1  (the Images stage's built-in image generator)

		qwen-image\       its AI models, about 11 GB:
		                  qwen-image-2.1-turbo-Q4_K_M.gguf, the Turbo (the default model)
		                  in the size spellcast3d-config.ini picks (from
		                  huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF),
		                  qwen-image-2.1-Q4_K_M.gguf, the full model, 4.2 GB more, only
		                  when you press its button under Model on the Images page (from
		                  huggingface.co/unsloth/Qwen-Image-2.1-GGUF),
		                  qwen_image_2.1_vae_bf16.safetensors (from the vae folder of
		                  huggingface.co/unsloth/Qwen-Image-2.1-FP8),
		                  Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf and
		                  Qwen3-VL-8B-Instruct-mmproj-F16.gguf, the text encoder and
		                  its vision part (from
		                  huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF, where the
		                  second one is called mmproj-F16.gguf)

	Spellcast3D downloads them by itself (in the background, the first time; the full
	model when you ask for it). If you already have them - for example from another copy
	of Spellcast3D - copy them here with these names before the first start: only what is
	missing is then downloaded. The ones in your HuggingFace download folder (on the same
	drive) are found by themselves.

	Another model size? Change the "quality" line in spellcast3d-config.ini (in the
	Spellcast3D folder) and start Spellcast3D again: only the Turbo of that size is
	downloaded (the full model and FLUX.3 Klein again from their buttons). A bigger size
	of a model that is already here is used instead of the configured one, with no download.
	A size you no longer use can be deleted from qwen-image\ to free disk space.

	The image model and its VAE are under Qwen's research licence (the Qwen RESEARCH
	LICENSE AGREEMENT); the pictures you make with them are yours, also for commercial
	work (Qwen, 21 Sep 2026). The Qwen3-VL text encoder is Apache-2.0. See
	'README - LICENSE.txt'. Notice:
	"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."

	Prefer Forge Neo with your own models (below)? Switch the backend line in
	spellcast3d-config.ini, then start Spellcast3D again.

	Keep the models on the same drive as Spellcast3D.

	FLUX.3 IMAGE KLEIN  (an optional third model of the Images stage, by Black Forest Labs)

		flux3-klein\      its AI models, about 6.8 GB, downloaded only when you press
		                  its button under Model on the Images page:
		                  flux-3-image-klein-UD-Q4_K_XL.gguf, the model in the size
		                  spellcast3d-config.ini picks (Q4_K_M; the others: UD-Q3_K_XL,
		                  UD-Q5_K_XL, Q6_K, Q8_0; from
		                  huggingface.co/unsloth/flux-3-image-klein-GGUF),
		                  Qwen3VL-4B-Instruct-Q4_K_M.gguf, its text encoder (from
		                  huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) and
		                  flux2-vae.safetensors, its VAE (split_files\vae of
		                  huggingface.co/Comfy-Org/Flux-2-Klein-4B)

	Already have them? Put them here with these names (or the model's GGUF into qwen-image\),
	on the same drive: the download links them instead of downloading them again.

	The model is under the FLUX Kommunity License: free for non-commercial use; commercial use
	of the pictures only for creators and companies under US$5 million yearly revenue. See
	'README - LICENSE.txt' (section 1b).

-------------------------------------------------------------------------------

	IMAGE MODELS  (for the Images stage when spellcast3d-config.ini picks Forge Neo)

		checkpoints\      image models (.safetensors), for example an SDXL model from
		                  civitai.com or huggingface.co. Spellcast3D comes without one.
		loras\            LoRAs
		vae\              VAE files
		text-encoders\    text encoders
		                  (Flux, Z-Image, Qwen-Image and similar models need these two
		                  as separate files)

	A new checkpoint or LoRA appears after you press Refresh next to the model list (or at
	the next start). New VAE and text-encoder files are read when the image generator starts.

	Sub-folders are fine. Already have a big collection somewhere else (another Forge,
	A1111 or ComfyUI)? Don't copy it: add that folder in Settings > Advanced >
	"Extra model folders".

-------------------------------------------------------------------------------

	PIXAL3D  (for the 3D stage)

		pixal3d\          Pixal3D's AI models, about 27 GB.

	Spellcast3D downloads them into this folder by itself (in the background, the first
	time). If you already have them - for example from StableProjectorz's Pixal3D - copy
	what is INSIDE its "MODELS" folder here before the first start:

		pixal3d\dinov3\
		pixal3d\RMBG-2.0\
		pixal3d\hub\
		pixal3d\torch\

	Only what is missing is then downloaded. (If you copy the whole MODELS folder into
	pixal3d\, Spellcast3D moves its contents up for you.)

	Once they are there, Spellcast3D writes beside four of them a copy that is quicker to
	load (the files ending in .model-dtypes.safetensors under pixal3d\hub, 11 GB; 11 GB more
	with the multi-view models). Deleting them only makes the 3D step read the originals,
	which takes longer; Spellcast3D makes them again the next time it updates Pixal3D.

	Pixal3D itself looks in code\pixal3d\MODELS: that is a link to this folder, which
	Spellcast3D makes for you. An older copy of Spellcast3D kept the models in
	code\pixal3d\MODELS; they are moved here at the first start (on the same drive that
	takes a second, nothing is copied).

	Keep the models on the same drive as Spellcast3D.

	HI3D  (the 3D stage's "Shape": finer shapes, refined from Pixal3D's)

		hi3d\             Hi3D's AI models, about 11.4 GB, downloaded with Pixal3D's:
		                  Hi3Dv1.0\ (Hi3D's networks with its LICENSE and NOTICE, from
		                  huggingface.co/hi3d-official/Hi3Dv1.0),
		                  dinov2-large\ (huggingface.co/facebook/dinov2-large),
		                  clip-vit-large-patch14\ (huggingface.co/openai/clip-vit-large-patch14)
		                  and trellis2-shape-encoder\ckpts\ (the two
		                  shape_enc_next_dc_f16c32_fp16 files of
		                  huggingface.co/microsoft/TRELLIS.2-4B)

	Already have them? Put them here in these folders before the first start (or Spellcast3D
	finds the ones in your HuggingFace download folder by itself, on the same drive): only
	what is missing is then downloaded. Without them, props set to Pixal3D work as before,
	and the 3D stage offers the download for props set to Hi3D.

	Hi3D is under the Hi3D Open License Agreement Version 1.0 (models\hi3d\Hi3Dv1.0\LICENSE),
	with its Acceptable Use Policy; DINOv2 is Apache-2.0, CLIP and TRELLIS.2's encoder MIT.
	See 'README - LICENSE.txt' (section 3b). Notice:
	"Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s)."

-------------------------------------------------------------------------------

	RETOPOLOGY  (for the Retopo stage's AI methods, LATO.2 and TriFlow)

		lato2\            LATO.2's AI models, about 4.8 GB:
		                  vflow.pt, vvae.pt, offset_head.pt, tflow.pt, tvae.pt and
		                  voxel_encoder.pt (from huggingface.co/0x4c48/LATO.2), and in
		                  dinov2\ the DINOv2 model it uses:
		                  dinov2\checkpoints\dinov2_vitl14_reg4_pretrain.pth and its code,
		                  dinov2\facebookresearch_dinov2_main\
		triflow\          TriFlow's AI models, about 1.3 GB:
		                  flow_model.safetensors, sdf_vae.safetensors and
		                  nvv_vae.safetensors (from huggingface.co/lihcxr/TriFlow)

	Spellcast3D downloads them by itself (in the background, the first time). If you already
	have them - for example from another copy of Spellcast3D - copy them here with the same
	names and sub-folders before the first start: only what is missing is then downloaded.
	The ones in your HuggingFace download folder (on the same drive) are found by themselves.

	TriFlow reads its models from code\triflow\checkpoints: Spellcast3D links them there for
	you (a link takes no disk space).

	LATO.2 and its models are MIT licensed (DINOv2: Apache-2.0). TriFlow and its models are
	free for NON-COMMERCIAL use only (code\triflow\LICENSE). See 'README - LICENSE.txt'.

	Keep the models on the same drive as Spellcast3D.

-------------------------------------------------------------------------------

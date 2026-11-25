-------------------------------------------------------------------------------

	LICENSES  -  please read this if you want to SELL what you make.

	This is a plain-language summary, not legal advice. The license files themselves
	(listed below) are what count.

-------------------------------------------------------------------------------

SHORT VERSION

	-  Images with Qwen-Image 2.1   the PICTURES you make are YOURS, also for commercial work:
	   (built in, the default):     Qwen said so publicly on 21 September 2026 (section 1).
	                                The MODEL itself is under Qwen's research license (quoted
	                                in section 1), so the page marks it "Research licence".

	-  Images with FLUX.3 Image     free for NON-COMMERCIAL use; the PICTURES may be used
	   Klein (optional download):   commercially only by creators and companies under US$5
	                                million yearly revenue (its FLUX Kommunity License,
	                                section 1b). The page marks it "Kommunity licence".

	-  Images with Forge Neo:       fine for commercial work, IF the image model (checkpoint)
	                                you chose allows it. Check each model's license.

	-  3D meshes (Pixal3D):         treat as NON-COMMERCIAL for now. Pixal3D itself is MIT,
	                                but it relies on parts that only allow non-commercial use
	                                (see below).

	-  3D shapes made with Hi3D     Hi3D's own license allows commercial use and leaves the
	   (the 3D stage's "Shape"):    shapes to you (section 3b), but its meshes still get
	                                Pixal3D's texture and pass through Pixal3D's parts, so
	                                treat them like Pixal3D's meshes for now. The page shows
	                                "Powered by Hi3D 1.0", as Hi3D's license asks.

	-  Retopology with
	   Instant Meshes or LATO.2:    fine for commercial work.

	-  Retopology with TriFlow:     NON-COMMERCIAL use only (a trial). The Retopo stage marks it
	                                "Non-commercial", and saving a model made with it says so.

	-  Projection (baking):         Spellcast3D's own code, fine for commercial work. A bake of a
	                                Pixal3D mesh is still made from that mesh (see above).


-------------------------------------------------------------------------------

SPELLCAST3D ITSELF
	(code\spellcast, the .bat files, prepare_release.py, these text files)

	GNU Affero General Public License v3 (AGPL-3.0), Copyright (c) 2026 Igor Aherne.
	Free to use, change and share, also commercially. If you share a changed version,
	or let people use it over a network, share your source code under the same license.
	Full text: LICENSE (in this folder).

	One exception: the few files that run TriFlow are under the MIT license instead
	(section 7).

	Spellcast3D is "glue" around other people's free tools. It talks to them over a local
	connection on your PC. Each of them keeps its own license:


-------------------------------------------------------------------------------

1.  QWEN-IMAGE 2.1  -  image generation, built in   (models\qwen-image)

	The Images stage's image generator unless you switch to Forge Neo (see "Prefer Forge
	Neo?" at the end of this section).

	THE PICTURES YOU MAKE ARE YOURS - ALSO FOR COMMERCIAL WORK.
	Qwen said so in a public statement on 21 September 2026 (Qwen Developers, @QwenDevs on X:
	https://x.com/QwenDevs/status/2101917379785838660):

	   "Outputs are not part of the licensed Materials. Users retain the rights to images and
	   other content they generate using the model."

	So you may use these pictures in work you sell or are paid for. That is why the page marks
	Qwen-Image 2.1 "Research licence" (the license of the model), not "Non-commercial". (What
	the LATER stages make from a picture follows their own licenses: a Pixal3D mesh, for
	example, see section 3.)

	THE MODEL ITSELF - its weights: the image model and its VAE - is under the
	Qwen RESEARCH LICENSE AGREEMENT (released 20 September 2026),
	Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd.:
	   https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE
	Word for word, its terms for using the model are (the "Materials" are Qwen's models - their
	weights and code - and their documentation, sections 1.e and 1.f):

	   2.a  You are granted a non-exclusive, worldwide, non-transferable and royalty-free
	        limited license [...] to use, reproduce, distribute, copy, create derivative works
	        of, and make modifications to the Materials FOR NON-COMMERCIAL PURPOSES ONLY.
	   1.i  "Non-Commercial" shall mean for research or evaluation purposes only.
	   2.b  You shall not use the Materials for any commercial purpose without obtaining a
	        separate commercial license from us. If you wish to use the Materials commercially,
	        you shall request a license from us at model-business@notice.qwencloud.com.

	How Spellcast3D reads the two together (not legal advice): these terms are about the
	Materials, and Qwen's statement says the pictures are not part of them, so the pictures
	are yours, also for commercial work. The model itself stays under the terms above.

	The license also says:
	-  if you use the model or its outputs to train or improve an AI model that you share or
	   offer to others, show "Built with Qwen" or "Improved using Qwen" in that model's
	   documentation (section 4.b);
	-  don't use "Qwen" as the main name of something you make with it (section 4.c);
	-  follow the export laws that apply to you (section 4.a).
	Its Notice (section 3.c):

	"Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) 2026 Hangzhou Tongyi Laboratory Technology Co., Ltd. All Rights Reserved."

	Spellcast3D does NOT ship these weights. Its installer downloads them on your PC, straight
	from their publishers' HuggingFace pages (pinned versions, checked by SHA-256), into
	models\qwen-image:
	-  the image model Qwen-Image-2.1-Turbo ("Turbo" on the Images page, the default), in the
	   size spellcast3d-config.ini picks: Unsloth's GGUF copies of Qwen/Qwen-Image-2.1-Turbo,
	   huggingface.co/unsloth/Qwen-Image-2.1-Turbo-GGUF (the same license:
	   huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/fc13ba1986fcb905854dc8de2f52eee395f49b5b/LICENSE);
	-  the full Qwen-Image 2.1 ("Full"), only when you press its button on the Images page:
	   Unsloth's GGUF copies of Qwen/Qwen-Image-2.1, huggingface.co/unsloth/Qwen-Image-2.1-GGUF
	   (the same license);
	-  its VAE: huggingface.co/unsloth/Qwen-Image-2.1-FP8, folder vae (the same license);
	-  our albedo add-on (Omni-Texture takes the lighting off each painted view with it), in
	   models\qwen-image\albedo: a LoRA and a fine-tuned VAE decoder for Qwen-Image 2.1, made
	   for Spellcast3D by its author with Marigold V2's recipe (Apache-2.0), from
	   huggingface.co/IgorAherne/spellcast3d-marigold-albedo-qwen-image-2.1 (the same license;
	   its NOTICE beside it: "Built with Qwen"). It learned from Apple's Hypersim (CC BY-SA 3.0)
	   and from Arb-Objaverse's objects (CC-BY and CC0; arb_objaverse_credits.csv beside it
	   names each one's author).
	Using them means accepting that license (its first paragraph says so).

	It also uses:
	-  Qwen3-VL-8B-Instruct by Qwen (the text encoder: it reads your words and your example
	   pictures), Apache-2.0 - free also for commercial use. Downloaded the same way from
	   huggingface.co/unsloth/Qwen3-VL-8B-Instruct-GGUF (Apache-2.0), with its vision part -
	   NOT from Qwen-Image 2.1's own repository, whose copy falls under the research license.
	-  stable-diffusion.cpp by leejet (runs the model on your graphics card): MIT License,
	   Copyright (c) 2023 leejet, with ggml (MIT, Copyright (c) 2023-2026 The ggml authors).
	   Spellcast3D uses its own build of it, changed to make the pictures faster. It is
	   based on leejet's official release master-929 (3f8527a) and on ggml 89c4413, and
	   the changes are under the same MIT License. ggml's SageAttention kernels (attention
	   with INT8 queries and keys) are adapted from SageAttention by thu-ml, Apache-2.0.
	   Its source is public, on github.com/IgorAherne/stable-diffusion.cpp (commit a39d914,
	   branch "spellcast3d") and github.com/IgorAherne/ggml (commit 253aa48, branch
	   "spellcast3d"); the installer downloads the build from the release
	   "spellcast-perf-a39d914" on the first of those pages. After install:
	   code\venvs\qwen\sdcpp\stable-diffusion.cpp.txt, ggml.txt and sageattention.txt (the
	   licenses) and SPELLCAST-BUILD.txt (what it was built from). Unlike the official
	   release, it holds no libwebp and no libwebm.
	-  NVIDIA's CUDA runtime files next to it (cudart64_12, cublas64_12, cublasLt64_12):
	   redistributable under NVIDIA's CUDA Toolkit license. They are linked from a PyTorch
	   already installed here (the same files), else downloaded, unchanged, from leejet's
	   official release of stable-diffusion.cpp (github.com/leejet/stable-diffusion.cpp).
	-  FastAPI, uvicorn and Pillow (its small Python environment): permissive (MIT / BSD style).
	-  For developers only, while a [profiling] line of spellcast3d-config.ini is switched on
	   (measuring where the time of a picture goes): pyinstrument (BSD-3-Clause) or PyTorch for
	   its profiler (BSD-3-Clause), added to that Python environment then. Not otherwise.

	Prefer Forge Neo? It runs the image models YOU choose, under their own licenses (section 2).
	Open spellcast3d-config.ini (in the Spellcast3D folder) with Notepad, put a ; in front of
	"backend = qwen", remove the ; in front of "backend = forge", save, then close Spellcast3D
	and start it again: it installs Forge Neo and keeps Qwen-Image 2.1's files, so switching
	back needs no download. From then on Spellcast3D no longer runs Qwen-Image 2.1. (To remove
	it from your PC too, delete the models\qwen-image folder: about 11 GB, 15 with the full model.)


1b. FLUX.3 IMAGE KLEIN  -  optional image model   (models\flux3-klein)

	A third choice under Model on the Images page, downloaded only when you press its button
	(it is not installed otherwise). FLUX.3 Image Klein is made by Black Forest Labs; the name
	FLUX only says which model it is - Spellcast3D is not made, endorsed or approved by them.

	THE MODEL is under the FLUX Kommunity License v1.0 (Black Forest Labs Inc.):
	   https://huggingface.co/unsloth/flux-3-image-klein-GGUF/blob/12ef13ce920d53f45fe805093910c899668b9e76/LICENSE.md
	In short (section 2.a; not legal advice):
	-  you may use the model for NON-COMMERCIAL purposes;
	-  you may use its OUTPUTS - the pictures - for COMMERCIAL purposes only if you are a
	   "Qualifying User": you, together with your affiliates, have a gross annualized revenue
	   of less than US$5,000,000 (section 1.e);
	-  then section 2.e also asks you to filter or review the pictures for unlawful or
	   infringing content before you share them, and to disclose that they were made with AI
	   where the law requires it.
	Bigger companies need a commercial license from Black Forest Labs (https://bfl.ai/licensing).
	The license also says (section 4): don't use the model or its pictures for military
	purposes, for surveillance, for biometric processing or to make unlawful content (4.a), and
	don't use them to train or improve another AI model that does what FLUX does (4.b); the
	export laws that apply to you hold too.
	Its Notice (section 3.b):

	"This FLUX Model is licensed by Black Forest Labs Inc. under the FLUX Kommunity License. Copyright Black Forest Labs Inc. IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL."

	Spellcast3D does NOT ship these weights. Pressing the button downloads them on your PC from
	HuggingFace (pinned versions, checked by SHA-256):
	-  the model, in the size spellcast3d-config.ini picks: Unsloth's GGUF copies,
	   huggingface.co/unsloth/flux-3-image-klein-GGUF (the same license; Unsloth's notice: they
	   quantized the weights to GGUF);
	-  its text encoder Qwen3-VL-4B-Instruct by Qwen, Apache-2.0:
	   huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF;
	-  FLUX.2's VAE, from huggingface.co/Comfy-Org/Flux-2-Klein-4B (Apache-2.0), the same file as
	   Black Forest Labs' own.
	It runs on the same stable-diffusion.cpp as Qwen-Image 2.1 (section 1).

	To remove it, delete the models\flux3-klein folder (about 7 GB) and pick Turbo or Full.


-------------------------------------------------------------------------------

2.  FORGE NEO  -  image generation   (code\forge)

	Used for the Images stage when spellcast3d-config.ini picks it (backend = forge).

	GNU Affero General Public License v3 (AGPL-3.0).
	Copyright (c) 2023 AUTOMATIC1111, (c) 2024 lllyasviel, (c) 2025 Haoming02.
	See code\forge\LICENSE

	Free to use, also commercially. If you change Forge's code AND let other people use it
	over a network, you must share your changed source code under the same license.

	The IMAGE MODEL you load (the checkpoint) is NOT part of Forge and has its own license.
	Some allow commercial use, some don't. Read the license on the page you downloaded it from.


-------------------------------------------------------------------------------

3.  PIXAL3D  -  image to 3D model   (code\pixal3d)

	Pixal3D's code and model weights: MIT License, Copyright (c) 2026 Tencent.
	See code\pixal3d\LICENSE and code\pixal3d\NOTICE

	Its NOTICE asks for responsible use: no non-consensual sexual content, no sexualized
	depictions of real people, nothing involving minors, no harassment, impersonation or
	privacy-violating uses. It lists these parts it builds on:
		TRELLIS.2 (MIT, Microsoft),  Direct3D-S2 (MIT, DreamTech),  MoGe (MIT, Microsoft),
		dinov2 (Apache-2.0).

	HOWEVER, when it installs, Pixal3D also downloads and uses these, and some of them are
	NOT free for commercial use:

	-  RMBG-2.0 by BRIA AI  (removes the background of your picture before it becomes 3D)
	       Creative Commons Attribution-NonCommercial 4.0  (CC BY-NC 4.0)
	       -> NON-COMMERCIAL only. BRIA sells commercial licenses.
	       (Not run for a picture that has its own transparent background, as Qwen-Image 2.1
	       makes them by default - but the parts below are always used.)
	       After install: code\pixal3d\MODELS\RMBG-2.0\LICENSE.txt

	-  nvdiffrast and nvdiffrec by NVIDIA  (render and texture the 3D model)
	       NVIDIA Source Code License: may only be used "non-commercially", which it defines
	       as "for research or evaluation purposes only".
	       License text inside code\pixal3d\whl\nvdiffrast-*.whl and nvdiffrec_render-*.whl

	-  DINOv3 by Meta  (image features)
	       DINOv3 License: commercial use is allowed, with conditions - e.g. no military,
	       warfare, nuclear, espionage or weapons uses, and you must follow trade sanctions.
	       After install: code\pixal3d\MODELS\dinov3\LICENSE.md

	-  MoGe-2 model weights by Microsoft (camera / depth estimation): MIT.
	-  NAF upsampler by valeo.ai: Apache-2.0.
	-  PyTorch, flash-attention, NATTEN, xformers, triton and the other libraries it installs
	   are under permissive licenses (BSD / MIT / Apache-2.0).

	So, like Trellis: use Pixal3D meshes for learning, personal projects, prototypes and
	evaluation. Don't sell them (or assets made from them) until these parts are replaced
	with commercial-friendly ones.

	If you know good alternatives that can be integrated easily, please tell Igor on Discord.


3b. HI3D 1.0  -  finer 3D shapes   (code\hi3d, models\hi3d)

	The 3D stage's "Shape": per prop, Hi3D (every new prop's) or Pixal3D's own. Hi3D refines
	the coarse shape Pixal3D makes into a finer one; the texture is Pixal3D's either way, so
	section 3 holds for every mesh of the 3D stage. Hi3D runs in Pixal3D's program.

	Hi3D's code and model weights are under the Hi3D Open License Agreement Version 1.0
	(October 22, 2026) of OCEANO HK HOLDING LIMITED. Its full text:
	   https://huggingface.co/hi3d-official/Hi3Dv1.0/blob/d502e938cbedc6e87aba8884614c5ca6d177a4b6/LICENSE
	and, after install, models\hi3d\Hi3Dv1.0\LICENSE and code\hi3d\LICENSE.
	In short (not legal advice; the license is what counts):
	-  you may use, change and share Hi3D, also commercially (section 2);
	-  the 3D shapes you make with it - its "Outputs" - are yours as far as Hi3D is concerned:
	   its owner claims no ownership of them, and using, selling or sharing them needs no
	   notice and no "Powered by Hi3D 1.0" of you (section 9);
	-  a product built on Hi3D for commercial gain must show "Powered by Hi3D 1.0", linked to
	   http://hi3d.ai/ (section 7). Spellcast3D shows it in the 3D stage, by Shape, while
	   Hi3D is picked or made the model shown. Spellcast3D is not made, endorsed or supported
	   by OCEANO HK HOLDING LIMITED, and is no official Hi3D product (section 7.1); the name
	   Hi3D is used only to say which model it is (section 8).

	By using Hi3D through Spellcast3D you accept its Acceptable Use Policy (Schedule 1 of
	its license), which Spellcast3D passes on to you as a condition: do not use Hi3D, or
	make or spread content with it, to
	-  break the law, sanctions, export controls or court orders;
	-  infringe copyright, patents, trademarks, trade secrets, privacy, publicity or image
	   rights, or other rights;
	-  exploit, sexualize, groom or traffic minors, or otherwise harm them;
	-  make or spread malware or break into systems, steal credentials or abuse networks;
	-  deceptively impersonate anybody, commit fraud, or make deceptive synthetic media
	   without the disclosure the law asks for;
	-  facilitate violence, terrorism, self-harm, unlawful discrimination, stalking,
	   harassment or targeted abuse;
	-  unlawfully collect, infer, disclose or use personal or sensitive information;
	-  get around technical safeguards in Hi3D (good-faith security research aside);
	-  make fully automated decisions with legal or similar effects on people (healthcare,
	   jobs, credit, housing, education, insurance, policing, essential services) without
	   the human oversight, testing, notice and appeal the law asks for;
	-  pass off a changed Hi3D, or a product, as an official Hi3D version or as endorsed by
	   OCEANO HK HOLDING LIMITED.
	Lawful research, testing and red-teaming with reasonable safeguards are fine.

	For whoever builds a product or service on Spellcast3D: once your products that use
	Hi3D together exceed 200,000 monthly active users, or earn USD 1,000,000 or more in a
	month, you must tell OCEANO HK HOLDING LIMITED and apply for a separate written license
	within 60 days (section 5).

	Spellcast3D does NOT ship Hi3D's weights. Its installer downloads them, with Pixal3D,
	onto your PC from their publishers' HuggingFace pages (pinned versions, checked by
	SHA-256), into models\hi3d, unchanged:
	-  Hi3D v1.0 (huggingface.co/hi3d-official/Hi3Dv1.0): its networks, with its LICENSE,
	   NOTICE, AUTHORS and THIRD_PARTY_LICENSES.md beside them (models\hi3d\Hi3Dv1.0);
	-  DINOv2-large by Meta (huggingface.co/facebook/dinov2-large), Apache-2.0;
	-  CLIP ViT-L/14 by OpenAI (huggingface.co/openai/clip-vit-large-patch14), MIT;
	-  TRELLIS.2's shape encoder by Microsoft (huggingface.co/microsoft/TRELLIS.2-4B), MIT.
	Hi3D's code, in code\hi3d, is Spellcast3D's copy of github.com/hi3d-official/Hi3Dv1.0
	(github.com/IgorAherne/Hi3Dv1.0, branch spellcast3d),
	under the same license, with its notices kept. Spellcast3D changed some of its files to
	run it in Pixal3D's program on artists' PCs; each changed file says so at its top. Hi3D's
	own NOTICE (code\hi3d\NOTICE) names the parts it builds on: TRELLIS (MIT, Microsoft), Pointcept / Point Transformer V3 (MIT),
	O-CNN (MIT) and FlexiCubes (Apache-2.0, NVIDIA).
	It also needs spconv and cumm (Apache-2.0), pyfqmr (MIT) and their helpers pccm and
	ccimport (MIT), installed into Pixal3D's Python environment: permissive.

	Its Notice (section 8.2):

	"Hi3D is licensed under the Hi3D Open License Agreement Version 1.0. Copyright © 2026 OCEANO HK HOLDING LIMITED. All rights not expressly granted are reserved. The Hi3D trademarks are owned by OCEANO HK HOLDING LIMITED or its applicable affiliate(s)."


-------------------------------------------------------------------------------

4.  INSTANT MESHES  -  retopology + UVs   (code\instant-meshes, code\whl)

	BSD-style license. Copyright (c) 2015 Wenzel Jakob, Daniele Panozzo, Marco Tarini and
	Olga Sorkine-Hornung. See code\instant-meshes\LICENSE.txt

	Free to use, also commercially; keep the copyright notice when you share it.
	Note its extra clause: if you publish improvements to its source code without a separate
	license, you give the authors a free license to use them.

	The prebuilt wheel in code\whl also contains Eigen (MPL-2.0, https://eigen.tuxfamily.org)
	and pybind11 (BSD-3-Clause). The Python libraries it installs (numpy, trimesh, xatlas,
	FastAPI, ...) are permissive (BSD / MIT / Apache-2.0). xatlas (MIT, Jonathan Young; its
	Python binding MIT, Markus Worchel) comes as Spellcast3D's own build (code\whl), with one
	speed fix to its chart merging; its license file is inside the wheel.
	The Projection row's baker (Spellcast3D's own code, code\spellcast\launchers\spellbake) runs
	in the same environment and adds embreex (BSD-2-Clause), which bundles Intel Embree and
	oneTBB (Apache-2.0).
	Instant Meshes also makes the UVs of the clean meshes LATO.2 and TriFlow make.


-------------------------------------------------------------------------------

5.  LATO.2  -  AI retopology   (code\lato2, models\lato2)

	LATO.2's code and model weights: MIT License, Copyright (c) 2026 LATO.2.
	See code\lato2\LICENSE. The weights come from huggingface.co/0x4c48/LATO.2 (MIT).

	Free to use, also commercially; keep the copyright notice when you share it.

	It also uses:
	-  DINOv2 by Meta (it looks at a picture of your model): code and weights Apache-2.0.
	       After install: models\lato2\dinov2\facebookresearch_dinov2_main\LICENSE
	-  o_voxel (from Microsoft's TRELLIS.2, MIT) and flash-attention (BSD-3-Clause): the same
	   prebuilt wheels as Pixal3D's, from code\pixal3d\whl.
	-  spconv and cumm (Apache-2.0), PyTorch, numpy, scipy, trimesh, FastAPI and the other
	   libraries its environment installs: permissive (BSD / MIT / Apache-2.0).

	LATO.2 does NOT use nvdiffrast (NVIDIA, non-commercial): Spellcast3D draws the picture
	LATO.2 looks at with plain PyTorch instead. Nothing non-commercial is involved.


-------------------------------------------------------------------------------

6.  TRIFLOW  -  AI retopology, NON-COMMERCIAL ONLY   (code\triflow, models\triflow)

	Automotive Development Public Non-Commercial License 1.0 (ADPNCL 1.0).
	Copyright (c) 2026 Haoxuan Li; license steward Audi AG (Copyright (c) 2024 Audi AG,
	ai-licenses@audi.de). See code\triflow\LICENSE
	Its model weights (huggingface.co/lihcxr/TriFlow) are under the same license.

	-> NON-COMMERCIAL only: "not primarily intended for or directed towards commercial
	   advantage or monetary compensation". Use it for learning, personal projects,
	   prototypes and evaluation. For work you sell, or are paid for, pick Instant Meshes or
	   LATO.2 in the Retopo stage instead.

	TriFlow's source is shipped under its own license, with its LICENSE file, in code\triflow,
	in Spellcast3D's modified version (github.com/IgorAherne/triflow). Its
	weights are downloaded on your PC by the installer; Spellcast3D does not ship them.

	Its license (section 3.4) says every change must come with a file that documents the
	differences and the date of each change, and a clear statement that it is derived from
	TriFlow, naming its author; and that this documentation, together with the changed code,
	must be provided to the Contributor (the license's notice names Audi AG,
	ai-licenses@audi.de) - even if the changed code is never shared with anybody else.

	Spellcast3D's changes (less graphics memory and time, small holes closed) are listed,
	each with its date, in code\triflow\CHANGES.md; every changed file says so at its top and
	keeps its original notice. Spellcast3D provides that documentation and its code to the
	Contributor, as section 3.4 asks. Nothing else is added inside TriFlow's folder except
	links to its weights in code\triflow\checkpoints (the folder TriFlow keeps for them; the
	installer checks that every other file is byte for byte Spellcast3D's version), and it
	runs in a program of its own, separate from Spellcast3D.

	If you change TriFlow's code yourself, the same applies to your changes: document them
	with their dates, and provide that documentation with your changed code to the
	Contributor.

	TriFlow also uses:

	-  MeshLib by AMV Consulting, LLC (MeshInspector)  (measures and projects the mesh)
	       "Non-commercial & education" trial license - not open source. It allows only
	       non-commercial, evaluation or educational use, can be ended by its owner at any
	       time, and forbids passing MeshLib on to anybody. So Spellcast3D never ships it:
	       the installer gets it from PyPI straight onto your PC when it installs TriFlow.
	       After install: code\venvs\triflow\Lib\site-packages\meshlib-*.dist-info\licenses\LICENSE
	       More: https://github.com/MeshInspector/MeshLib

	-  Parts inside code\triflow\third_party (each with its own license file there):
	       TRELLIS (MIT, Microsoft) with FlexiCubes (Apache-2.0, NVIDIA), Direct3D-S2 (MIT,
	       DreamTech), pyfqmr-triflow (MIT, Kramer84).
	       One TRELLIS file TriFlow loads carries an Inria notice instead: "free for
	       non-commercial, research and evaluation use"
	       (code\triflow\third_party\TRELLIS\trellis\representations\gaussian\general_utils.py).

	-  Two libraries nobody publishes for Windows, which Spellcast3D builds from their
	   published source (code\whl; how: code\release\wheels and the .json files next to them):
	       torchsparse (MIT, MIT HAN Lab), with the sparsehash headers inside (BSD-3-Clause,
	       Google), and pyfqmr-triflow (MIT, Kramer84). Their license files are inside the
	       wheels.

	-  The Python libraries its environment installs (spconv, open3d, numba, hydra, trimesh,
	   transformers, ...) are mostly permissive (BSD / MIT / Apache-2.0); plyfile (GPL-3.0)
	   and easydict (LGPL-3.0) are among them, installed unchanged on your PC.
	   NVIDIA's kaolin (partly non-commercial) is NOT installed: a small stand-in of
	   Spellcast3D's own takes its place (section 7).


-------------------------------------------------------------------------------

7.  THE FILES THAT RUN TRIFLOW  -  MIT, not AGPL

	    code\spellcast\launchers\triflow_server.py
	    code\spellcast\launchers\triflow_stubs\          (the kaolin stand-in)

	These load TriFlow's code in TriFlow's own program, never inside Spellcast3D, so they are
	under the MIT license below instead of the AGPL. The three small helpers
	triflow_server.py also loads into that program, attn_shim.py, spconv_shim.py and
	_loopback_guard.py (in the same folder), may be used under these MIT terms as well.

	MIT License

	Copyright (c) 2026 Igor Aherne

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all
	copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
	SOFTWARE.


-------------------------------------------------------------------------------

8.  PROGRAMS BUNDLED IN THE 'tools' FOLDER

	-  Python 3.11.9  (tools\python)
	       Python Software Foundation License - permissive.  See tools\python\LICENSE.txt

	-  Git for Windows, MinGit 2.55.0  (tools\git)
	       GNU General Public License v2, plus the licenses of its parts.
	       See tools\git\LICENSE.txt.  Source: https://github.com/git-for-windows/git

	-  Microsoft Visual C++ runtime  (tools\python\msvcp140*.dll, vcruntime140*.dll,
	   concrt140.dll, vcomp140.dll)
	       Copyright (c) Microsoft Corporation. Redistributable runtime files, shipped under
	       Microsoft's license terms for redistributable Visual C++ code, so you don't have to
	       install the "Visual C++ Redistributable" yourself.


-------------------------------------------------------------------------------

YOUR WORK

	Spellcast3D claims nothing on the images and models you make. What you may do with them
	depends on the licenses above: for pictures, on the image generator (Qwen-Image 2.1's are
	yours, also for commercial work, as Qwen said; with Forge Neo, it depends on the image
	model you chose); for 3D models, on the non-commercial parts of Pixal3D (Hi3D claims
	nothing on the shapes it makes); and on the retopology method: a clean mesh TriFlow made
	is for non-commercial use only.


Any questions, or info about alternatives? Write to Igor on Discord:

https://discord.gg/aWbnX2qan2
https://stableprojectorz.com

October 2026

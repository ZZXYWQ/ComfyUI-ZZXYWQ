# ZZX Nodes — Paints-Undo 与视频工具

把成图逐步撤销为少上色、线稿、少笔画的关键帧序列。使用原版 Paints-Undo 模型推理，不是褪色或擦除滤镜。

![Actual inference example](workflows/PaintsUndo-sequence-example.jpg)

## 安装

在 ComfyUI 的 `custom_nodes` 目录克隆本仓库，然后使用 **运行 ComfyUI 的 Python** 安装根目录依赖并重启：

```sh
git clone https://github.com/ZZXYWQ/ComfyUI-ZZXYWQ.git
python -m pip install -r ComfyUI-ZZXYWQ/requirements.txt
```

Windows portable 在其根目录执行：

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-ZZXYWQ\requirements.txt
```

不需要安装 `Paints-UNDO/requirements.txt`；该文件只供保留的上游 Gradio/视频实验代码使用。节点主流程不依赖 Gradio、xformers、bitsandbytes、PEFT 或视频模型。

## 多套 ComfyUI / Studio 安装注意

GitHub 更新只更新代码，不会把模型和依赖同步到其他 ComfyUI 安装。安装依赖、节点和模型时必须使用正在运行的那一套目录与 Python。

例如 Studio 位于 `D:\ComfyUI_Studio\ComfyUI_windows_portable` 时，模型应在该目录的 `ComfyUI/models/paints_undo`，不能假定旧的 `D:\ComfyUI_windows_portable` 与它共用模型。可用 `PAINTS_UNDO_MODEL_DIR` 显式指定已有完整模型缓存。

遇到 `no file named model.safetensors` 等错误时，检查 `single_frame/text_encoder/model.safetensors`、`single_frame/unet/diffusion_pytorch_model.safetensors` 和 `single_frame/vae/diffusion_pytorch_model.safetensors` 是否下载完整；仅有 config.json 或临时下载文件不够。`single_frame` 目录存在时节点会使用这套本地模型。不要把缺权重误当成节点已完成安装。

Studio 实际运行环境及回归结果见 [TESTING.md](TESTING.md#installed-comfyui-studio-verification)。

## 使用

导入 `workflows/workflow-PaintsUndoSequence.json`，在 Load Image 选择自己的图片，然后运行：

`Load Image → ZZX_PaintsUndoSequence → Save Image / Preview Image`

- `prompt`：留空自动调用原版 WD14 MOAT tagger，推荐这样使用；也可输入已有标签。
- `operations`：严格递增的撤销位置，默认 `400,600,800,900,950,999`。0 是原图，999 接近最早的笔画。它不是采样步数。
- `sampling_steps`：默认 50；示例使用 30。
- `cfg`：默认 3.0，与上游关键帧配置一致。
- `resolution`：最长边，默认 640；宽高按比例取整为 64 的倍数，不裁切画面。
- `include_original` / `include_blank`：添加首张原图和末张白纸。白纸是合成终帧，不是模型推理结果。
- `seed`：各个撤销位置共享初始噪声，以提高帧间一致性。

输出是标准 ComfyUI `IMAGE` 批次 `[帧数, 高, 宽, 3]`，float32、RGB、0–1。也可连接你已有的视频保存节点制作逐帧动画。这个精简节点输出关键帧，并未加载上游耗显存的多帧视频插值模型。

旧 `ZZX_PaintsUndo` 节点名、输入和输出接口保留，现在正确返回单张图片。一次只接收一张 RGB 输入；多图批次会明确报错，避免静默丢失图片。

模型模拟绘画过程，并不能恢复原画真实的图层历史。上色和笔画变化的位置取决于图片与模型；它不保证每一步严格只减不增，某些细节会重绘。可调整 operations 和 seed。

## 接入其他项目 / HTTP API

`workflows/paints_undo_api.json` 是可提交到 ComfyUI `/prompt` 的 API 格式；`workflow-PaintsUndoSequence.json` 是拖入界面的可视工作流，两者用途不同。

1. 将图片通过 ComfyUI `POST /upload/image` 上传，使用返回的文件名替换 API JSON 节点 `1.inputs.image`（如有子目录，带上子目录）。
2. 修改节点 `2.inputs` 中的 operations、seed 等参数。
3. 将 JSON 包装为 `{"prompt": <API JSON>}`，POST 到 `/prompt`。
4. 用返回的 `prompt_id` 查询 `/history/{prompt_id}`；完成后从节点 `3` 的 outputs.images 读取 filename/subfolder/type，并通过 `/view` 获取图片。

输出图片按撤销顺序排列。默认 6 个撤销位置加原图、白纸共 8 帧；旧节点只有 1 帧。API 示例的 `paints_undo_example.jpg` 是占位文件名，需先上传图片或复制 `Paints-UNDO/imgs/1.jpg` 到 ComfyUI/input 并改成该名字。

## 模型与资源

首次运行自动下载 `lllyasviel/paints_undo_single_frame` 和 WD14 MOAT tagger，总计数 GB，需要网络。缓存默认在 `ComfyUI/models/paints_undo`，不修改全局 HF_HOME。

可设置环境变量 `PAINTS_UNDO_MODEL_DIR` 指定缓存目录；也可将完整单帧模型（tokenizer、text_encoder、vae、unet 子目录）放到该目录下的 `single_frame/`。已缓存后可以离线运行。

模型按 VAE → CLIP → UNet → VAE 分阶段移入显卡，每次只生成一帧，完成或失败后卸载。VAE 使用 float32 避免半精度溢出导致黑图。支持 CUDA；无 CUDA 时回退 CPU（速度较慢）。其他加速后端尚未验证。

## 验证与开发

```sh
python -m unittest discover -s tests -v
python scripts/render_example.py
```

第二条命令运行真实模型，生成 PNG 帧、GIF 和对比图到 `outputs/`。第一次运行会下载模型。详见 [测试记录](TESTING.md)。

## 上游

模型及核心采样/条件模块来自 [lllyasviel/Paints-UNDO](https://github.com/lllyasviel/Paints-UNDO)，上游许可证保留于 `Paints-UNDO/LICENSE`。原版 Gradio 和视频源码保留以便参考，但不会在节点加载时导入。

## 视频转码与流录制（v1.2）

这两个节点只需要 Python 标准库与系统 **FFmpeg + ffprobe**，无需 Tkinter、OpenCV 或显卡。把工具加入 PATH，或设置 `ZZX_FFMPEG` / `ZZX_FFPROBE` 为可执行文件完整路径。重启 ComfyUI 后生效。

### VideoFormatConverter

导入 `workflows/workflow-VideoFormatConverter.json`，填写服务器上的 `video_path` 后执行。输出完整文件路径和兼容 VideoHelperSuite 的 `VHS_VIDEOINFO`；仅运行转码不需要安装 VideoHelperSuite。

- `output_path` 是目录；留空使用 ComfyUI/output。自动创建目录并使用递增文件名，避免覆盖已有文件。
- `output_filename` 只填写文件名，不包含路径；留空使用输入视频名称。
- `fill` 拉伸到目标宽高；`crop` 保持比例放大后居中裁剪。输出像素宽高比为 1:1。
- `audio_codec=copy` 保留音轨，忽略音频码率、声道和采样率；原音轨必须与目标容器兼容。其他模式下 `original` 保持原声道数；无音轨视频也可转码。
- `output_enabled=false` 返回空路径与空信息字典，不执行转码。
- `opencl_acceleration` 为兼容旧工作流保留，已不再启用无效的 OpenCL 解码路径。需要 GPU 编码时选择 NVENC。

### StreamRecorder

导入 `workflows/workflow-StreamRecorder.json`，填写可访问的 HTTP(S)/HLS、RTMP、RTSP 等输入地址；也支持本地视频路径。默认录制一分钟，使用 CPU H.264。输出录制文件完整路径，可连接转码节点。

- `use_local_time=true` 时按 **ComfyUI 服务器本地时间**等待 `HH/MM/SS` 或 `HH:MM:SS`；当天已过去的时间会报错。等待期间占用当前执行任务，可取消。
- 支持 MP4、MOV、MKV、AVI、FLV、WebM、HLS 与 DASH；WebM 使用 AV1 + Opus，其他录制格式转为 AAC 音频。
- HLS 输出 `.m3u8` 与独立编号的 `.ts` 分段；DASH 输出 `.mpd` 与 `.m4s` 分段。移动时需保留整套文件。
- 旧值 `m3u8/http/https` 映射为 HLS，`rtmp` 映射为 FLV，`srt/rtsp` 保存为 MPEG-TS（`.ts`）。这些选项只决定本地文件容器，不向远端推流。原来无法正确工作的 `mss` 已移除，旧工作流请改用 MP4 或 DASH。
- `use_cuda` 控制硬件解码；NVENC 编码单独由 `codec` 控制。需要支持的 NVIDIA 显卡、驱动与 FFmpeg 构建，失败时会明确报错，不会悄悄改用其他编码器。

两个节点的 `video_quality` 统一为 CRF/CQ，**数字越小画质越高、文件通常越大**，建议从 23 开始。旧工作流参数会保留，但相同数字不再沿用原来方向不一致的码率公式。

两者均为输出节点，每次排队执行都会写出新文件，支持 ComfyUI 的取消操作。录制网络读取设有 15 秒超时，整体设有“录制时长 + 60 秒”的上限；极慢的软件编码可能触发这个上限。失败或取消时删除本次未完成的输出。帧数在容器未提供时按时长与帧率估算，可变帧率素材的估算值可能不同于精确解码计数。

FFmpeg 参数及容器规则参考[官方命令文档](https://ffmpeg.org/ffmpeg.html)和[格式文档](https://ffmpeg.org/ffmpeg-formats.html)。

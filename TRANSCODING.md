# 高级转码（v1.3）

面向 ComfyUI 的多功能 FFmpeg 转码工具，覆盖类似 ShanaEncoder 的常用操作；并非 ShanaEncoder 的复制品或完整替代。旧 `VideoFormatConverter`、`StreamRecorder` 和 Paints-Undo 节点保留，旧工作流不需要迁移。

## 开始使用

安装 FFmpeg 与 ffprobe，并将其加入 PATH。也可设置 `ZZX_FFMPEG`、`ZZX_FFPROBE`。更新插件后重启 ComfyUI，在 **ZZX / Video Advanced** 找到新节点。

最简单：添加 **Advanced Converter / 高级转码**，填写服务器上的视频路径、选择预设、运行。输出目录留空使用 ComfyUI/output，文件名自动编号，保留已有文件。

完整设置：导入 `workflows/workflow-AdvancedVideoConverter.json`：

`音频字幕设置 → 视频设置 → 高级转码`

批量：导入 `workflows/workflow-BatchVideoConverter.json`。只需要常用预设时，可以删除两个设置节点，直接使用转码节点内的预设。

运行结束后转码节点显示文件路径、编码器和尺寸等结果。可将输出路径接入 VideoHelperSuite 的 Load Video (Path)，将 `video_info` 接入 VHS VideoInfo。音频提取的 `video_info` 是空字典，不适合连接视频信息节点。

## 新节点

| 节点 | 功能 |
| --- | --- |
| Advanced Converter | 单文件转码、无损重新封装、音频提取、GIF |
| Batch Converter | 文件夹或逐行文件列表，递归扫描，保留子目录，错误报告 |
| Video Settings | 容器、编码、画质、画面处理、截取与变速 |
| Audio & Subtitles | 音轨、编码、声道、音量、响度、字幕 |
| Preset File | 保存／加载 JSON 预设，保存时自动编号，不覆盖 |
| Media Info | 显示所有流、语言、编码和文件信息；检测本机编码器 |

## 格式与编码

输出容器：MP4、MKV、MOV、AVI、WebM、TS、MPEG、FLV、OGV、GIF、HLS、DASH，以及 MP3、M4A、AAC、WAV、FLAC、Opus、OGG 音频。

视频编码：H.264、HEVC、AV1（SVT/AOM）、VP8、VP9、ProRes、FFV1、MPEG-4、MPEG-2、WMV2、Theora、GIF。包含 NVIDIA NVENC、Intel QSV、AMD AMF 的 H.264／HEVC／AV1 选项；需要对应的硬件、驱动与 FFmpeg 支持。

音频编码：AAC、MP3、Opus、Vorbis、FLAC、ALAC、16/24-bit PCM、AC3、EAC3；也可复制音轨或关闭音频。

容器与编码不是任意组合。例如 WebM 要求 VP8/VP9/AV1＋Opus/Vorbis；ProRes 使用 MOV/MKV；FFV1 使用 MKV/AVI。节点提前检查常见冲突，其他不支持的组合由 FFmpeg 报错。失败文件会清理。

`auto_h264/auto_hevc/auto_av1` 依次实测 NVENC、QSV、AMF，再回退软件编码。显式指定硬件编码器时，失败不会擅自换编码。**Media Info → test_hardware** 会实际编码一帧，区分“编译进 FFmpeg”和“本机可用”；检测结果可通过 refresh 刷新。一次 8-bit 检测不保证全部分辨率、profile 和 10-bit 模式可用。

## 视频设置

- `quality` 模式使用相应编码器的 CRF/CQ/QP；通常数字越小质量越高，但不同编码器数字不可直接比较。ProRes 用 profile 控制档次，FFV1 使用无损编码，GIF 使用调色板。
- `vbr` 使用目标码率，`cbr` 设置码率与缓冲约束；最终码率仍受编码器及素材影响，不代表每个时间片严格恒定。
- `two_pass` 对相同素材执行两遍，支持 libx264、libvpx-vp9、mpeg4、mpeg2video。码率单位 kbps。
- `encoding_speed` 为 fast / balanced / slow，映射到各编码器的速度预设。
- `pixel_format=auto` 通常输出 8-bit yuv420p，ProRes 使用 yuv422p10le，FFV1 保留编码器支持的输入像素格式。可选择 10-bit；转换为 10-bit 不会凭空恢复原始细节，也不会自动执行 HDR → SDR 色调映射。
- `profile` 留空为默认，可填写对应 FFmpeg 编码器支持的 profile，例如 H.264 的 `high`，HEVC 的 `main10`，ProRes 的 `3`。不支持的 profile 会报错。
- 宽高为 0 保持源尺寸；只填写一边，另一边按比例计算。`fit` 保持比例缩入范围，`pad` 补黑边到目标尺寸，`crop` 放大后居中裁剪，`stretch` 拉伸。编码需要时补齐偶数尺寸。
- 可设置四边裁剪、旋转、翻转、去隔行、降噪、锐化、亮度／对比度／饱和度。非方形像素素材在调整尺寸后输出 SAR=1；特殊变形宽银幕素材建议检查预览。
- `frame_rate=0` 不强制帧率；其他值通过补帧／丢帧转为恒定帧率，**不是 AI 运动插帧**。
- `start_seconds` 和 `duration_seconds` 按源素材时间选取；时长为 0 到结尾。`playback_speed` 支持 0.25–4，音频同步变速。直接复制编码的截取受关键帧限制，需要逐帧精确截取时重新编码。
- `video_track` 为视频流内部从 0 开始的编号，忽略封面图像。`metadata=strip` 去除源标签和章节；编码器自身必要标签可能仍存在。截取／变速时不复制原章节，避免错误时间。

`copy` 只重新封装，不重新编码，不能与缩放、烧录、变速等画面处理同时使用。FFV1 预设在不进行过滤及像素格式转换时保留解码图像；从有损素材转换为无损不会恢复已经损失的信息。

## 音频与字幕

`audio_tracks` 支持 `first`、`all`、`0,2` 等从 0 开始的音轨编号。Media Info 可查看轨道顺序和语言。音频文件提取一次选择一条音轨；无音轨输入可转换视频，但不能提取音频。

可调整码率、采样率（0 自动）、声道（原始／单声道／立体声／5.1）、音量 dB，或启用单遍响度标准化（-16 LUFS、-1.5 dBTP）。增加声道数不等于生成真实空间声。音频 copy 不支持修改声道、采样率、音量或速度。

字幕模式：

- `none`：不输出字幕。
- `copy`：复制全部内嵌字幕流，要求目标容器支持原字幕编码。
- `embed`：选择内嵌字幕，或外部 SRT/ASS/SSA/VTT，转换为 MP4/MOV 的 mov_text 或 MKV 的 SRT。转换会丢失复杂 ASS 样式。
- `burn`：将所选字幕绘制到画面，保留 libass 支持的样式，需要 FFmpeg 带 subtitles/libass 滤镜。

`subtitle_track` 为字幕流内部编号。`subtitle_file` 留空选取内嵌轨道；填写则使用外部字幕。`subtitle_language` 为封装字幕语言标签。内嵌位图字幕目前不支持转换为文本／烧录，需先做 OCR 或使用文本字幕。变速时选择 burn 或 none，以免封装字幕时间失配。缺少字体或自定义 ASS 附件字体可能影响显示。

## 批量与预设

批量节点优先使用 `file_list`（每行一个绝对文件路径）；留空后扫描 `input_directory`。扩展名用逗号分隔。目录扫描会跳过指定的独立输出子目录；在同一目录重复运行前应检查输入范围，避免再次转码上一轮输出。开始前确定文件列表，执行过程中不会把新文件加入队列。

`max_files` 超过限制时报错，不会静默截断。`continue_on_error=true` 保留成功文件并逐项报告失败；关闭则在第一个错误处停止。取消会终止正在运行的 FFmpeg，删除当前未完成文件；已经完成的文件保留。为了控制 CPU/GPU 占用，批量逐文件执行。

覆盖顺序为：**内置预设 → 连接的 settings → preset_json**。Video Settings 输出完整设置，连接后会覆盖内置预设；建议选择 Custom。JSON 可只包含需要覆盖的字段。

保存预设：将 Video Settings 的 settings 接到 Preset File，mode 选 save，preset_file 填服务器上 `.json` 路径；返回实际保存路径。mode=load 可加载该文件并连接到转码节点。工作流本身也会保存设置。参考 `presets/` 中的示例。

`extra_options_json` 允许少量高级编码参数，例如 `{"-g":48,"-bf":2,"-tune":"film"}`。采用固定白名单，不支持添加输入、输出、任意滤镜或命令。具体允许项见节点报错提示和源码。

## API 与验证

`workflows/advanced_video_api.json` 是可提交到 `/prompt` 的最小例子；替换服务器文件路径。其他 workflow JSON 是拖入前端的可视工作流。

```sh
python -m unittest discover -s tests -p test_video.py -v
python -m unittest discover -s tests -p test_transcode.py -v
```

测试记录见 [TESTING.md](TESTING.md)。参数语义参考 [FFmpeg 编码器文档](https://ffmpeg.org/ffmpeg-codecs.html)、[滤镜文档](https://ffmpeg.org/ffmpeg-filters.html)和[容器文档](https://ffmpeg.org/ffmpeg-formats.html)。

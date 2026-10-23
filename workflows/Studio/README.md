# Studio 工作流库

把本目录中的 `03_Video` 合并复制到 ComfyUI 的
`user/default/workflows/Studio/`，即可从工作流侧栏按功能打开。
首次安装或更新节点后重启 ComfyUI；添加工作流后刷新工作流列表或页面。

| 分类 | 工作流 | 用途 |
| --- | --- | --- |
| `03_Video/ZZX_Transcode` | `01_Advanced_Converter.json` | 视频设置＋音频字幕设置＋高级转码 |
| `03_Video/ZZX_Transcode` | `02_Batch_Converter.json` | 文件夹／文件列表批量处理 |
| `03_Video/ZZX_Transcode` | `03_Basic_Converter.json` | 原版接口的基础转码 |
| `03_Video/ZZX_Recording` | `01_Stream_Recorder.json` | 流媒体录制 |

这些文件与仓库 `workflows/` 根目录的对应示例内容一致。保留原位置，避免旧链接失效。
它们不包含个人素材路径或测试输出；运行前填写输入视频路径、批量输入目录或流地址。
输出目录留空使用 ComfyUI/output。高级选项及预设说明见[转码指南](../../TRANSCODING.md)。

原有 Paints-Undo 工作流继续保留在 Studio 的图片编辑／视频分类中，不重复安装或覆盖。

2.StreamRecorder

A streaming media receives a local recording node:
Streaming formats available: rtmp, .m3u8.
Optional formats: mp4, mov, mkv, avi.
Optional encoding: av1, h264, h265.
Optional: whether to use local time,If yes, then this start_time takes effect, the time inside is hours/minutes/seconds.
Optional: recording time (slider).
Optional: whether to use cuda.


一个流媒体收到录本地的节点：
可输入流媒体格式：rtmp，.m3u8。
可选格式：mp4，mov，mkv，avi。
可选编码：av1，h264，h265。
可选:是否使用本地时间，
     如果是，则这start_time生效，里面的时间是时/分/秒。
可选:录制时间（滑块）。
可选:是否使用cuda。

AVI (avi): h264

MOV (mov): h264, hevc

MKV (mkv): av1, h264, hevc

MP4 (mp4): av1, h264, hevc

HLS (hls): av1, h264, hevc

DASH (dash): av1, h264, hevc

MSS (mss): av1, h264, hevc

SRT (srt): h264, hevc

FLV (flv): h264

WebM (webm): av1, h264

RTMP (rtmp): h264

RTSP (rtsp): h264, hevc

M3U8 (m3u8): av1, h264, hevc
======
1.VideoFormatConverter：

A video transcoding node:
Optional formats: mp4, mov, mkv, avi.
Optional encoding: av1, h264, h265.
Optional frame rate: 8, 15, 24, 25, 30, 50, 59, 60, 120.
Optional width and height: (not listed one by one).
Optional audio: mp3, aac.
Optional frequency: 44100, 48000.

一个视频转码的节点：
可选格式：mp4，mov，mkv，avi。
可选编码：av1，h264，h265。
可选帧率：8，15，24，25，30，50，59，60，120。
可选宽高：（不一一列举）。
可选音频：mp3，aac。
可选频率：44100，48000。


0.
Options can be added to the above nodes,
Please give feedback if you have any questions

以上节点均可添加选项，
有问题请反馈

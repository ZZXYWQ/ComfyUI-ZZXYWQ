import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const reports = new Set(["ZZX_VideoConvertAdvanced", "ZZX_VideoBatchConvert", "ZZX_TranscodePreset", "ZZX_MediaInfo"]);
app.registerExtension({
    name: "ZZX.TranscodeReports",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!reports.has(nodeData.name)) return;
        const previous = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            previous?.apply(this, arguments);
            if (!message.text) return;
            if (!this.zzxReportWidget) {
                this.zzxReportWidget = ComfyWidgets.STRING(this, "result", ["STRING", { multiline: true }], app).widget;
                this.zzxReportWidget.options.serialize = false;
                if (this.zzxReportWidget.inputEl) this.zzxReportWidget.inputEl.readOnly = true;
            }
            this.zzxReportWidget.value = message.text.join("\n");
            const size = this.computeSize();
            size[0] = Math.max(size[0], this.size[0]);
            this.setSize(size);
            this.setDirtyCanvas(true, true);
        };
    },
});

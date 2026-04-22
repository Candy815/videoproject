"""
Gradio Web 界面 - 视频抄袭检测系统
启动命令: python app.py
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from pipeline.detection_pipeline import DetectionPipeline
from config.settings import SIMILARITY_THRESHOLD, CLIP_SIMILARITY_THRESHOLD


class VideoDetectionApp:
    """视频抄袭检测 Gradio 应用"""

    def __init__(self):
        self.pipeline = None
        self._init_pipeline()
        # 初始化动态帧提取器
        self.frame_extractor = None
        self._init_frame_extractor()

    def _init_pipeline(self):
        """初始化检测流水线"""
        try:
            self.pipeline = DetectionPipeline()
            print("✓ 检测流水线初始化成功")
        except Exception as e:
            print(f"✗ 检测流水线初始化失败: {e}")

    def _init_frame_extractor(self):
        """初始化动态帧提取器"""
        try:
            from core.frame_extractor_dynamic import DynamicFrameExtractor
            self.frame_extractor = DynamicFrameExtractor()
            print("✓ 动态帧提取器初始化成功")
        except Exception as e:
            print(f"✗ 动态帧提取器初始化失败: {e}")

    def detect_video(self, video_file, video_url, keywords, similarity_threshold, clip_threshold):
        """
        检测视频（支持本地文件和URL）- 保持原有接口兼容
        """
        result = self.detect_video_with_frames(video_file, video_url, keywords, similarity_threshold, clip_threshold)
        # 返回前三个值，保持与原接口兼容
        return result[0], result[1], result[2]

    def detect_video_with_frames(self, video_file, video_url, keywords, similarity_threshold, clip_threshold):
        """
        检测视频并返回帧对比数据（新接口）
        返回值: (summary, compare_video, report_file, gallery_data)
        """
        # 参数验证
        if video_file is None and not video_url:
            return "❌ 请上传视频文件或输入视频URL", None, None, []

        # 临时修改阈值
        import config.settings
        original_sim_threshold = config.settings.SIMILARITY_THRESHOLD
        original_clip_threshold = config.settings.CLIP_SIMILARITY_THRESHOLD

        try:
            config.settings.SIMILARITY_THRESHOLD = similarity_threshold
            config.settings.CLIP_SIMILARITY_THRESHOLD = clip_threshold

            # 解析关键词
            user_keywords = None
            if keywords and keywords.strip():
                user_keywords = [kw.strip() for kw in keywords.split(',')]

            # 执行检测
            if video_file is not None:
                # 本地文件检测
                report = self.pipeline.detect_infringement(
                    video_file,
                    user_keywords=user_keywords,
                    similarity_threshold=similarity_threshold
                )
            else:
                # URL 检测
                if hasattr(self.pipeline, 'detect_from_url'):
                    report = self.pipeline.detect_from_url(
                        video_url,
                        user_keywords=user_keywords,
                        top_k=10
                    )
                else:
                    return "❌ URL检测功能未启用", None, None, []

            # 生成结果摘要
            summary = self._generate_summary(report)

            # 获取对比视频路径
            compare_video = report.compare_video_path if report.compare_video_path else None

            # 获取报告路径
            report_file = report.report_file_path if report.report_file_path else None

            # ========== 新增：构建帧对比 Gallery 数据 ==========
            gallery_data = self._build_gallery_data(report, video_file if video_file else video_url)

            return summary, compare_video, report_file, gallery_data

        except Exception as e:
            return f"❌ 检测失败: {str(e)}", None, None, []
        finally:
            # 恢复原始阈值
            config.settings.SIMILARITY_THRESHOLD = original_sim_threshold
            config.settings.CLIP_SIMILARITY_THRESHOLD = original_clip_threshold

    def _build_gallery_data(self, report, query_video_path: str) -> List:
        """
        构建帧对比 Gallery 数据
        返回格式: List[Tuple[str, str]] 或 List[Dict]
        """
        from config.settings import MAX_FRAME_PAIRS_PER_VIDEO

        gallery_data = []

        if not report or not report.matches:
            return []

        # 按匹配视频ID分组
        matches_by_video = {}
        for match in report.matches:
            vid = match.matched_video_id
            if vid not in matches_by_video:
                matches_by_video[vid] = []
            matches_by_video[vid].append(match)

        for video_id, video_matches in matches_by_video.items():
            video_matches.sort(key=lambda x: x.similarity, reverse=True)
            top_matches = video_matches[:MAX_FRAME_PAIRS_PER_VIDEO]

            for match in top_matches:
                frame_path = self._extract_frame(query_video_path, match.query_timestamp)

                if frame_path:
                    # 构建标签
                    label = f"视频: {video_id[:25]}... | 时间: {match.query_timestamp:.1f}s | 相似度: {match.similarity:.2%}"
                    if match.matched_keywords:
                        label += f" | 关键词: {', '.join(match.matched_keywords[:2])}"

                    # Gallery 组件期望 (image, label) 元组格式
                    gallery_data.append((frame_path, label))

        return gallery_data
    def _extract_frame(self, video_path: str, timestamp: float) -> str:
        """
        从视频中提取指定时间点的帧（动态提取，不永久存储）
        返回临时帧图片路径
        """
        if self.frame_extractor is None:
            return None

        try:
            frame_path = self.frame_extractor.extract_frame_at_timestamp(video_path, timestamp)
            return frame_path
        except Exception as e:
            print(f"提取帧失败: {e}")
            return None

    def _generate_summary(self, report):
        """生成检测结果摘要（Markdown格式）"""
        if not report or report.total_matches == 0:
            return """
### 📊 检测结果

| 项目 | 结果 |
|------|------|
| 侵权等级 | ✅ **未检测到侵权** |
| 匹配片段数 | 0 |
| 最高相似度 | - |

> 未发现明显抄袭内容，视频原创性较高。
"""

        # 侵权等级对应的颜色和图标
        level_icons = {
            'high': '🔴 **高度侵权**',
            'medium': '🟠 **中度侵权**',
            'low': '🟡 **低度侵权**',
            'none': '🟢 **未侵权**'
        }

        semantic_info = ""
        if report.clip_summary and report.clip_summary.get('total_semantic_matches', 0) > 0:
            keywords = report.clip_summary.get('detected_keywords', [])
            semantic_info = f"\n| 语义匹配关键词 | {', '.join(keywords[:5])} |"

        # 构建匹配详情表格
        matches_table = ""
        for i, m in enumerate(report.matches[:10]):
            keywords_str = ', '.join(m.matched_keywords[:2]) if m.matched_keywords else '-'
            matches_table += f"| {i + 1} | {m.matched_video_id[:20]}... | {m.similarity:.2%} | {m.query_timestamp:.1f}s | {keywords_str} |\n"

        return f"""
### 📊 检测结果

| 项目 | 结果 |
|------|------|
| 侵权等级 | {level_icons.get(report.infringement_level, report.infringement_level)} |
| 匹配片段数 | {report.total_matches} |
| 最高相似度 | {report.max_similarity:.2%} |
| 平均相似度 | {report.avg_similarity:.2%}{semantic_info} |

---

### 📋 侵权片段详情

| 序号 | 匹配视频ID | 相似度 | 时间点(秒) | 语义关键词 |
|------|-----------|--------|-----------|-----------|
{matches_table}

---

### 📈 检测建议

{self._get_suggestion(report.infringement_level)}
"""

    def _get_suggestion(self, level):
        """根据侵权等级返回建议"""
        suggestions = {
            'high': "⚠️ 建议立即下架侵权内容，保留证据并联系法务处理。",
            'medium': "📝 建议进一步人工审核，确认是否构成侵权。",
            'low': "💡 可能存在巧合相似，建议人工复核。",
            'none': "✅ 视频内容原创性良好，可正常发布。"
        }
        return suggestions.get(level, "请人工复核检测结果。")

    def clear_outputs(self):
        """清除输出"""
        return None, None, None, []


# ========== 创建 Gradio 界面 ==========
def create_interface():
    """创建 Gradio 界面"""
    app = VideoDetectionApp()

    with gr.Blocks(title="视频抄袭检测系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 视频抄袭检测系统
        ### 基于多模态指纹检索与CLIP语义验证的智能侵权检测平台
        """)

        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📤 输入源")

                with gr.Tabs():
                    with gr.TabItem("📁 上传文件"):
                        video_input = gr.File(
                            label="上传视频文件",
                            file_types=[".mp4", ".avi", ".mov", ".mkv"],
                            type="filepath"
                        )

                    with gr.TabItem("🔗 视频链接"):
                        url_input = gr.Textbox(
                            label="视频URL",
                            placeholder="https://example.com/video.mp4",
                            lines=1
                        )

                gr.Markdown("### ⚙️ 检测参数")

                keywords_input = gr.Textbox(
                    label="自定义关键词（可选，用英文逗号分隔）",
                    placeholder="商标, logo, 侵权内容, 抄袭画面",
                    lines=2
                )

                with gr.Row():
                    sim_threshold = gr.Slider(
                        label="相似度阈值",
                        minimum=0.5,
                        maximum=0.95,
                        value=0.7,
                        step=0.05,
                        info="越高越严格，推荐0.7"
                    )

                    clip_threshold = gr.Slider(
                        label="语义阈值",
                        minimum=0.5,
                        maximum=0.95,
                        value=0.7,
                        step=0.05,
                        info="CLIP语义匹配阈值"
                    )

                detect_btn = gr.Button("🔍 开始检测", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空结果", variant="secondary")

            # 右侧：结果区域（使用 Tabs 组织）
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("📋 检测结果摘要"):
                        result_output = gr.Markdown("等待检测...")

                    with gr.TabItem("🎥 视频对比"):
                        video_output = gr.Video(label="并排对比视频", height=400)

                    # ========== 新增：帧级图片对比 Tab ==========
                    with gr.TabItem("🖼️ 帧级图片对比"):
                        gr.Markdown("### 🖼️ 相似帧图片对比")
                        gr.Markdown("以下展示待检测视频与库视频中匹配的关键帧画面")

                        frame_gallery = gr.Gallery(
                            label="帧对比结果",
                            columns=2,
                            rows=3,
                            height=500,
                            object_fit="contain",
                            show_label=True,
                            preview=True
                        )

                        gr.Markdown("""
                        <div style="text-align: center; color: #888; margin-top: 10px;">
                        <small>每张图片上方显示：匹配视频ID | 时间戳 | 相似度 | 语义关键词</small>
                        </div>
                        """)

                    with gr.TabItem("📄 侵权报告"):
                        report_output = gr.File(label="下载报告（JSON/TXT格式）")

        # 页脚信息
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #888;">
        ⚡ 基于 PyAV + Milvus + CLIP + Qwen2.5 构建 | 流式处理，不存储视频 | 支持本地文件与URL检测
        </div>
        """)

        # ========== 绑定检测事件（使用新方法，返回4个值） ==========
        detect_btn.click(
            fn=app.detect_video_with_frames,
            inputs=[video_input, url_input, keywords_input, sim_threshold, clip_threshold],
            outputs=[result_output, video_output, report_output, frame_gallery]
        ).then(
            fn=None,
            js="() => { setTimeout(() => { window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'}); }, 500); }"
        )

        # 清空事件
        clear_btn.click(
            fn=app.clear_outputs,
            inputs=[],
            outputs=[result_output, video_output, report_output, frame_gallery]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()

    # 启动服务
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 改为 True 可生成公网链接
        debug=False
    )
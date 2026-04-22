"""
主入口文件
"""
import argparse
import sys
from pathlib import Path
from loguru import logger
from pipeline.training_pipeline import TrainingPipeline
from pipeline.detection_pipeline import DetectionPipeline
from config.settings import RAW_VIDEO_DIR, REPORTS_DIR, COMPARISONS_DIR


# ========== 辅助函数 ==========
def read_urls_from_file(file_path: str) -> list:
    """从文件中读取 URL 列表（每行一个）"""
    urls = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 跳过空行和注释
                urls.append(line)
    return urls


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='视频侵权检测系统')
    parser.add_argument('command', choices=['train', 'detect', 'batch_detect'],
                        help='执行命令: train(训练), detect(检测单个), batch_detect(批量检测)')
    parser.add_argument('--input', '-i', type=str, help='输入视频文件或目录路径')
    parser.add_argument('--output', '-o', type=str, default=str(REPORTS_DIR),
                        help='输出目录 (默认: ./data/reports)')
    parser.add_argument('--milvus-host', type=str, default='localhost',
                        help='Milvus主机地址 (默认: localhost)')
    parser.add_argument('--milvus-port', type=str, default='19530',
                        help='Milvus端口 (默认: 19530)')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='相似度阈值 (默认: 0.7)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='返回前K个结果 (默认: 10)')
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='自定义检测关键词，用空格分隔 (例: --text "商标 logo 品牌")')
    parser.add_argument('--url', '-u', type=str, default=None,
                        help='视频 URL 地址（用于流式处理）')
    parser.add_argument('--urls', type=str, default=None,
                        help='包含 URL 列表的文本文件路径（每行一个 URL）')

    # ========== 新增：流式处理控制参数 ==========
    parser.add_argument('--no-streaming', action='store_true', default=False,
                        help='禁用流式处理（会将视频下载到本地）')
    parser.add_argument('--download-dir', type=str, default='./downloaded_videos',
                        help='当禁用流式处理时，视频下载目录 (默认: ./downloaded_videos)')

    args = parser.parse_args()

    try:
        # ========== 训练模式 ==========
        if args.command == 'train':
            logger.info("启动训练模式")

            # 初始化训练流水线（支持流式处理参数）
            pipeline = TrainingPipeline(
                milvus_host=args.milvus_host,
                milvus_port=args.milvus_port,
                use_streaming=not args.no_streaming,  # 默认启用流式处理，除非指定 --no-streaming
                download_dir=args.download_dir if args.no_streaming else None
            )

            # 根据流式处理模式显示提示
            if not args.no_streaming:
                logger.info("流式处理模式已启用: 视频将边下载边处理，处理完后自动删除，不会保存到磁盘")
            else:
                logger.info(f"本地下载模式: 视频将保存到 {args.download_dir}")

            # ========== 1. 从单个 URL 训练 ==========
            if args.url:
                logger.info(f"从 URL 训练: {args.url}")

                if not args.no_streaming:
                    # 流式处理模式
                    result = pipeline.train_from_url_streaming(args.url)
                    logger.info(f"URL 训练完成: 视频ID={result.get('video_id')}, 帧数={result.get('frame_count')}")
                else:
                    # 下载到本地模式
                    result = pipeline.train_from_url(args.url)
                    logger.info(f"URL 训练完成: 视频ID={result.get('video_id')}, 帧数={result.get('frame_count')}")

            # ========== 2. 从 URL 列表文件批量训练 ==========
            elif args.urls:
                logger.info(f"从文件批量训练 URL: {args.urls}")

                # 读取 URL 列表
                urls = read_urls_from_file(args.urls)
                logger.info(f"共 {len(urls)} 个 URL")

                if not urls:
                    logger.error("URL 列表为空")
                    sys.exit(1)

                # 批量训练
                if not args.no_streaming:
                    # 流式处理模式
                    result = pipeline.train_from_urls_batch_streaming(urls)
                else:
                    # 下载到本地模式
                    result = pipeline.train_from_urls_batch(urls)

                logger.info(f"批量训练完成: 成功={result['success']}, 失败={result['failed']}")

            # ========== 3. 从本地文件训练（原有逻辑） ==========
            elif args.input:
                input_path = Path(args.input)

                if input_path.is_file():
                    logger.info(f"训练本地视频文件: {args.input}")
                    result = pipeline.train_single_video(str(input_path))
                    logger.info(f"训练完成: {result}")

                elif input_path.is_dir():
                    logger.info(f"批量训练目录: {args.input}")
                    result = pipeline.train_from_directory(str(input_path))
                    logger.info(f"批量训练完成: 处理{result['processed']}个视频, 失败{result['failed']}个")

                else:
                    logger.error(f"输入路径不存在: {args.input}")
                    sys.exit(1)

            # ========== 4. 使用默认目录训练 ==========
            else:
                logger.info(f"使用默认目录训练: {RAW_VIDEO_DIR}")
                result = pipeline.train_from_directory(str(RAW_VIDEO_DIR))
                logger.info(f"训练完成: {result}")

            # 显示索引统计
            stats = pipeline.get_index_stats()
            logger.info(f"索引统计: {stats}")

        # ========== 检测模式 ==========
        elif args.command == 'detect':
            logger.info("启动检测模式")

            pipeline = DetectionPipeline(
                milvus_host=args.milvus_host,
                milvus_port=args.milvus_port
            )

            # 解析用户关键词
            user_keywords = None
            if args.text:
                user_keywords = args.text.split()
                logger.info(f"用户指定关键词: {user_keywords}")

            # ========== 1. 从 URL 检测（流式） ==========
            if args.url:
                logger.info(f"流式检测模式（不下载视频）: {args.url}")

                report = pipeline.detect_from_url(
                    args.url,
                    user_keywords=user_keywords,
                    top_k=args.top_k
                )

                # 输出检测结果
                logger.info(f"检测完成: 侵权等级={report.infringement_level}, "
                          f"匹配数量={report.total_matches}, "
                          f"语义匹配数={report.clip_summary.get('total_semantic_matches', 0)}")
                logger.info(f"报告已保存: {report.report_file_path}")

            # ========== 2. 从本地文件检测 ==========
            elif args.input:
                input_path = Path(args.input)

                if input_path.is_file():
                    logger.info(f"检测单个视频: {args.input}")

                    report = pipeline.detect_infringement(
                        str(input_path),
                        user_keywords=user_keywords,
                        top_k=args.top_k,
                        similarity_threshold=args.threshold
                    )

                    # 输出检测结果
                    logger.info(f"检测完成: 侵权等级={report.infringement_level}, "
                              f"匹配数量={report.total_matches}, "
                              f"语义匹配数={report.clip_summary.get('total_semantic_matches', 0)}")
                    logger.info(f"报告已保存: {report.report_file_path}")

                elif input_path.is_dir():
                    logger.info(f"批量检测目录: {args.input}")

                    from utils.file_utils import list_video_files
                    video_files = list_video_files(str(input_path))

                    if not video_files:
                        logger.warning(f"目录中没有找到视频文件: {input_path}")
                        return

                    logger.info(f"找到 {len(video_files)} 个视频文件")

                    for video_file in video_files:
                        try:
                            logger.info(f"检测视频: {video_file.name}")
                            report = pipeline.detect_infringement(
                                str(video_file),
                                user_keywords=user_keywords,
                                top_k=args.top_k,
                                similarity_threshold=args.threshold
                            )
                            logger.info(f"完成: {video_file.name} - {report.infringement_level}")
                        except Exception as e:
                            logger.error(f"检测失败 {video_file}: {e}")

                    return  # 批量检测完成后直接返回

                else:
                    logger.error(f"路径不存在: {args.input}")
                    sys.exit(1)

            else:
                logger.error("请指定要检测的视频文件 (--input) 或 URL (--url)")
                sys.exit(1)

        # ========== 批量检测模式 ==========
        elif args.command == 'batch_detect':

            if not args.input:
                logger.error("请指定包含视频文件的目录: --input <directory>")
                sys.exit(1)

            input_dir = Path(args.input)

            if not input_dir.is_dir():
                logger.error(f"输入路径不是目录: {args.input}")
                sys.exit(1)

            logger.info(f"启动批量检测模式: {input_dir}")

            pipeline = DetectionPipeline(
                milvus_host=args.milvus_host,
                milvus_port=args.milvus_port
            )

            # 解析用户关键词
            user_keywords = None
            if args.text:
                user_keywords = args.text.split()
                logger.info(f"批量检测使用关键词: {user_keywords}")

            # 获取所有视频文件
            from utils.file_utils import list_video_files
            video_files = list_video_files(str(input_dir))

            if not video_files:
                logger.warning(f"目录中没有找到视频文件: {input_dir}")
                return

            logger.info(f"找到 {len(video_files)} 个视频文件")

            # 批量检测
            reports = []
            for video_file in video_files:
                try:
                    logger.info(f"检测视频: {video_file}")
                    report = pipeline.detect_infringement(
                        str(video_file),
                        user_keywords=user_keywords,
                        top_k=args.top_k,
                        similarity_threshold=args.threshold
                    )
                    reports.append(report)
                    logger.info(f"完成: {video_file.name} - {report.infringement_level}")
                except Exception as e:
                    logger.error(f"检测失败 {video_file}: {e}")

            # 输出汇总信息
            logger.info("=" * 50)
            logger.info("批量检测完成")
            logger.info(f"总计: {len(reports)}/{len(video_files)} 个视频完成检测")

            infringement_counts = {
                'high': sum(1 for r in reports if r.infringement_level == 'high'),
                'medium': sum(1 for r in reports if r.infringement_level == 'medium'),
                'low': sum(1 for r in reports if r.infringement_level == 'low'),
                'none': sum(1 for r in reports if r.infringement_level == 'none')
            }

            for level, count in infringement_counts.items():
                logger.info(f"{level}侵权: {count}")

            logger.info(f"所有报告保存在: {REPORTS_DIR}")

    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
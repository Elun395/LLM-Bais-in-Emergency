from __future__ import annotations

import argparse
import os
from pathlib import Path

# Ensure headless plotting works in CI/servers
os.environ.setdefault("MPLBACKEND", "Agg")

from .phase1_word_assoc import run_phase1, run_phase1_with_templates, plot_phase1
from .phase2_relative_decision import run_phase2_with_templates, plot_phase2_by_context, plot_phase2_template_by_context, plot_phase2_combined, plot_phase2_contexts_combined
from .phase3_bureaucratic_decision import run_phase3_with_templates, plot_phase3_by_context, plot_phase3_template_by_context, plot_phase3_combined, plot_phase3_contexts_combined
from .phase4_official_role import run_phase4_with_templates, plot_phase4_by_context, plot_phase4_template_by_context, plot_phase4_combined, plot_phase4_contexts_combined
# Removed import for phase3_persona since it doesn't exist
from .utils import save_json


def parse_args():
    p = argparse.ArgumentParser(description="LLM Bias Measurement Toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Phase 1
    p1 = sub.add_parser("phase1", help="Run Word Association Test bias")
    p1.add_argument("--data", default="data/data1.json", help="Path to data1.json")
    p1.add_argument("--model", default="gpt-4o-mini")
    p1.add_argument("--trials", type=int, default=5)
    p1.add_argument("--per", type=int, default=6, help="Attributes per group (X/Y)")
    p1.add_argument("--seed", type=int, default=42)
    p1.add_argument("--mock", action="store_true", help="Use offline mock (no API)")
    p1.add_argument("--out-json", default="outputs/phase1_results.json")
    p1.add_argument("--out-png", default="outputs/phase1_bias.png")
    p1.add_argument("--responses-dir", default="outputs/responses")
    p1.add_argument("--api-key", default=None, help="API key (overrides OPENAI_API_KEY env)")
    p1.add_argument("--api-base", default=None, help="Custom OpenAI-compatible base URL")

    # Phase 2
    p2 = sub.add_parser("phase2", help="Run decision bias test")
    p2.add_argument("--data", default="data/data1.json", help="Path to data1.json")
    p2.add_argument("--model", default="gpt-4o-mini")
    p2.add_argument("--trials", type=int, default=5)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--mock", action="store_true")
    p2.add_argument("--out-json", default="outputs/phase2_results.json")
    p2.add_argument("--out-png-context", default="outputs/phase2_bias_by_context.png", 
                   help="Output path for context-separated plot")
    p2.add_argument("--out-png-template1", default="outputs/phase2_bias_template1.png",
                   help="Output path for template 1 plot")
    p2.add_argument("--out-png-template2", default="outputs/phase2_bias_template2.png",
                   help="Output path for template 2 plot")
    p2.add_argument("--out-png-template3", default="outputs/phase2_bias_template3.png",
                   help="Output path for template 3 plot")
    p2.add_argument("--out-png-combined", default="outputs/phase2_bias_combined.png",
                   help="Output path for combined plot")
    p2.add_argument("--out-png-contexts-combined", default="outputs/phase2_bias_contexts_combined.png",
                   help="Output path for contexts combined templates plot")
    p2.add_argument("--responses-dir", default="outputs/responses")
    p2.add_argument("--api-key", default=None)
    p2.add_argument("--api-base", default=None)

    # Phase 3
    p3 = sub.add_parser("phase3", help="Run bureaucratic decision bias test")
    p3.add_argument("--data", default="data/data1.json", help="Path to data1.json")
    p3.add_argument("--model", default="gpt-4o-mini")
    p3.add_argument("--trials", type=int, default=5)
    p3.add_argument("--seed", type=int, default=42)
    p3.add_argument("--mock", action="store_true")
    p3.add_argument("--out-json", default="outputs/phase3_results.json")
    p3.add_argument("--out-png-context", default="outputs/phase3_bias_by_context.png", 
                   help="Output path for context-separated plot")
    p3.add_argument("--out-png-template1", default="outputs/phase3_bias_template1.png",
                   help="Output path for template 1 plot")
    p3.add_argument("--out-png-template2", default="outputs/phase3_bias_template2.png",
                   help="Output path for template 2 plot")
    p3.add_argument("--out-png-template3", default="outputs/phase3_bias_template3.png",
                   help="Output path for template 3 plot")
    p3.add_argument("--out-png-combined", default="outputs/phase3_bias_combined.png",
                   help="Output path for combined plot")
    p3.add_argument("--out-png-contexts-combined", default="outputs/phase3_bias_contexts_combined.png",
                   help="Output path for contexts combined templates plot")
    p3.add_argument("--responses-dir", default="outputs/responses")
    p3.add_argument("--api-key", default=None)
    p3.add_argument("--api-base", default=None)

    # Phase 4
    p4 = sub.add_parser("phase4", help="Run official role decision bias test")
    p4.add_argument("--data", default="data/data1.json", help="Path to data1.json")
    p4.add_argument("--model", default="gpt-4o-mini")
    p4.add_argument("--trials", type=int, default=5)
    p4.add_argument("--seed", type=int, default=42)
    p4.add_argument("--mock", action="store_true")
    p4.add_argument("--out-json", default="outputs/phase4_results.json")
    p4.add_argument("--out-png-context", default="outputs/phase4_bias_by_context.png", 
                   help="Output path for context-separated plot")
    p4.add_argument("--out-png-template1", default="outputs/phase4_bias_template1.png",
                   help="Output path for template 1 plot")
    p4.add_argument("--out-png-template2", default="outputs/phase4_bias_template2.png",
                   help="Output path for template 2 plot")
    p4.add_argument("--out-png-template3", default="outputs/phase4_bias_template3.png",
                   help="Output path for template 3 plot")
    p4.add_argument("--out-png-combined", default="outputs/phase4_bias_combined.png",
                   help="Output path for combined plot")
    p4.add_argument("--out-png-variant", default="outputs/phase4_bias_by_variant.png",
                   help="Output path for variant-separated plot")
    p4.add_argument("--out-png-contexts-combined", default="outputs/phase4_bias_contexts_combined.png",
                   help="Output path for contexts combined templates plot")
    p4.add_argument("--responses-dir", default="outputs/responses")
    p4.add_argument("--api-key", default=None)
    p4.add_argument("--api-base", default=None)

    return p.parse_args()


def main():
    args = parse_args()
    Path("outputs").mkdir(parents=True, exist_ok=True)
    responses_dir = Path(getattr(args, "responses_dir", "outputs/responses"))
    responses_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "phase1":
        # 调用 run_phase1_with_templates 获取所有模板结果
        template_results, combined_results = run_phase1_with_templates(
            args.data,
            model=args.model,
            trials=args.trials,
            n_per_group=args.per,
            seed=args.seed,
            mock=args.mock,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        # 保存每个模板的结果
        for template_id in range(1, 4):
            template_output_path = f"outputs/phase1_template{template_id}_results.json"
            save_json(template_results[f"template_{template_id}"]["rows"], template_output_path)
            print(f"Saved Template {template_id} results to {template_output_path}")

        # 保存合并后的综合结果
        save_json(combined_results["combined_rows"], args.out_json)
        print(f"Saved combined results to {args.out_json}")

        # 保存响应日志
        save_json(combined_results["combined_responses"], responses_dir / "phase1_responses.json")
        print(f"Saved responses to {responses_dir / 'phase1_responses.json'}")

        # 绘制每个模板的结果图
        for template_id in range(1, 4):
            template_plot_path = f"outputs/phase1_template{template_id}_bias.png"
            plot_phase1(template_results[f"template_{template_id}"]["rows"], out_path=template_plot_path)
            print(f"Saved Template {template_id} plot to {template_plot_path}")

        # 绘制合并后的综合结果图
        plot_phase1(combined_results["combined_rows"], out_path=args.out_png)
        print(f"Saved combined plot to {args.out_png}")
    elif args.cmd == "phase2":
        # 调用 run_phase2_with_templates 获取所有模板结果
        template_results, combined_results = run_phase2_with_templates(
            args.data,
            model=args.model,
            trials=args.trials,
            seed=args.seed,
            mock=args.mock,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        # 保存每个模板的结果
        for template_id in range(1, 4):
            template_output_path = f"outputs/phase2_template{template_id}_results.json"
            save_json(template_results[f"template_{template_id}"]["rows"], template_output_path)
            print(f"Saved Template {template_id} results to {template_output_path}")

        # 保存合并后的综合结果
        save_json(combined_results["combined_rows"], args.out_json)
        print(f"Saved combined results to {args.out_json}")

        # 保存响应日志
        save_json(combined_results["combined_responses"], responses_dir / "phase2_responses.json")
        print(f"Saved responses to {responses_dir / 'phase2_responses.json'}")

        # 生成三种情境下，基于每种提示词变体各自的结果图（3张）
        for template_id in range(1, 4):
            template_results_data = template_results[f"template_{template_id}"]["rows"]
            template_plot_path = getattr(args, f"out_png_template{template_id}")
            plot_phase2_template_by_context(template_results_data, template_id, out_path=template_plot_path)
            print(f"Saved template {template_id} plot to {template_plot_path}")
        
        # 生成三种情境下，三种提示词变体合并的结果图
        plot_phase2_contexts_combined(combined_results["combined_rows"], out_path=args.out_png_contexts_combined)
        print(f"Saved contexts combined templates plot to {args.out_png_contexts_combined}")
        
        # 生成综合所有提示词和情境的结果图
        plot_phase2_combined(combined_results["combined_rows"], out_path=args.out_png_combined)
        print(f"Saved combined plot to {args.out_png_combined}")
    elif args.cmd == "phase3":
        # 调用 run_phase3_with_templates 获取所有模板结果
        template_results, combined_results = run_phase3_with_templates(
            args.data,
            model=args.model,
            trials=args.trials,
            seed=args.seed,
            mock=args.mock,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        # 保存每个模板的结果
        for template_id in range(1, 4):
            template_output_path = f"outputs/phase3_template{template_id}_results.json"
            save_json(template_results[f"template_{template_id}"]["rows"], template_output_path)
            print(f"Saved Template {template_id} results to {template_output_path}")

        # 保存合并后的综合结果
        save_json(combined_results["combined_rows"], args.out_json)
        print(f"Saved combined results to {args.out_json}")

        # 保存响应日志
        save_json(combined_results["combined_responses"], responses_dir / "phase3_responses.json")
        print(f"Saved responses to {responses_dir / 'phase3_responses.json'}")

        # 生成三种情境下，基于每种提示词变体各自的结果图（3张）
        for template_id in range(1, 4):
            template_results_data = template_results[f"template_{template_id}"]["rows"]
            template_plot_path = getattr(args, f"out_png_template{template_id}")
            plot_phase3_template_by_context(template_results_data, template_id, out_path=template_plot_path)
            print(f"Saved template {template_id} plot to {template_plot_path}")
        
        # 生成三种情境下，三种提示词变体合并的结果图
        plot_phase3_contexts_combined(combined_results["combined_rows"], out_path=args.out_png_contexts_combined)
        print(f"Saved contexts combined templates plot to {args.out_png_contexts_combined}")
        
        # 生成综合所有提示词和情境的结果图
        plot_phase3_combined(combined_results["combined_rows"], out_path=args.out_png_combined)
        print(f"Saved combined plot to {args.out_png_combined}")
    elif args.cmd == "phase4":
        # 调用 run_phase4_with_templates 获取所有模板结果
        template_results, combined_results = run_phase4_with_templates(
            args.data,
            model=args.model,
            trials=args.trials,
            seed=args.seed,
            mock=args.mock,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        # 保存每个模板的结果
        for template_id in range(1, 4):
            template_output_path = f"outputs/phase4_template{template_id}_results.json"
            save_json(template_results[f"template_{template_id}"]["rows"], template_output_path)
            print(f"Saved Template {template_id} results to {template_output_path}")

        # 保存合并后的综合结果
        save_json(combined_results["combined_rows"], args.out_json)
        print(f"Saved combined results to {args.out_json}")

        # 保存响应日志
        save_json(combined_results["combined_responses"], responses_dir / "phase4_responses.json")
        print(f"Saved responses to {responses_dir / 'phase4_responses.json'}")

        # 生成三种情境下，基于每种提示词变体各自的结果图（3张）
        for template_id in range(1, 4):
            template_results_data = template_results[f"template_{template_id}"]["rows"]
            template_plot_path = getattr(args, f"out_png_template{template_id}")
            plot_phase4_template_by_context(template_results_data, template_id, out_path=template_plot_path)
            print(f"Saved template {template_id} plot to {template_plot_path}")
        
        # 生成三种情境下，三种提示词变体合并的结果图（1张）
        plot_phase4_contexts_combined(combined_results["combined_rows"], out_path=args.out_png_contexts_combined)
        print(f"Saved contexts combined templates plot to {args.out_png_contexts_combined}")
        
        # 生成综合所有提示词和情境的结果图（1张）
        plot_phase4_combined(combined_results["combined_rows"], out_path=args.out_png_combined)
        print(f"Saved combined plot to {args.out_png_combined}")

if __name__ == "__main__":
    main()
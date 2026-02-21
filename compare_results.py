import json

# Load STITCH result
with open(r'D:\workspace\contextual-intent\stitch_output\completed-qwenplus-traj-8\answer_evaluation\answer_evaluation_v1.json') as f:
    stitch_data = json.load(f)

# Load Vanilla RAG result
with open(r'D:\workspace\contextual-intent\stitch_output\completed-qwenplus-traj-8\answer_evaluation\answer_evaluation_v1_vanilla_rag.json') as f:
    vanilla_data = json.load(f)

def calculate_metrics(data):
    results = data.get('question_answer_evaluation_results', [])
    if not results:
        return None, None, None, 0

    total_p = sum(r.get('precision', 0) for r in results)
    total_r = sum(r.get('recall', 0) for r in results)
    total_f1 = sum(r.get('f1', 0) for r in results)
    n = len(results)

    return total_p/n, total_r/n, total_f1/n, n

stitch_p, stitch_r, stitch_f1, stitch_n = calculate_metrics(stitch_data)
vanilla_p, vanilla_r, vanilla_f1, vanilla_n = calculate_metrics(vanilla_data)

print("=" * 80)
print("STITCH vs Vanilla RAG 性能对比 (traj-8 数据集, 6 个问题)")
print("=" * 80)
print()
print("【STITCH】标签过滤 + 语义检索 (Steps 1-5):")
print(f"  问题数量:        {stitch_n}")
print(f"  Macro Precision: {stitch_p:.4f}")
print(f"  Macro Recall:    {stitch_r:.4f}")
print(f"  Macro F1:        {stitch_f1:.4f}")
print()
print("【Vanilla RAG】仅语义相似性检索 (Step 5):")
print(f"  问题数量:        {vanilla_n}")
print(f"  Macro Precision: {vanilla_p:.4f}")
print(f"  Macro Recall:    {vanilla_r:.4f}")
print(f"  Macro F1:        {vanilla_f1:.4f}")
print()
print("=" * 80)
print("性能差异 (Vanilla RAG - STITCH):")
print(f"  ΔPrecision: {vanilla_p - stitch_p:+.4f} ({(vanilla_p/stitch_p - 1)*100:+.2f}%)")
print(f"  ΔRecall:    {vanilla_r - stitch_r:+.4f} ({(vanilla_r/stitch_r - 1)*100:+.2f}%)")
print(f"  ΔF1:        {vanilla_f1 - stitch_f1:+.4f} ({(vanilla_f1/stitch_f1 - 1)*100:+.2f}%)")
print("=" * 80)
print()

if vanilla_f1 > stitch_f1:
    print("⚠️  意外结果：Vanilla RAG 的 F1 分数高于 STITCH！")
    print()
    print("可能原因分析：")
    print("1. 数据集规模小 (traj-8 仅 62 turns)，标签过滤可能过于激进")
    print("2. LLM 标签选择可能误过滤掉相关上下文")
    print("3. 对于这个特定任务，语义相似性检索已足够")
    print("4. traj-8 数据集特性可能特别适合稠密检索")
    print()
    print("建议：")
    print("- 在更大数据集 (Medium/Large) 上验证，STITCH 优势应更明显")
    print("- 检查 STITCH 检索结果中的标签过滤是否过于严格")
    print("- 分析具体问题的检索差异（哪些问题 Vanilla RAG 表现更好）")
elif vanilla_f1 < stitch_f1:
    improvement = stitch_f1 - vanilla_f1
    improvement_pct = (stitch_f1 / vanilla_f1 - 1) * 100
    print(f"✅ 符合预期：STITCH F1 分数高于 Vanilla RAG！")
    print()
    print(f"标签过滤收益：F1 提升 {improvement:.4f} ({improvement_pct:+.2f}%)")
    print()
    print("结论：")
    print("- STITCH 的标签过滤策略 (Steps 1-4) 有效提升了检索质量")
    print("- 结构化标注 + 标签检索 优于 纯语义相似性检索")
    print("- 验证了论文的核心贡献")
else:
    print("📊 两种方法性能相当 (F1 分数相同)")
    print()
    print("可能原因：")
    print("- 数据集太小，差异不明显")
    print("- 标签过滤和语义检索在此数据集上效果相似")

print()
print("=" * 80)
print("详细问题级别对比：")
print("=" * 80)

stitch_results = stitch_data.get('question_answer_evaluation_results', [])
vanilla_results = vanilla_data.get('question_answer_evaluation_results', [])

for i, (s, v) in enumerate(zip(stitch_results, vanilla_results), 1):
    s_f1 = s.get('f1', 0)
    v_f1 = v.get('f1', 0)
    diff = v_f1 - s_f1
    winner = "Vanilla" if v_f1 > s_f1 else ("STITCH" if s_f1 > v_f1 else "平局")

    question_content = s.get('question_answer_generation_result', {}).get('question', {}).get('content', 'N/A')
    question_short = question_content[:60] + "..." if len(question_content) > 60 else question_content

    print(f"\n问题 {i}: {question_short}")
    print(f"  STITCH F1:     {s_f1:.4f}")
    print(f"  Vanilla RAG F1: {v_f1:.4f}")
    print(f"  差异:          {diff:+.4f} (胜者: {winner})")

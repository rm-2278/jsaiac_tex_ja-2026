#!/bin/bash

# 2分間研究概要プレゼンテーション作成スクリプト
# Usage: ./create_presentation.sh

echo "=== 階層的世界モデル研究概要プレゼンテーション作成 ==="

# LaTeX Beamerプレゼンテーションをコンパイル
echo "Step 1: Beamerプレゼンテーションをコンパイル中..."
platex presentation-slides.tex
platex presentation-slides.tex  # 参照解決のため2回実行
dvipdfmx presentation-slides.dvi

if [ $? -eq 0 ]; then
    echo "✓ プレゼンテーションPDFが生成されました: presentation-slides.pdf"
else
    echo "✗ プレゼンテーションのコンパイルに失敗しました"
    exit 1
fi

# 実験動画ファイルのリストを作成
echo -e "\nStep 2: 利用可能な実験動画ファイル:"
echo "=== サブゴール可視化動画 ==="
find media/videos/report -name "*.gif" | head -5 | while read file; do
    echo "  - $file"
done

echo -e "\n=== 方策行動動画 ==="
find media/videos/train_stats -name "*.gif" | head -5 | while read file; do
    echo "  - $file"
done

echo -e "\nStep 3: プレゼンテーション推奨構成（2分間）:"
echo "  1. タイトル + 背景説明 (20秒)"
echo "  2. 研究目的と手法 (20秒)" 
echo "  3. 実験環境説明 (20秒)"
echo "  4. 主要発見の説明 (30秒)"
echo "  5. 内部状態可視化 (30秒)"
echo "  6. 数式と理論検証 (10秒)"
echo "  7. 結論と将来展望 (10秒)"

echo -e "\nStep 4: 動画作成の提案:"
echo "  方法1: プレゼンテーションPDFをスライド録画"
echo "  方法2: OBSなどで画面録画しながらGIFアニメーションを再生"
echo "  方法3: FFmpegで静的スライドと動的GIFを組み合わせ"

echo -e "\n=== プレゼンテーション完了 ==="
echo "ファイル: presentation-slides.pdf"
echo "次のステップ: PDFを開いて録画、または動画編集ソフトで動的要素を追加"